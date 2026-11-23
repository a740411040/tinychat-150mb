"""Frozen data identity and optimizer-boundary checkpoint storage for stage 1."""

from contextlib import contextmanager, suppress
import json
import math
import os
from pathlib import Path
import random
import re
from tempfile import NamedTemporaryFile
from uuid import uuid4

import numpy as np
import torch

from .chat_format import SPECIAL_IDS


SPLITS = ("train", "validation", "test")
CHECKPOINT_SCHEMA = 1


def _reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def load_json(value):
    return json.loads(value, parse_constant=_reject_constant)


def read_json(path):
    return load_json(path.read_text(encoding="utf-8"))


def require_positive_integer(value, field):
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def file_identity(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_frozen_manifest(data_dir, stage, vocab_size):
    manifest_path = (data_dir / "manifest.json").resolve()
    manifest_file = file_identity(manifest_path)
    manifest = read_json(manifest_path)
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["frozen"] is not True):
        raise ValueError(f"Expected a frozen stage-1 manifest at {manifest_path}")
    if stage == "pretrain" and manifest["corpus_id"] not in {"stage1_v1", "pretrain_clean_v2"}:
        raise ValueError("Pretraining requires an explicitly supported frozen corpus")
    if (type(manifest["vocab_size"]) is not int or manifest["vocab_size"] != vocab_size
            or manifest["special_token_ids"] != SPECIAL_IDS):
        raise ValueError("Manifest vocabulary or special token IDs do not match stage 1")
    tokenizer_hash = manifest["tokenizer_sha256"]
    if not isinstance(tokenizer_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", tokenizer_hash):
        raise ValueError("Manifest tokenizer_sha256 must be a lowercase SHA-256 digest")
    tokenizer_path = Path(manifest["tokenizer_path"])
    if not tokenizer_path.is_absolute():
        raise ValueError("Manifest tokenizer_path must be absolute")
    tokenizer_path = tokenizer_path.resolve()
    paths = {"tokenizer.json": tokenizer_path}
    for split in SPLITS:
        entry = manifest["splits"][split]
        documents = require_positive_integer(entry["documents"], f"{split}.documents")
        tokens = require_positive_integer(entry["tokens"], f"{split}.tokens")
        jsonl_name = f"{split}.jsonl"
        paths[jsonl_name] = data_dir / jsonl_name
        if stage == "pretrain":
            binary_name = f"{split}.bin"
            require_positive_integer(entry["binary_bytes"], f"{split}.binary_bytes")
            if tokens < 2 or entry["binary_bytes"] != 2 * tokens:
                raise ValueError(f"Invalid manifest token count and binary size for {split}")
            paths[binary_name] = data_dir / binary_name
        else:
            assistant_tokens = require_positive_integer(entry["assistant_tokens"], f"{split}.assistant_tokens")
            if assistant_tokens >= tokens or manifest["counts"][split] != documents:
                raise ValueError(f"Invalid SFT manifest counts for {split}")

    files = {}
    for name, path in paths.items():
        observed = file_identity(path)
        if not observed["bytes"] or observed != manifest["files"][name]:
            raise ValueError(f"Frozen file identity changed: {path}")
        if path.suffix == ".bin" and observed["bytes"] != manifest["splits"][path.stem]["binary_bytes"]:
            raise ValueError(f"Frozen binary size does not match manifest: {path}")
        files[name] = {"path": str(path.resolve()), **observed}

    if file_identity(manifest_path) != manifest_file:
        raise ValueError(f"Manifest changed while being read: {manifest_path}")
    return manifest, {
        "manifest": {"path": str(manifest_path), **manifest_file},
        "files": files,
        "tokenizer_sha256": tokenizer_hash,
    }


def verify_file_identity(data_identity):
    for entry in [data_identity["manifest"], *data_identity["files"].values()]:
        recorded = {"bytes": entry["bytes"], "mtime_ns": entry["mtime_ns"]}
        if file_identity(Path(entry["path"])) != recorded:
            raise ValueError(f"Frozen file identity changed: {entry['path']}")


def capture_rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _atomic_file(path, write, mode):
    encoding = None if "b" in mode else "utf-8"
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                            delete=False, mode=mode, encoding=encoding) as file:
        temporary = Path(file.name)
        try:
            write(file)
            file.flush()
            os.fsync(file.fileno())
            file.close()
            os.replace(temporary, path)
        finally:
            with suppress(OSError):
                file.close()
                temporary.unlink(missing_ok=True)


def atomic_json(path, data):
    payload = json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _atomic_file(path, lambda file: file.write(payload), "w")


def write_checkpoint(run_dir, step, state):
    path = run_dir / f"step_{step:08d}.pt"
    if path.exists():
        path = run_dir / f"step_{step:08d}_{uuid4().hex}.pt"
    _atomic_file(path, lambda file: torch.save(state, file), "wb")
    publish_latest(run_dir, path, step)
    return path


def publish_latest(run_dir, checkpoint, step):
    atomic_json(run_dir / "latest.json", {"checkpoint": str(checkpoint.resolve()), "step": step})


def resolve_resume(run_dir, requested):
    latest_path = run_dir / "latest.json"
    if requested is None:
        if latest_path.exists() or (run_dir / "steps.jsonl").exists() or any(run_dir.glob("step_*.pt")):
            raise ValueError(f"Run already has training state. Resume with --resume {latest_path}")
        return None
    requested = requested.resolve()
    if requested.suffix == ".json":
        requested = Path(read_json(requested)["checkpoint"]).resolve()
    if latest_path.exists() and requested != Path(read_json(latest_path)["checkpoint"]).resolve():
        raise ValueError(f"Resume must use the latest committed checkpoint for {run_dir}")
    return requested


@contextmanager
def lock_run(run_dir):
    with (run_dir / ".training.lock").open("a+b") as file:
        file.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"Run directory is already locked: {run_dir}") from error
        else:
            import fcntl

            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Run directory is already locked: {run_dir}") from error
        yield
