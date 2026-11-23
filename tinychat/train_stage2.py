"""Train the approved Stage 2 pilots, stable/decay branches, or one SFT epoch."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
import math
import os
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F

from .chat_format import IGNORE_LABEL, SPECIAL_IDS, manifest_final_eos
from .inference import checked_tokenizer
from .model import ModelConfig, TinyLM
from .stage2_runtime import DiskBudgetError, ensure_disk_budget
from .train_stage1 import (
    PretrainStream, Progress, Schedule, emit, handle_shutdown, loss_sum, optimizer_for,
    positive_argument, read_sft_data, seed_everything, sft_batches, training_device,
)
from .training_state import (
    CHECKPOINT_SCHEMA as LEGACY_CHECKPOINT_SCHEMA,
    SPLITS, _atomic_file, atomic_json, capture_rng, file_identity, lock_run,
    read_frozen_manifest, read_json, require_positive_integer, restore_rng, verify_file_identity,
)


CHECKPOINT_SCHEMA = 2
PRETRAIN_CORPUS = "pretrain_stage2_500m"
PILOT_UPDATES = 153
PILOT_WARMUP_UPDATES = 20
WARMUP_TOKENS = 5_000_000
STABLE_TOKENS = 450_000_000
FINAL_TOKENS = 500_000_000
LEGACY_BASE_TOKENS = 100_000_000
MILESTONES = (100_000_000, 250_000_000)
VALIDATION_TOKENS = 131_072
FINAL_VALIDATION_TOKENS = 1_000_000
CHECKPOINT_OVERHEAD = 4 * 1024 * 1024
PRETRAIN_LRS = (6e-4, 1e-3)
SFT_LR = 6e-5


@dataclass(frozen=True)
class WSD:
    unit: str
    warmup: int
    stable_until: int
    end: int
    peak: float

    def learning_rate(self, completed):
        if completed <= self.warmup:
            return self.peak * completed / self.warmup
        if completed <= self.stable_until:
            return self.peak
        if completed >= self.end:
            return 0.0
        fraction = (completed - self.stable_until) / (self.end - self.stable_until)
        return self.peak * (1 + math.cos(math.pi * fraction)) / 2


def pretrain_schedule(phase, lr):
    if phase == "pilot":
        return WSD("updates", PILOT_WARMUP_UPDATES, PILOT_UPDATES, PILOT_UPDATES, lr)
    return WSD("tokens", WARMUP_TOKENS, STABLE_TOKENS,
               FINAL_TOKENS if phase == "decay" else STABLE_TOKENS, lr)


def read_pretrain_manifest(data_dir, vocab_size):
    manifest_path = data_dir / "manifest.json"
    manifest_stat = file_identity(manifest_path)
    manifest = read_json(manifest_path)
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["frozen"] is not True or manifest["corpus_id"] != PRETRAIN_CORPUS
            or manifest["dtype"] != "<u2" or type(manifest["vocab_size"]) is not int
            or manifest["vocab_size"] != vocab_size or manifest["special_token_ids"] != SPECIAL_IDS):
        raise ValueError("Expected a compatible frozen Stage 2 pretraining manifest")

    tokenizer_path = Path(manifest["tokenizer_path"])
    if not tokenizer_path.is_absolute():
        raise ValueError("Frozen tokenizer path must be absolute")
    paths = {"tokenizer.json": tokenizer_path}
    binary_sizes = {}
    for split in SPLITS:
        paths[f"{split}.jsonl"] = data_dir / f"{split}.jsonl"
        name = f"{split}.bin"
        paths[name] = data_dir / name
        binary_sizes[name] = binary_size(manifest["splits"][split], split)
    general = manifest["stage2"]["general_validation"]
    if general["path"] != "general_validation.bin":
        raise ValueError("Expected the fixed general_validation.bin stream")
    paths[general["path"]] = data_dir / general["path"]
    binary_sizes[general["path"]] = binary_size(general, "general_validation")

    files = {}
    for name, path in paths.items():
        observed = file_identity(path)
        if (not observed["bytes"] or observed != manifest["files"][name]
                or observed["bytes"] != binary_sizes.get(name, observed["bytes"])):
            raise ValueError(f"Frozen file identity changed: {path}")
        files[name] = {"path": str(path.resolve()), **observed}
    identity = {
        "manifest": {"path": str(manifest_path.resolve()), **manifest_stat},
        "files": files, "tokenizer_sha256": manifest["tokenizer_sha256"],
    }
    verify_file_identity(identity)
    return manifest, identity


def binary_size(entry, name):
    require_positive_integer(entry["documents"], f"{name}.documents")
    tokens = require_positive_integer(entry["tokens"], f"{name}.tokens")
    size = require_positive_integer(entry["binary_bytes"], f"{name}.binary_bytes")
    if tokens < 2 or size != 2 * tokens:
        raise ValueError(f"Invalid manifest token count and binary size for {name}")
    return size


def masked_token_ids(manifest):
    train = manifest["splits"]["train"]
    counts = train["special_token_counts"]
    if (set(counts) != set(SPECIAL_IDS) or any(type(count) is not int or count < 0 for count in counts.values())
            or counts["<|eos|>"] == 0 or sum(counts.values()) > train["tokens"]):
        raise ValueError("Expected actual counts for all frozen special tokens, including EOS")
    return tuple(token_id for token, token_id in SPECIAL_IDS.items() if not counts[token])


def pretrain_loss_sum(model, batch, device, *, masked, excluded):
    # The stream's artificial right padding is paired with IGNORE_LABEL.
    valid = batch.labels.ne(IGNORE_LABEL)
    if ((excluded[batch.inputs] | excluded[batch.labels.clamp_min(0)]) & valid).any():
        raise ValueError("Frozen pretraining stream contains a masked control token")
    inputs, labels = batch.inputs.to(device), batch.labels.to(device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with autocast:
        logits = model(inputs).index_fill_(-1, masked, -torch.inf)
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum",
                               ignore_index=IGNORE_LABEL)


def evaluate(model, batches, device, compute_loss):
    previous_mode = model.training
    model.eval()
    losses, targets = torch.zeros((), device=device), 0
    try:
        with torch.inference_mode():
            for batch in batches:
                losses += compute_loss(model, batch, device)
                targets += batch.targets
        loss = losses.item() / targets
        return {"loss": loss, "valid_loss_tokens": targets, "nonfinite": not math.isfinite(loss)}
    finally:
        model.train(previous_mode)


def bucket_order(rows, seed, batch_size):
    rng = np.random.default_rng(seed)
    ordered = sorted(rng.permutation(len(rows)).tolist(), key=lambda index: len(rows[index][0]))
    batches = [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
    tail = batches.pop() if len(batches[-1]) < batch_size else []
    return [index for batch in rng.permutation(len(batches)) for index in batches[batch]] + tail


def tail_padding(count, micro_batch, sequence_length):
    last_micro = count % (micro_batch * sequence_length)
    tail = last_micro % sequence_length
    return sequence_length - tail if last_micro > sequence_length and tail else 0


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def finite_validation(state, key):
    result = state["validations"].get(key)
    if result is None or result["step"] != state["progress"]["updates"]:
        return False
    scopes = {"sft"} if state["stage"] == "sft" else {"general", "mixed"}
    if key == "final" and state["settings"]["phase"] in {"stable", "decay"}:
        scopes.add("general_full_vocab")
    return set(result["scopes"]) == scopes and all(
        value["nonfinite"] is False and math.isfinite(value["loss"]) for value in result["scopes"].values()
    )


def completed_checkpoint(state, phase):
    settings, progress = state["settings"], state["progress"]
    return (settings["phase"] == phase and state["completed"] is True
            and progress["tokens"] == settings["planned_tokens"]
            and progress["updates"] == settings["planned_updates"]
            and progress["examples"] == settings["planned_examples"] and finite_validation(state, "final"))


def role_checkpoint(latest, role):
    """Publish a durable role without changing the committed latest checkpoint."""
    if role.exists() and role.samefile(latest):
        return
    temporary = role.with_name(f".{role.name}.{uuid4().hex}.tmp")
    try:
        try:
            os.link(latest, temporary)
        except OSError:
            ensure_disk_budget(role, required_bytes=latest.stat().st_size)
            with latest.open("rb") as source:
                _atomic_file(role, lambda file: shutil.copyfileobj(source, file, 1024 * 1024), "wb")
        else:
            os.replace(temporary, role)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(checkpoint_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(checkpoint_bytes(item) for item in value)
    return 0


class Trainer:
    def __init__(self, args):
        self.started = perf_counter()
        self.args = args
        self.phase = args.phase
        self.stage = "sft" if self.phase == "sft" else "pretrain"
        self.run_dir, self.latest = args.run_dir, args.run_dir / "latest.pt"
        self.log_path = args.run_dir / "steps.jsonl"
        self.config = ModelConfig(**read_json(args.config))
        self.device = training_device(args, self.config)
        self.order = None
        self.start_tokens, self.start_updates = 0, 0
        self.base_checkpoint, self.branch_checkpoint = None, None
        self.masked_ids = ()
        self.validations, self.completed, self.saved_completed = {}, False, False
        self.published_roles = set()
        self.progress, self.saved_progress = Progress(), None
        self.boundary_rng, self.optimizer_unsafe = None, False
        self.eval_seconds, self.train_seconds, self.save_seconds, self.prior_seconds = 0.0, 0.0, 0.0, 0.0

        if self.stage == "pretrain":
            self.manifest, self.data_identity = read_pretrain_manifest(args.data_dir, self.config.vocab_size)
            self.configure_pretraining()
        else:
            self.manifest, self.data_identity = read_frozen_manifest(args.data_dir, "sft", self.config.vocab_size)
            self.configure_sft()
        self.tokenizer_hash = self.manifest["tokenizer_sha256"]
        checked_tokenizer(self.manifest["tokenizer_path"], self.tokenizer_hash, self.config.vocab_size)
        self.settings = self.make_settings()

        seed_everything(args.seed, self.device)
        self.model = TinyLM(self.config, activation_checkpointing=self.device.type == "cuda")
        if not args.resume and not args.branch_from and self.stage == "pretrain" and args.residual_init == "scaled":
            with torch.no_grad():
                scale = 1 / math.sqrt(2 * self.config.n_layers)
                for block in self.model.blocks:
                    block.attn.o_proj.weight.mul_(scale)
                    block.mlp.down_proj.weight.mul_(scale)
        if args.init_checkpoint:
            self.initialize_sft(args.init_checkpoint)
        self.model.to(self.device)
        self.optimizer = optimizer_for(self.model, self.device, self.schedule.learning_rate(0))
        if self.stage == "sft":
            self.optimizer.defaults["weight_decay"] = 0.01
            self.optimizer.param_groups[0]["weight_decay"] = 0.01
        if args.resume:
            self.restore(args.resume)
        elif args.branch_from:
            self.restore(args.branch_from, branch=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def configure_pretraining(self):
        args = self.args
        self.capacity = args.micro_batch * args.grad_accum * self.config.max_seq_len
        self.schedule = pretrain_schedule(self.phase, args.lr)
        if self.phase == "decay":
            self.start_tokens = STABLE_TOKENS
            self.start_updates = math.ceil(STABLE_TOKENS / self.capacity)
        self.planned_tokens = PILOT_UPDATES * self.capacity if self.phase == "pilot" else self.schedule.end
        self.planned_updates = self.start_updates + math.ceil((self.planned_tokens - self.start_tokens) / self.capacity)
        self.planned_examples = 0
        self.train = self.pretrain_stream("train")
        self.validation = {"general": self.pretrain_stream("general_validation"),
                           "mixed": self.pretrain_stream("validation")}
        if len(self.train.tokens) <= self.planned_tokens:
            raise ValueError(f"Pretraining requires {self.planned_tokens + 1} frozen train tokens")
        self.masked_ids = masked_token_ids(self.manifest)
        masked_cpu = torch.tensor(self.masked_ids, dtype=torch.long)
        excluded = torch.zeros(self.config.vocab_size, dtype=torch.bool)
        excluded[masked_cpu] = True
        self.compute_loss = partial(pretrain_loss_sum, masked=masked_cpu.to(self.device), excluded=excluded)

    def configure_sft(self):
        if manifest_final_eos(self.manifest):
            raise ValueError("Stage 2 SFT requires frozen final_eos=False data")
        rows = read_sft_data(self.args.data_dir, self.manifest, self.config)
        self.train, self.validation = rows["train"], rows["validation"]
        self.capacity = self.args.micro_batch * self.args.grad_accum
        self.order = bucket_order(self.train, self.args.seed, self.capacity)
        self.planned_examples = len(self.train)
        self.planned_tokens = self.manifest["splits"]["train"]["assistant_tokens"]
        self.planned_updates = math.ceil(self.planned_examples / self.capacity)
        self.schedule = Schedule("updates", self.planned_updates, max(1, math.ceil(self.planned_updates * 0.03)),
                                 self.args.lr, self.args.lr * 0.1)
        self.compute_loss = loss_sum

    def pretrain_stream(self, name):
        return PretrainStream(self.args.data_dir / f"{name}.bin", self.config.max_seq_len, self.config.vocab_size)

    def make_settings(self):
        args = self.args
        return {
            "phase": self.phase, "seed": args.seed, "device": self.device.type,
            "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
            "lr": args.lr, "residual_init": args.residual_init, "masked_token_ids": list(self.masked_ids),
            "planned_tokens": self.planned_tokens, "planned_updates": self.planned_updates,
            "planned_examples": self.planned_examples,
            "start_tokens": self.start_tokens, "start_updates": self.start_updates,
            "checkpoint_every": args.checkpoint_every, "eval_every": args.eval_every,
            "final_eos": False if self.stage == "sft" else None,
        }

    def initialize_sft(self, path):
        state = load_checkpoint(path)
        if (state["stage"] != "pretrain" or state["config"] != asdict(self.config)
                or state["tokenizer_sha256"] != self.tokenizer_hash
                or state["data_identity"]["tokenizer_sha256"] != self.tokenizer_hash):
            raise ValueError("SFT base architecture or frozen tokenizer differs")
        progress, settings = state["progress"], state["settings"]
        if state["schema_version"] == LEGACY_CHECKPOINT_SCHEMA:
            result = state["validations"].get("final", {})
            compatible = (progress["tokens"] == settings["planned_tokens"] == LEGACY_BASE_TOKENS
                          and progress["updates"] == settings["planned_updates"]
                          and result.get("step") == progress["updates"]
                          and result.get("nonfinite") is False and math.isfinite(result["loss"]))
        else:
            compatible = (state["schema_version"] == CHECKPOINT_SCHEMA and completed_checkpoint(state, "decay")
                          and progress["tokens"] == FINAL_TOKENS)
        if not compatible:
            raise ValueError("SFT requires a completed compatible pretraining base")
        self.model.load_state_dict(state["model"])
        self.base_checkpoint = {"path": str(path), "pretrain_tokens": progress["tokens"],
                                **file_identity(path)}

    def expected_progress(self, step, *, branch=False):
        start_tokens = 0 if branch else self.start_tokens
        start_updates = 0 if branch else self.start_updates
        end_tokens = STABLE_TOKENS if branch else self.planned_tokens
        if self.stage == "pretrain":
            tokens = min(end_tokens, start_tokens + (step - start_updates) * self.capacity)
            padding = tail_padding(start_tokens, self.args.micro_batch, self.config.max_seq_len)
            padding += tail_padding(tokens - start_tokens, self.args.micro_batch, self.config.max_seq_len)
            return Progress(step, tokens, 0, tokens, padding)
        indices = self.order[:step * self.capacity]
        micros = sft_batches(self.train, indices, self.args.micro_batch)
        targets, inputs, padding = 0, 0, 0
        for batch in micros:
            targets += batch.targets
            inputs += batch.input_tokens
            padding += batch.padding_tokens
        return Progress(step, targets, len(indices), inputs, padding)

    def restore(self, path, *, branch=False):
        state = load_checkpoint(path)
        schedule = pretrain_schedule("stable", self.args.lr) if branch else self.schedule
        expected = {
            "schema_version": CHECKPOINT_SCHEMA, "stage": self.stage, "config": asdict(self.config),
            "scheduler": asdict(schedule), "data_identity": self.data_identity,
            "tokenizer_sha256": self.tokenizer_hash, "sft_order": self.order,
        }
        if not branch:
            expected.update(settings=self.settings, run_dir=str(self.run_dir))
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Checkpoint field {key!r} differs from this run")
        if branch:
            if not completed_checkpoint(state, "stable") or Path(state["run_dir"]).resolve() == self.run_dir:
                raise ValueError("Decay requires a completed compatible stable checkpoint in a separate run")
            keys = ("seed", "device", "micro_batch", "grad_accum", "lr", "residual_init", "masked_token_ids")
            if any(state["settings"][key] != self.settings[key] for key in keys):
                raise ValueError("Branch training settings differ from the stable parent")

        progress = Progress(**state["progress"])
        max_step = self.start_updates if branch else self.planned_updates
        if (any(type(value) is not int or value < 0 for value in asdict(progress).values())
                or not self.start_updates <= progress.updates <= max_step
                or progress != self.expected_progress(progress.updates, branch=branch)):
            raise ValueError("Checkpoint progress differs from its optimizer boundary")
        cursor = progress.examples if self.stage == "sft" else progress.tokens
        position = progress.updates if schedule.unit == "updates" else progress.tokens
        if (state["global_tokens"] != progress.tokens or state["train_cursor"] != cursor
                or state["scheduler_position"] != position):
            raise ValueError("Checkpoint cursor differs from its schedule")
        self.check_optimizer(state["optimizer"], schedule.learning_rate(position), progress.updates)
        if (type(state["completed"]) is not bool
                or (state["completed"] and not completed_checkpoint(state, state["settings"]["phase"]))):
            raise ValueError("Checkpoint completion metadata is inconsistent")

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        self.progress = progress
        if branch:
            self.branch_checkpoint = {"path": str(path), "global_tokens": progress.tokens, **file_identity(path)}
        else:
            self.saved_progress = progress
            self.completed = self.saved_completed = state["completed"]
            self.base_checkpoint, self.branch_checkpoint = state["base_checkpoint"], state["branch_checkpoint"]
            self.validations = state["validations"]
            timing = state["timing"]
            metrics = self.run_dir / "metrics.json"
            if metrics.exists():
                previous = read_json(metrics)
                if previous.get("checkpoint") == str(path) and previous.get("checkpoint_step") == progress.updates:
                    timing = previous
            self.prior_seconds = timing["elapsed_seconds"]
            self.train_seconds = timing["training_seconds"]
            self.eval_seconds = timing["evaluation_seconds"]
            self.save_seconds = timing["checkpoint_seconds"]

    def check_optimizer(self, state, lr, updates):
        groups = self.optimizer.state_dict()["param_groups"]
        expected = [{**group, "lr": lr} for group in groups]
        if state["param_groups"] != expected:
            raise ValueError("Checkpoint optimizer groups differ from the configured schedule")
        parameters = {param for group in groups for param in group["params"]}
        moments = state["state"]
        if set(moments) != (parameters if updates else set()) or any(
            moment["step"].item() != updates for moment in moments.values()
        ):
            raise ValueError("Checkpoint optimizer moments differ from the completed updates")

    def timing(self):
        return {"elapsed_seconds": self.prior_seconds + perf_counter() - self.started,
                "training_seconds": self.train_seconds, "evaluation_seconds": self.eval_seconds,
                "checkpoint_seconds": self.save_seconds}

    def checkpoint_state(self):
        progress = self.progress
        return {
            "schema_version": CHECKPOINT_SCHEMA, "stage": self.stage, "run_dir": str(self.run_dir),
            "settings": self.settings, "config": asdict(self.config), "data_identity": self.data_identity,
            "tokenizer_sha256": self.tokenizer_hash, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "rng": capture_rng(self.device),
            "scheduler": asdict(self.schedule), "sft_order": self.order,
            "progress": asdict(progress), "global_tokens": progress.tokens,
            "scheduler_position": progress.updates if self.schedule.unit == "updates" else progress.tokens,
            "train_cursor": progress.examples if self.stage == "sft" else progress.tokens,
            "base_checkpoint": self.base_checkpoint, "branch_checkpoint": self.branch_checkpoint,
            "validations": self.validations, "completed": self.completed, "timing": self.timing(),
        }

    def publish_roles(self):
        roles = []
        if self.phase == "sft" and self.validations.get("best", {}).get("step") == self.progress.updates:
            roles.append("best.pt")
        if self.completed and self.phase != "pilot":
            roles.append("stable.pt" if self.phase == "stable" else "final.pt")
        for name in roles:
            if name not in self.published_roles:
                started = perf_counter()
                role_checkpoint(self.latest, self.run_dir / name)
                self.save_seconds += perf_counter() - started
                self.published_roles.add(name)

    def save(self):
        if self.saved_progress != self.progress or self.saved_completed != self.completed:
            verify_file_identity(self.data_identity)
            started = perf_counter()
            state = self.checkpoint_state()
            ensure_disk_budget(self.run_dir, required_bytes=checkpoint_bytes(state) + CHECKPOINT_OVERHEAD)
            _atomic_file(self.latest, lambda file: torch.save(state, file), "wb")
            self.saved_progress = self.progress
            self.saved_completed = self.completed
            self.published_roles.clear()
            self.save_seconds += perf_counter() - started
            emit({"event": "checkpoint", "step": self.progress.updates, "global_tokens": self.progress.tokens,
                  "checkpoint": str(self.latest), **self.timing()}, self.log_path)
        self.publish_roles()

    def validate(self, label):
        started = perf_counter()
        if self.stage == "sft":
            order = sorted(range(len(self.validation)), key=lambda index: len(self.validation[index][0]))
            scopes = {"sft": (sft_batches(self.validation, order, self.args.micro_batch), loss_sum)}
        else:
            limit = FINAL_VALIDATION_TOKENS if label == "final" else VALIDATION_TOKENS
            scopes = {name: (stream.batches(0, min(limit, len(stream.tokens) - 1), self.args.micro_batch),
                             self.compute_loss)
                      for name, stream in self.validation.items()}
            if label == "final" and self.phase in {"stable", "decay"}:
                stream = self.validation["general"]
                count = min(VALIDATION_TOKENS, len(stream.tokens) - 1)
                scopes["general_full_vocab"] = (stream.batches(0, count, self.args.micro_batch), loss_sum)
        results = {name: evaluate(self.model, batches, self.device, compute_loss)
                   for name, (batches, compute_loss) in scopes.items()}
        duration = perf_counter() - started
        self.eval_seconds += duration
        result = {"step": self.progress.updates, "global_tokens": self.progress.tokens,
                  "scopes": results, "evaluation_seconds": duration}
        emit({"event": "validation", "label": label, **result}, self.log_path)
        if any(value["nonfinite"] for value in results.values()):
            raise FloatingPointError(f"Non-finite {label} validation loss")
        self.validations[label] = result
        if self.stage == "sft" and label != "start":
            best_loss = self.validations.get("best", {}).get("scopes", {}).get("sft", {}).get("loss", math.inf)
            if results["sft"]["loss"] < best_loss:
                self.validations["best"] = result
                return True
        return False

    def update(self):
        started = perf_counter()
        self.boundary_rng = capture_rng(self.device)
        if self.stage == "pretrain":
            count = min(self.capacity, self.planned_tokens - self.progress.tokens)
            batches = list(self.train.batches(self.progress.tokens, count, self.args.micro_batch))
            examples = 0
        else:
            indices = self.order[self.progress.examples:self.progress.examples + self.capacity]
            batches = list(sft_batches(self.train, indices, self.args.micro_batch))
            examples = len(indices)
        next_progress = self.progress.advanced(batches, examples)
        targets = next_progress.tokens - self.progress.tokens
        position = next_progress.updates if self.schedule.unit == "updates" else next_progress.tokens
        lr = self.schedule.learning_rate(position)

        total = torch.zeros((), device=self.device)
        for batch in batches:
            loss = self.compute_loss(self.model, batch, self.device)
            (loss / targets).backward()
            total += loss.detach()
        loss = (total / targets).item()
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        if not math.isfinite(grad_norm):
            raise FloatingPointError("Non-finite training gradient norm")

        self.optimizer_unsafe = True
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.progress, self.boundary_rng, self.optimizer_unsafe = next_progress, None, False
        duration = perf_counter() - started
        self.train_seconds += duration
        emit({"event": "update", "step": self.progress.updates, "global_tokens": self.progress.tokens,
              "tokens_this_update": targets, "examples": self.progress.examples,
              "loss": loss, "grad_norm": grad_norm, "lr": lr, "update_seconds": duration,
              "input_tokens_per_second": sum(batch.input_tokens for batch in batches) / duration,
              "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None},
             self.log_path)

    def run(self):
        initial_step = self.progress.updates
        reason, error, recovery_error, code = "complete", None, None, 0
        pause_trigger = None
        try:
            emit({"event": "resume" if self.args.resume else "start", "phase": self.phase,
                  "pid": os.getpid(), "step": initial_step, "global_tokens": self.progress.tokens,
                  "settings": self.settings}, self.log_path)
            self.save()
            if not self.completed:
                if "start" not in self.validations and not self.args.pause_file.is_file():
                    self.validate("start")
                while self.progress.updates < self.planned_updates:
                    if self.args.pause_file.is_file():
                        pause_trigger = "file"
                        break
                    ensure_disk_budget(self.run_dir)
                    before = self.progress.tokens
                    self.update()
                    step = self.progress.updates
                    if self.args.pause_file.is_file():
                        pause_trigger = "file"
                        break
                    if step == self.planned_updates:
                        break
                    milestone = next((point for point in MILESTONES if before < point <= self.progress.tokens), None)
                    best = self.validate(f"tokens_{milestone}" if milestone else "periodic") if (
                        (self.phase == "stable" and milestone) or step % self.args.eval_every == 0
                    ) else False
                    if best or step % self.args.checkpoint_every == 0:
                        self.save()
                    if self.args.max_updates and step - initial_step >= self.args.max_updates:
                        pause_trigger = "max_updates"
                        break
                if self.args.pause_file.is_file():
                    pause_trigger = "file"
                if pause_trigger:
                    reason = "paused"
                else:
                    if (self.progress.tokens, self.progress.examples) != (self.planned_tokens, self.planned_examples):
                        raise RuntimeError("Training cursor differs from the approved budget")
                    self.validate("final")
                    self.completed = True
                self.save()
        except (Exception, KeyboardInterrupt) as failure:
            reason, code = stop_status(failure)
            error = f"{type(failure).__name__}: {failure}"
            if not self.optimizer_unsafe and not isinstance(failure, (DiskBudgetError, FloatingPointError)):
                try:
                    if self.boundary_rng is not None:
                        self.optimizer.zero_grad(set_to_none=True)
                        restore_rng(self.boundary_rng)
                    self.save()
                except (Exception, KeyboardInterrupt) as recovery:
                    recovery_error = f"{type(recovery).__name__}: {recovery}"

        metrics = {
            "phase": self.phase, "stage": self.stage, "stop_reason": reason,
            "checkpoint": str(self.latest) if self.saved_progress is not None else None,
            "checkpoint_step": self.saved_progress.updates if self.saved_progress is not None else None,
            "checkpoint_global_tokens": self.saved_progress.tokens if self.saved_progress is not None else None,
            "completed_updates": self.progress.updates, "global_tokens": self.progress.tokens,
            "examples": self.progress.examples, "settings": self.settings, "validation": self.validations,
            "base_checkpoint": self.base_checkpoint, "branch_checkpoint": self.branch_checkpoint,
            "error": error, "recovery_error": recovery_error, "optimizer_state_unsafe": self.optimizer_unsafe,
            "pause_trigger": pause_trigger, "pause_file": str(self.args.pause_file),
            **self.timing(),
        }
        atomic_json(self.run_dir / "metrics.json", metrics)
        emit({"event": "stop", **metrics}, self.log_path)
        return code


def stop_status(error):
    if isinstance(error, DiskBudgetError):
        return "paused_disk", 75
    if isinstance(error, KeyboardInterrupt):
        return "interrupted", 130
    return "failure", 1


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "stable", "decay", "sft"), required=True)
    for name in ("data-dir", "run-dir", "config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    for name in ("resume", "branch-from", "init-checkpoint"):
        source.add_argument(f"--{name}", type=Path)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--residual-init", choices=("original", "scaled"), default="scaled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-every", type=positive_argument)
    parser.add_argument("--checkpoint-every", type=positive_argument, default=250)
    parser.add_argument("--max-updates", type=positive_argument, help="Pause after this invocation's updates")
    parser.add_argument("--pause-file", type=Path, help="Pause at a safe update boundary while this file exists")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must fit an unsigned 32-bit integer")
    if args.phase == "sft":
        if not (args.resume or args.init_checkpoint) or args.lr not in (None, SFT_LR):
            parser.error("SFT requires --init-checkpoint or --resume and uses a peak LR of 6e-5")
        args.lr, args.eval_every = SFT_LR, args.eval_every or 50
        args.micro_batch, args.grad_accum = 8, 4
        args.residual_init = None
    else:
        if ((args.phase == "decay" and not (args.branch_from or args.resume))
                or (args.phase != "decay" and args.branch_from) or args.init_checkpoint):
            parser.error("Decay requires --branch-from or --resume; pilot and stable start from random weights")
        if args.lr is not None and args.lr not in PRETRAIN_LRS:
            parser.error("Approved pretraining peak LRs are 6e-4 and 1e-3")
        args.lr, args.eval_every = args.lr or PRETRAIN_LRS[0], args.eval_every or (25 if args.phase == "pilot" else 250)
        args.micro_batch, args.grad_accum = 8, 8
    for name in ("data_dir", "run_dir", "config", "resume", "branch_from", "init_checkpoint"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    args.pause_file = (args.pause_file or (args.run_dir / "PAUSE")).resolve()
    return args


def export(argv):
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description="Export a completed Stage 2 base or SFT weights")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    state = load_checkpoint(args.checkpoint)
    phase = state["settings"]["phase"]
    if (state["schema_version"] != CHECKPOINT_SCHEMA or phase not in {"decay", "sft"}
            or not (completed_checkpoint(state, phase) or (phase == "sft" and finite_validation(state, "best")))):
        raise ValueError("Export requires a completed decay/SFT checkpoint or a finite SFT best checkpoint")
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_file_identity(state["data_identity"])
    tokenizer = Path(state["data_identity"]["files"]["tokenizer.json"]["path"])
    checked_tokenizer(tokenizer, state["tokenizer_sha256"], state["config"]["vocab_size"])
    weights = {name: value.to(dtype=torch.bfloat16).contiguous() for name, value in state["model"].items()}
    ensure_disk_budget(args.output, required_bytes=checkpoint_bytes(weights) + tokenizer.stat().st_size + CHECKPOINT_OVERHEAD)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".stage2-export-", dir=args.output.parent) as temp:
        staging = Path(temp) / "artifact"
        staging.mkdir()
        metadata = {"format": "pt", "purpose": f"stage2-{phase}", "tokenizer_sha256": state["tokenizer_sha256"]}
        save_file(weights, str(staging / "model.safetensors"), metadata=metadata)
        atomic_json(staging / "config.json", state["config"])
        atomic_json(staging / "chat_format.json", {"special_token_ids": SPECIAL_IDS, "final_eos": False,
                    "implementation": "tinychat.chat_format.encode_conversation",
                    "context_tokens": state["config"]["max_seq_len"], "padding": "right",
                    "labels": "assistant content/end_turn only"})
        shutil.copyfile(tokenizer, staging / "tokenizer.json")
        if args.output.exists():
            raise FileExistsError(args.output)
        staging.rename(args.output)
    emit({"event": "export", "phase": phase, "output": str(args.output.resolve()),
          "weight_bytes": (args.output / "model.safetensors").stat().st_size})
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "export":
        try:
            return export(argv[1:])
        except (Exception, KeyboardInterrupt) as error:
            reason, code = stop_status(error)
            emit({"event": "export_stopped", "stop_reason": reason, "error": str(error)})
            return code
    args = parse_args(argv)
    previous_handler = signal.signal(signal.SIGTERM, handle_shutdown)
    try:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with lock_run(args.run_dir):
            latest = args.run_dir / "latest.pt"
            if args.resume and (args.resume != latest or not latest.is_file()):
                raise ValueError(f"Resume must use this run's latest committed checkpoint: {latest}")
            if not args.resume and (latest.exists() or any(args.run_dir.glob("*.pt"))
                                    or (args.run_dir / "latest.json").exists()):
                raise ValueError(f"Run already contains checkpoints: {args.run_dir}")
            if args.device == "cuda" and ModelConfig(**read_json(args.config)) != ModelConfig():
                raise ValueError("Formal Stage 2 training requires the frozen 74.4M model config")
            ensure_disk_budget(args.run_dir)
            return Trainer(args).run()
    except (Exception, KeyboardInterrupt) as failure:
        reason, code = stop_status(failure)
        emit({"event": "startup_stopped", "phase": args.phase, "stop_reason": reason,
              "error": f"{type(failure).__name__}: {failure}"})
        return code
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
