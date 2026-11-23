"""Single-sequence inference with the frozen stage-1 chat format."""
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tokenizers import Tokenizer

from tinychat.chat_format import check_special_ids, encode_conversation
from tinychat.model import ModelConfig, TinyLM


def checked_tokenizer(path, expected_hash, expected_vocab):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not expected_hash or digest != expected_hash:
        raise ValueError("Actual tokenizer does not match frozen checkpoint/artifact identity")
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    check_special_ids(tokenizer)
    if tokenizer.get_vocab_size() != expected_vocab:
        raise ValueError("Tokenizer vocabulary differs from model")
    return tokenizer, digest, len(raw)


def check_checkpoint(checkpoint, tokenizer_path, *, completed_stage=None):
    if checkpoint.get("stage") not in {"pretrain", "sft"}:
        raise ValueError("Expected a stage-1 checkpoint, not a provisional pilot")
    identity = checkpoint["data_identity"]
    digest = checkpoint["tokenizer_sha256"]
    if identity["tokenizer_sha256"] != digest:
        raise ValueError("Checkpoint tokenizer identity is inconsistent")
    tokenizer, digest, size = checked_tokenizer(tokenizer_path, digest, checkpoint["config"]["vocab_size"])
    recorded = identity["files"]["tokenizer.json"]
    actual_path = Path(tokenizer_path).resolve()
    if recorded["bytes"] != size:
        raise ValueError("Frozen tokenizer byte size changed")
    if actual_path == Path(recorded["path"]).resolve() and actual_path.stat().st_mtime_ns != recorded["mtime_ns"]:
        raise ValueError("Original frozen tokenizer file identity changed")
    if completed_stage is not None:
        if checkpoint["stage"] != completed_stage:
            raise ValueError("Checkpoint stage does not match requested export")
        progress, settings = checkpoint["progress"], checkpoint["settings"]
        if progress["tokens"] != settings["planned_tokens"] or progress["updates"] != settings["planned_updates"]:
            raise ValueError("Checkpoint has not completed its training budget")
        if completed_stage == "pretrain" and progress["tokens"] != 100_000_000:
            raise ValueError("Base export requires exactly 100M pretraining target tokens")
        if completed_stage == "sft" and (progress["examples"] != settings["planned_examples"] or not checkpoint.get("base_checkpoint")):
            raise ValueError("SFT export requires a completed epoch derived from a base checkpoint")
        for phase in ("final", "final_comparable"):
            validation = checkpoint["validations"].get(phase)
            if validation is None or validation["nonfinite"] or validation["step"] != progress["updates"]:
                raise ValueError("Completed checkpoint lacks finite final validation")
    return tokenizer, digest


def load_model(source, tokenizer_path=None, device="cuda"):
    source = Path(source)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if source.is_dir():
        config = ModelConfig(**json.loads((source / "config.json").read_text(encoding="utf-8")))
        tokenizer_path = tokenizer_path or source / "tokenizer.json"
        with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as artifact:
            metadata = artifact.metadata() or {}
        tokenizer, digest, _ = checked_tokenizer(tokenizer_path, metadata.get("tokenizer_sha256"), config.vocab_size)
        weights = load_file(str(source / "model.safetensors"), device="cpu")
    else:
        checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
        config = ModelConfig(**checkpoint["config"])
        weights = checkpoint["model"]
        if tokenizer_path is None:
            raise ValueError("Checkpoint inference needs an explicit frozen tokenizer")
        tokenizer, digest = check_checkpoint(checkpoint, tokenizer_path)
    with torch.device("meta"):
        model = TinyLM(config, activation_checkpointing=False)
    model.load_state_dict(weights, assign=True, strict=True)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    model.tokenizer_sha256 = digest
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens=96):
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be within 1..256")
    prompt = encode_conversation(tokenizer, messages, generation_prompt=True)["input_ids"]
    if len(prompt) + max_new_tokens > model.config.max_seq_len:
        raise ValueError(f"Prompt {len(prompt)} + output budget {max_new_tokens} exceeds context {model.config.max_seq_len}; shorten whole turns")
    device = next(model.parameters()).device
    tokens = list(prompt)
    generated = []
    stop_ids = {tokenizer.token_to_id("<|eos|>"), tokenizer.token_to_id("<|end_turn|>")}
    start = time.perf_counter()
    reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        next_id = int(model(x)[0, -1].argmax().item())
        tokens.append(next_id)
        generated.append(next_id)
        if next_id in stop_ids:
            reason = "end_token"
            break
    elapsed = time.perf_counter() - start
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "raw_text": tokenizer.decode(generated, skip_special_tokens=False),
        "generated_ids": generated,
        "input_tokens": len(prompt),
        "output_tokens": len(generated),
        "stop_reason": reason,
        "elapsed_seconds": elapsed,
    }


@torch.inference_mode()
def generate_batch(model, tokenizer, conversations, max_new_tokens=96):
    """Generate each row greedily and report time from batch start to its own stop."""
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be within 1..256")
    tokens = [encode_conversation(tokenizer, messages, generation_prompt=True)["input_ids"]
              for messages in conversations]
    prompt_lengths = [len(row) for row in tokens]
    if not tokens:
        return []
    for index, length in enumerate(prompt_lengths, 1):
        if length + max_new_tokens > model.config.max_seq_len:
            raise ValueError(f"Batch prompt {index} length {length} + output budget {max_new_tokens} "
                             f"exceeds context {model.config.max_seq_len}; shorten whole turns")

    device = next(model.parameters()).device
    pad_id = tokenizer.token_to_id("<|pad|>")
    stop_ids = {tokenizer.token_to_id("<|eos|>"), tokenizer.token_to_id("<|end_turn|>")}
    active = list(range(len(tokens)))
    row_indices = torch.arange(len(tokens), device=device)
    generated = [[] for _ in tokens]
    reasons = ["max_new_tokens"] * len(tokens)
    elapsed = [0.0] * len(tokens)

    start = time.perf_counter()
    for _ in range(max_new_tokens):
        width = max(len(tokens[index]) for index in active)
        # Pack each row's last real position alongside its padded prefix for one device transfer.
        packed = torch.tensor([
            [len(tokens[index]) - 1] + tokens[index] + [pad_id] * (width - len(tokens[index]))
            for index in active
        ], dtype=torch.long, device=device)
        # Causal attention keeps the future padding invisible to each real position.
        logits = model(packed[:, 1:])
        next_ids = logits[row_indices[:len(active)], packed[:, 0]].argmax(dim=-1).tolist()
        del logits
        duration = time.perf_counter() - start
        remaining = []
        for index, next_id in zip(active, next_ids, strict=True):
            generated[index].append(next_id)
            elapsed[index] = duration
            if next_id in stop_ids:
                reasons[index] = "end_token"
            else:
                tokens[index].append(next_id)
                remaining.append(index)
        if not remaining:
            break
        active = remaining

    return [{
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "raw_text": tokenizer.decode(ids, skip_special_tokens=False),
        "generated_ids": ids,
        "input_tokens": prompt_lengths[index],
        "output_tokens": len(ids),
        "stop_reason": reasons[index],
        "elapsed_seconds": elapsed[index],
    } for index, ids in enumerate(generated)]
