"""Run a time-limited CUDA training benchmark on local token streams."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F


SEED = 42
LEARNING_RATE = 3e-4
WARMUP_UPDATES = 2
HEARTBEAT_SECONDS = 30


class SequentialBatcher:
    """Cycle over a uint16 stream, advancing by the number of input positions."""

    def __init__(self, path, micro_batch, seq_len):
        size = path.stat().st_size
        if size < 4 or size % 2:
            raise ValueError(f"{path} must contain at least two complete uint16 tokens")
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.micro_batch = micro_batch
        self.seq_len = seq_len
        self.batch_tokens = micro_batch * seq_len
        self.cursor = 0
        self.wraps = 0

    def state_dict(self):
        return {"cursor": self.cursor, "wraps": self.wraps}

    def load_state_dict(self, state):
        cursor, wraps = state["cursor"], state["wraps"]
        if not 0 <= cursor < len(self.tokens) or wraps < 0:
            raise ValueError(f"Invalid batcher state: {state}")
        self.cursor, self.wraps = cursor, wraps

    def next_batch(self):
        end = self.cursor + self.batch_tokens
        if end < len(self.tokens):
            window = self.tokens[self.cursor : end + 1].astype(np.int64)
        else:
            positions = np.arange(self.cursor, end + 1) % len(self.tokens)
            window = self.tokens[positions].astype(np.int64)
        wraps, self.cursor = divmod(end, len(self.tokens))
        self.wraps += wraps
        shape = (self.micro_batch, self.seq_len)
        return torch.from_numpy(window[:-1].reshape(shape)), torch.from_numpy(
            window[1:].reshape(shape)
        )


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def read_data_identity(data_dir):
    meta_path = data_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        meta_fields = ("repo", "revision", "file", "seed", "splits", "vocab_size", "tokenizer_path")
        source = {key: meta[key] for key in meta_fields}
        source["special_token_ids"] = meta.get("special_token_ids")
        files = {}
        for name in ("train.bin", "tokenizer.json"):
            stat = (data_dir / name).stat()
            files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Could not read pilot data identity from {meta_path}: {error}") from error
    return {"meta": source, "files": files}


def validate_resume_data(checkpoint, data_identity):
    if checkpoint.get("data_identity") != data_identity:
        raise SystemExit("Checkpoint data identity does not match the local pilot data")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def log_json(record, *, indent=None):
    return json.dumps(json_safe(record), indent=indent, allow_nan=False)


def next_token_loss(logits, targets):
    return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())


def evaluate(model, batcher, dtype):
    num_batches = min(8, (len(batcher.tokens) - 1) // batcher.batch_tokens)
    if num_batches < 1:
        raise ValueError("Validation requires at least one full batch plus one target token")
    model.eval()
    total_loss = torch.zeros((), device="cuda")
    try:
        with torch.inference_mode():
            for _ in range(num_batches):
                inputs, targets = (tensor.cuda() for tensor in batcher.next_batch())
                with torch.autocast("cuda", dtype=dtype):
                    logits = model(inputs)
                total_loss += next_token_loss(logits, targets)
                del logits
        return total_loss.item() / num_batches
    finally:
        model.train()


def optimizer_for(model):
    decay, no_decay = [], []
    embedding = model.token_embedding.weight
    for param in model.parameters():
        if param is embedding or param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )


def lr_multiplier(update_index):
    return min((update_index + 1) / WARMUP_UPDATES, 1.0)


def train(model, optimizer, scheduler, scaler, batcher, dtype, args, steps, start_update):
    device = next(model.parameters()).device
    stats = {
        "stop_reason": "time_limit",
        "nan": False,
        "nonfinite": False,
        "oom": False,
        "fp16_overflow_attempts": 0,
        "completed_updates": 0,
        "trained_tokens": 0,
        "consumed_tokens": 0,
        "discarded_tokens": 0,
        "steady_state_tokens": 0,
        "steady_state_seconds": 0.0,
        "training_compute_seconds": 0.0,
        "checkpoint_boundary_intact": True,
    }
    start_time = perf_counter()
    deadline = start_time + args.seconds
    heartbeat = start_time + HEARTBEAT_SECONDS
    warmup_finished_at = None
    step_cursor = step_rng = None
    pending_tokens = 0
    compute_start = None
    optimizer.zero_grad(set_to_none=True)
    try:
        while perf_counter() < deadline:
            step_cursor, step_rng = batcher.state_dict(), capture_rng()
            update_compute = 0.0
            total_loss = torch.zeros((), device=device)
            for _ in range(args.grad_accum):
                if perf_counter() >= deadline:
                    break
                inputs, targets = batcher.next_batch()
                pending_tokens += batcher.batch_tokens
                stats["consumed_tokens"] += batcher.batch_tokens
                compute_start = perf_counter()
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                    logits = model(inputs.to(device))
                loss = next_token_loss(logits, targets.to(device))
                total_loss += loss.detach()
                scaler.scale(loss / args.grad_accum).backward()
                del logits, loss
                torch.cuda.synchronize(device)
                now = perf_counter()
                elapsed = now - compute_start
                compute_start = None
                update_compute += elapsed
                stats["training_compute_seconds"] += elapsed
                if now >= heartbeat:
                    print(
                        f"elapsed={now - start_time:.0f}s updates={stats['completed_updates']} "
                        f"trained_tokens={stats['trained_tokens']} pending_tokens={pending_tokens}",
                        flush=True,
                    )
                    heartbeat = now + HEARTBEAT_SECONDS
            if perf_counter() >= deadline:
                break

            compute_start = perf_counter()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step_loss, norm = torch.stack((total_loss / args.grad_accum, grad_norm)).tolist()
            if not math.isfinite(step_loss) or not math.isfinite(norm):
                stats["nan"] |= math.isnan(step_loss) or math.isnan(norm)
                stats["nonfinite"] = True
                if scaler.is_enabled() and math.isfinite(step_loss):
                    # GradScaler recorded the overflow before clipping. Reduce its scale
                    # and retry from the step's original stream and random states.
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    batcher.load_state_dict(step_cursor)
                    restore_rng(step_rng)
                    stats["fp16_overflow_attempts"] += 1
                    stats["discarded_tokens"] += pending_tokens
                    pending_tokens = 0
                    elapsed = perf_counter() - compute_start
                    compute_start = None
                    stats["training_compute_seconds"] += elapsed
                    continue
                stats["stop_reason"] = "nonfinite"
                stats["training_compute_seconds"] += perf_counter() - compute_start
                break
            if perf_counter() >= deadline:
                stats["training_compute_seconds"] += perf_counter() - compute_start
                break

            lr = optimizer.param_groups[0]["lr"]
            stats["checkpoint_boundary_intact"] = False
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - compute_start
            compute_start = None
            update_compute += elapsed
            stats["training_compute_seconds"] += elapsed
            stats["completed_updates"] += 1
            stats["trained_tokens"] += pending_tokens
            stats["checkpoint_boundary_intact"] = True
            update = start_update + stats["completed_updates"]
            if stats["completed_updates"] > WARMUP_UPDATES:
                stats["steady_state_tokens"] += pending_tokens
                stats["steady_state_seconds"] += update_compute
            elif stats["completed_updates"] == WARMUP_UPDATES:
                warmup_finished_at = perf_counter()
            steps.write(
                log_json(
                    {"update": update, "loss": step_loss, "lr": lr, "grad_norm": norm,
                     "tokens": pending_tokens, "compute_seconds": update_compute,
                     "cursor": batcher.cursor, "wraps": batcher.wraps}
                )
                + "\n"
            )
            steps.flush()
            pending_tokens = 0
            optimizer.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError as error:
        stats["oom"] = True
        stats["stop_reason"] = "oom"
        stats["error"] = str(error)
        if compute_start is not None:
            stats["training_compute_seconds"] += perf_counter() - compute_start
    finally:
        optimizer.zero_grad(set_to_none=True)
        if pending_tokens:
            stats["discarded_tokens"] += pending_tokens
            batcher.load_state_dict(step_cursor)
            restore_rng(step_rng)
    end_time = perf_counter()
    wall_time = end_time - start_time
    stats["wall_time_seconds"] = wall_time
    stats["effective_tokens_per_second"] = stats["trained_tokens"] / wall_time if wall_time else None
    steady_seconds = stats["steady_state_seconds"]
    stats["steady_state_tokens_per_second"] = (
        stats["steady_state_tokens"] / steady_seconds if steady_seconds else None
    )
    stats["steady_state_timing"] = "compute time of successful updates after two warmup updates"
    post_warmup_wall_time = end_time - warmup_finished_at if warmup_finished_at is not None else None
    stats["post_warmup_wall_time_seconds"] = post_warmup_wall_time
    stats["post_warmup_effective_tokens_per_second"] = (
        stats["steady_state_tokens"] / post_warmup_wall_time if post_warmup_wall_time else None
    )
    return stats


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def time_limit(value):
    result = int(value)
    if not 60 <= result <= 1200:
        raise argparse.ArgumentTypeError("must be between 60 and 1200 seconds")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=time_limit, default=720)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/pilot"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--micro-batch", type=positive_int, default=1)
    parser.add_argument("--grad-accum", type=positive_int, default=64)
    parser.add_argument("--resume", type=Path, help="Trusted local checkpoint.pt file or its run directory")
    return parser.parse_args()


def validation_loss(model, args, seq_len, dtype, metrics, stage):
    try:
        loss = evaluate(
            model, SequentialBatcher(args.data_dir / "validation.bin", args.micro_batch, seq_len), dtype
        )
    except torch.cuda.OutOfMemoryError as error:
        metrics["validation_errors"][stage] = str(error)
        metrics["oom"] = True
        return None
    metrics["nan"] |= math.isnan(loss)
    metrics["nonfinite"] |= not math.isfinite(loss)
    return loss if math.isfinite(loss) else None


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA")
    from .model import ModelConfig, TinyLM

    key_files = ("checkpoint.pt", "checkpoint.pt.partial", "metrics.json", "steps.jsonl")
    if any((args.run_dir / name).exists() for name in key_files):
        raise SystemExit(f"Run directory already contains benchmark outputs: {args.run_dir}")
    config = ModelConfig(**json.loads(args.config.read_text(encoding="utf-8-sig")))
    data_identity = read_data_identity(args.data_dir)
    batcher = SequentialBatcher(args.data_dir / "train.bin", args.micro_batch, config.max_seq_len)
    validation = SequentialBatcher(
        args.data_dir / "validation.bin", args.micro_batch, config.max_seq_len
    )
    if len(validation.tokens) <= validation.batch_tokens:
        raise SystemExit("Validation requires at least one full batch plus one target token")
    del validation
    run_settings = {
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "train_stream_tokens": len(batcher.tokens),
        "effective_tokens_per_update": batcher.batch_tokens * args.grad_accum,
        "seed": SEED,
    }
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    run_settings["precision"] = str(dtype).removeprefix("torch.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = TinyLM(config, activation_checkpointing=True).to(device="cuda", dtype=torch.float32)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    start_update = start_tokens = 0
    rng = capture_rng()
    if args.resume:
        path = args.resume / "checkpoint.pt" if args.resume.is_dir() else args.resume
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != asdict(config) or checkpoint["run_settings"] != run_settings:
            raise SystemExit("Checkpoint configuration, precision, or stream length does not match")
        validate_resume_data(checkpoint, data_identity)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        batcher.load_state_dict(checkpoint["batcher"])
        start_update = checkpoint["completed_updates"]
        start_tokens = checkpoint["trained_tokens"]
        rng = checkpoint["rng"]
        del checkpoint

    args.run_dir.mkdir(parents=True, exist_ok=True)
    validation_metrics = {"nan": False, "nonfinite": False, "oom": False, "validation_errors": {}}
    with (args.run_dir / "steps.jsonl").open("x", encoding="utf-8") as steps:
        print(
            f"device={torch.cuda.get_device_name()} precision={run_settings['precision']} "
            f"seconds={args.seconds} tokens_per_update={run_settings['effective_tokens_per_update']} "
            f"start_update={start_update}",
            flush=True,
        )
        before = validation_loss(model, args, config.max_seq_len, dtype, validation_metrics, "before")
        restore_rng(rng)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stats = train(model, optimizer, scheduler, scaler, batcher, dtype, args, steps, start_update)
    stats["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
    stats["cuda_max_reserved_bytes"] = torch.cuda.max_memory_reserved()
    stats["time_limit_seconds"] = args.seconds
    stats["total_completed_updates"] = start_update + stats["completed_updates"]
    stats["total_trained_tokens"] = start_tokens + stats["trained_tokens"]
    stats["batcher"] = batcher.state_dict()
    stats["config"] = asdict(config)
    stats["run_settings"] = run_settings
    stats["benchmark_lr_schedule"] = {
        "warmup_updates": WARMUP_UPDATES,
        "peak_lr": LEARNING_RATE,
        "after_warmup": "constant",
    }
    stats["checkpoint_saved"] = stats["checkpoint_boundary_intact"]
    if stats["checkpoint_saved"]:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": capture_rng(),
            "batcher": stats["batcher"],
            "completed_updates": stats["total_completed_updates"],
            "trained_tokens": stats["total_trained_tokens"],
            "config": stats["config"],
            "run_settings": run_settings,
            "data_identity": data_identity,
        }
        partial_path = args.run_dir / "checkpoint.pt.partial"
        with partial_path.open("xb") as handle:
            torch.save(payload, handle)
        partial_path.rename(args.run_dir / "checkpoint.pt")
    stats["validation_loss_before"] = before
    stats["validation_loss_after"] = validation_loss(
        model, args, config.max_seq_len, dtype, validation_metrics, "after"
    )
    stats["validation_errors"] = validation_metrics["validation_errors"]
    for key in ("nan", "nonfinite", "oom"):
        stats[key] |= validation_metrics[key]
    with (args.run_dir / "metrics.json").open("x", encoding="utf-8") as handle:
        handle.write(log_json(stats, indent=2) + "\n")
    print(
        f"stop={stats['stop_reason']} updates={stats['completed_updates']} "
        f"trained_tokens={stats['trained_tokens']} discarded_tokens={stats['discarded_tokens']} "
        f"checkpoint_saved={stats['checkpoint_saved']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
