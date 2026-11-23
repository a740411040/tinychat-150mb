"""Run the fixed-budget stage-1 pretraining or the single SFT epoch."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import signal
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .chat_format import IGNORE_LABEL, encode_conversation, manifest_final_eos
from .model import ModelConfig, TinyLM
from .special_token_ablation import (
    DEFAULT_RESET_SEED,
    GRADIENT_STEPS,
    ablation_settings,
    embedding_geometry,
    embedding_gradient_norms,
    reset_unseen_specials,
    validate_reset_provenance,
)
from .training_state import (
    CHECKPOINT_SCHEMA,
    SPLITS,
    atomic_json,
    capture_rng,
    json_safe,
    lock_run,
    load_json,
    publish_latest,
    read_frozen_manifest,
    read_json,
    require_positive_integer,
    resolve_resume,
    restore_rng,
    verify_file_identity,
    write_checkpoint,
)


MAX_PRETRAIN_TOKENS = 100_000_000
VALIDATION_TOKENS = 131_072
FINAL_VALIDATION_TOKENS = 1_000_000


@dataclass(frozen=True)
class MicroBatch:
    inputs: torch.Tensor
    labels: torch.Tensor
    targets: int
    input_tokens: int
    padding_tokens: int = 0


class PretrainStream:
    def __init__(self, path, sequence_length, vocab_size):
        self.tokens = np.memmap(path, mode="r", dtype="<u2")
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

    def batches(self, start, count, micro_batch):
        if start < 0 or count < 1 or start + count >= len(self.tokens):
            raise ValueError("Pretraining window would exceed its frozen token stream")
        capacity = micro_batch * self.sequence_length
        for offset in range(start, start + count, capacity):
            positions = min(capacity, start + count - offset)
            window = self.tokens[offset:offset + positions + 1].astype(np.int64)
            if np.any(window >= self.vocab_size):
                raise ValueError(f"Token outside vocabulary at stream offset {offset}")
            full_rows, tail = divmod(positions, self.sequence_length)
            if not tail:
                shape = (full_rows, self.sequence_length)
                yield MicroBatch(
                    torch.from_numpy(window[:-1].reshape(shape)),
                    torch.from_numpy(window[1:].reshape(shape)),
                    positions, positions,
                )
            elif not full_rows:
                yield MicroBatch(
                    torch.from_numpy(window[None, :-1]),
                    torch.from_numpy(window[None, 1:]),
                    tail,
                    tail,
                )
            else:
                shape = (full_rows + 1, self.sequence_length)
                inputs = torch.zeros(shape, dtype=torch.long)
                labels = torch.full_like(inputs, IGNORE_LABEL)
                inputs.flatten()[:positions] = torch.from_numpy(window[:-1])
                labels.flatten()[:positions] = torch.from_numpy(window[1:])
                yield MicroBatch(inputs, labels, positions, positions, inputs.numel() - positions)


def sft_batch(rows):
    width = max(len(ids) - 1 for ids, _, _ in rows)
    inputs = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full_like(inputs, IGNORE_LABEL)
    input_tokens = 0
    for index, (ids, row_labels, _) in enumerate(rows):
        length = len(ids) - 1
        inputs[index, :length] = ids[:-1]
        labels[index, :length] = row_labels[1:]
        input_tokens += length
    return MicroBatch(inputs, labels, sum(row[2] for row in rows), input_tokens,
                      inputs.numel() - input_tokens)


def read_sft_rows(path, tokenizer, split_meta, config, *, final_eos=True):
    rows, seen_ids = [], set()
    input_tokens = assistant_tokens = 0
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            row = load_json(line)
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                raise ValueError(f"Invalid or duplicate SFT ID at {path}:{line_number}")
            if not isinstance(row["source"], str) or not row["source"]:
                raise ValueError(f"Missing SFT source at {path}:{line_number}")
            ids, labels = row["input_ids"], row["labels"]
            expected = encode_conversation(tokenizer, row["messages"], final_eos=final_eos)
            if (ids != expected["input_ids"] or labels != expected["labels"]
                    or len(ids) > config.max_seq_len or any(type(token) is not int for token in ids)
                    or any(type(label) is not int for label in labels)):
                raise ValueError(f"Invalid encoding or assistant mask at {path}:{line_number}")
            targets = sum(label != IGNORE_LABEL for label in labels[1:])
            if not targets:
                raise ValueError(f"Missing assistant targets at {path}:{line_number}")
            seen_ids.add(identifier)
            input_tokens += len(ids)
            assistant_tokens += targets
            rows.append((torch.tensor(ids), torch.tensor(labels), targets))
    if (len(rows), input_tokens, assistant_tokens) != (
        split_meta["documents"], split_meta["tokens"], split_meta["assistant_tokens"]
    ):
        raise ValueError(f"SFT data totals differ from frozen manifest: {path}")
    return rows, seen_ids


def read_sft_data(data_dir, manifest, config):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(manifest["tokenizer_path"])
    if (tokenizer.get_vocab_size() != config.vocab_size or any(
        tokenizer.token_to_id(token) != expected
        for token, expected in manifest["special_token_ids"].items()
    )):
        raise ValueError("SFT tokenizer vocabulary or special token IDs differ from the manifest")
    rows_by_split, unique_ids = {}, set()
    final_eos = manifest_final_eos(manifest)
    for split in SPLITS:
        rows, identifiers = read_sft_rows(data_dir / f"{split}.jsonl", tokenizer,
                                         manifest["splits"][split], config, final_eos=final_eos)
        if unique_ids & identifiers:
            raise ValueError(f"SFT IDs overlap previous splits: {split}")
        rows_by_split[split] = rows
        unique_ids.update(identifiers)
    return rows_by_split


def sft_batches(rows, indices, micro_batch):
    for start in range(0, len(indices), micro_batch):
        yield sft_batch([rows[index] for index in indices[start:start + micro_batch]])


@dataclass(frozen=True)
class Schedule:
    unit: str
    budget: int
    warmup: int
    peak: float
    minimum: float

    def learning_rate(self, completed):
        if completed >= self.budget:
            return self.minimum
        if completed <= self.warmup:
            return self.peak * completed / self.warmup
        fraction = (completed - self.warmup) / (self.budget - self.warmup)
        return self.minimum + (self.peak - self.minimum) * (1 + math.cos(math.pi * fraction)) / 2


@dataclass(frozen=True)
class Progress:
    updates: int = 0
    tokens: int = 0
    examples: int = 0
    input_tokens: int = 0
    padding_tokens: int = 0

    def advanced(self, batches, examples):
        return Progress(self.updates + 1, self.tokens + sum(batch.targets for batch in batches),
                        self.examples + examples,
                        self.input_tokens + sum(batch.input_tokens for batch in batches),
                        self.padding_tokens + sum(batch.padding_tokens for batch in batches))


def optimizer_for(model, device, lr):
    decay, no_decay = [], []
    for param in model.parameters():
        group = no_decay if param is model.token_embedding.weight or param.ndim < 2 else decay
        group.append(param)
    return torch.optim.AdamW(
        [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=device.type == "cuda",
    )


def loss_sum(model, batch, device):
    inputs, labels = batch.inputs.to(device), batch.labels.to(device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with autocast:
        logits = model(inputs)
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum",
                               ignore_index=IGNORE_LABEL)


def accumulate_gradients(model, batches, denominator, device):
    total = torch.zeros((), device=device)
    for batch in batches:
        summed_loss = loss_sum(model, batch, device)
        (summed_loss / denominator).backward()
        total += summed_loss.detach()
    return (total / denominator).item()


def evaluate(model, batches, device):
    previous_mode = model.training
    model.eval()
    losses = torch.zeros((), device=device)
    tokens = 0
    try:
        with torch.inference_mode():
            for batch in batches:
                losses += loss_sum(model, batch, device)
                tokens += batch.targets
        loss = losses.item() / tokens
        return {"loss": loss, "valid_loss_tokens": tokens, "nonfinite": not math.isfinite(loss)}
    finally:
        model.train(previous_mode)


def positive_argument(value):
    try:
        return require_positive_integer(int(value), "argument")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    for name in ("data-dir", "run-dir", "config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--micro-batch", type=positive_argument, default=8)
    parser.add_argument("--grad-accum", type=positive_argument, default=8)
    parser.add_argument("--checkpoint-every", type=positive_argument, default=250)
    parser.add_argument("--eval-every", type=positive_argument, default=250)
    parser.add_argument("--target-tokens", type=positive_argument)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=positive_argument, help="Pause after this invocation's updates for testing")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="CPU is for tiny test models only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-unseen-specials", action="store_true")
    parser.add_argument("--reset-special-seed", type=int, default=DEFAULT_RESET_SEED)
    parser.add_argument("--special-ablation-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "pretrain":
        if args.target_tokens is None or args.target_tokens > MAX_PRETRAIN_TOKENS or args.init_checkpoint:
            parser.error("Pretraining requires --target-tokens <= 100000000 and random initialization")
        if args.reset_unseen_specials or args.special_ablation_diagnostics:
            parser.error("Special-token ablation options require SFT")
    elif args.target_tokens or bool(args.init_checkpoint) == bool(args.resume):
        parser.error("SFT requires exactly one of --init-checkpoint or --resume, and no --target-tokens")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must fit an unsigned 32-bit integer")
    if not 0 <= args.reset_special_seed < 2**32:
        parser.error("--reset-special-seed must fit an unsigned 32-bit integer")
    if args.reset_special_seed != DEFAULT_RESET_SEED and not args.reset_unseen_specials:
        parser.error("--reset-special-seed requires --reset-unseen-specials")
    return args


def training_device(args, config):
    device = torch.device(args.device)
    if device.type == "cpu":
        if (config.d_model > 128 or config.n_layers > 4 or config.ffn_dim > 512
                or config.vocab_size > 16_384 or config.max_seq_len > 512):
            raise ValueError("CPU mode requires a tiny test config")
    elif not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Formal stage-1 training requires a CUDA GPU with native BF16 support")
    return device


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Expected a complete stage-1 training checkpoint: {path}")
    return state


def emit(record, log_path=None):
    message = json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as file:
            file.write(message + "\n")
    print(message, flush=True)


class Trainer:
    def __init__(self, args, resume):
        self.started = perf_counter()
        self.args = args
        self.run_dir = args.run_dir
        self.log_path = self.run_dir / "steps.jsonl"
        self.config = ModelConfig(**read_json(args.config))
        self.device = training_device(args, self.config)
        manifest, self.data_identity = read_frozen_manifest(args.data_dir, args.stage, self.config.vocab_size)
        self.tokenizer_hash = manifest["tokenizer_sha256"]
        self.order = None

        if args.stage == "pretrain":
            self.train = self.pretrain_stream("train")
            self.validation = self.pretrain_stream("validation")
            if len(self.train.tokens) <= args.target_tokens:
                raise ValueError(f"Pretraining requires at least {args.target_tokens + 1} frozen train tokens, "
                                 f"found {len(self.train.tokens)}")
            self.capacity = args.micro_batch * args.grad_accum * self.config.max_seq_len
            self.planned_updates = math.ceil(args.target_tokens / self.capacity)
            self.planned_tokens = args.target_tokens
            self.planned_examples = 0
            self.schedule = Schedule("tokens", args.target_tokens, max(1, math.ceil(args.target_tokens * 0.03)),
                                     3e-4, 3e-5)
            self.completion_reason = "target_tokens_reached"
        else:
            rows = read_sft_data(args.data_dir, manifest, self.config)
            self.train, self.validation = rows["train"], rows["validation"]
            self.order = np.random.default_rng(args.seed).permutation(len(self.train)).tolist()
            self.capacity = args.micro_batch * args.grad_accum
            self.planned_examples = len(self.train)
            self.planned_tokens = manifest["splits"]["train"]["assistant_tokens"]
            self.planned_updates = math.ceil(len(self.train) / self.capacity)
            self.schedule = Schedule("updates", self.planned_updates,
                                     max(1, math.ceil(self.planned_updates * 0.03)), 3e-5, 3e-6)
            self.completion_reason = "epoch_complete"

        self.settings = {
            "stage": args.stage, "seed": args.seed, "device": self.device.type,
            "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
            "planned_updates": self.planned_updates, "planned_tokens": self.planned_tokens,
            "planned_examples": self.planned_examples,
            "checkpoint_every": args.checkpoint_every, "eval_every": args.eval_every,
        }
        if args.stage == "sft":
            self.settings["special_token_ablation"] = ablation_settings(
                manifest_final_eos(manifest), args.reset_unseen_specials, args.reset_special_seed
            )
        self.special_token_initialization = None
        self.progress = Progress()
        self.resume_history = []
        self.base_checkpoint = None
        self.validations = {}
        self.prior_elapsed = self.prior_eval = self.prior_save = self.prior_training = 0.0
        self.eval_seconds = self.save_seconds = self.training_seconds = 0.0
        self.checkpoint = resume
        self.checkpoint_step = -1
        self.boundary_rng = None
        self.optimizer_in_progress = None

        seed_everything(args.seed, self.device)
        self.model = TinyLM(self.config, activation_checkpointing=self.device.type == "cuda").to(self.device)
        if resume is None and args.init_checkpoint:
            self.initialize_from_base(args.init_checkpoint.resolve())
        self.optimizer = optimizer_for(self.model, self.device, self.schedule.learning_rate(0))
        if resume is not None:
            self.resume(resume)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def pretrain_stream(self, split):
        return PretrainStream(self.args.data_dir / f"{split}.bin", self.config.max_seq_len, self.config.vocab_size)

    def elapsed(self):
        return self.prior_elapsed + perf_counter() - self.started

    def initialize_from_base(self, path):
        state = load_checkpoint(path)
        if self.device.type == "cuda" and state["settings"]["planned_tokens"] != MAX_PRETRAIN_TOKENS:
            raise ValueError("CUDA SFT requires a pretraining checkpoint with the full 100,000,000-token budget")
        config = asdict(self.config)
        base_data = state["data_identity"]
        base_tokenizer = base_data["files"]["tokenizer.json"]
        current_tokenizer = self.data_identity["files"]["tokenizer.json"]
        tokenizer_fields = ("bytes", "mtime_ns")
        if (state["stage"] != "pretrain" or state["config"] != config
                or state["tokenizer_sha256"] != self.tokenizer_hash
                or any(base_tokenizer[key] != current_tokenizer[key] for key in tokenizer_fields)
                or state["progress"]["tokens"] != state["settings"]["planned_tokens"]
                or state["progress"]["updates"] != state["settings"]["planned_updates"]
                or "final" not in state["validations"] or state["validations"]["final"]["nonfinite"]):
            raise ValueError("SFT requires a finished compatible stage-1 pretraining checkpoint")
        self.model.load_state_dict(state["model"])
        if self.args.reset_unseen_specials:
            self.special_token_initialization = reset_unseen_specials(self.model, self.args.reset_special_seed)
        elif self.args.special_ablation_diagnostics:
            self.special_token_initialization = {"changed_ids": [], "after": embedding_geometry(self.model)}
        self.base_checkpoint = str(path)

    def resume(self, path):
        state = load_checkpoint(path)
        expected = {
            "stage": self.args.stage, "config": asdict(self.config), "settings": self.settings,
            "scheduler": asdict(self.schedule), "data_identity": self.data_identity,
            "tokenizer_sha256": self.tokenizer_hash, "sft_order": self.order, "scaler": None,
        }
        for key, value in expected.items():
            actual = state.get(key)
            if key == "settings" and self.args.stage == "sft" and isinstance(actual, dict):
                # Checkpoints created before the SFT ablation used the legacy defaults.
                actual = {"special_token_ablation": ablation_settings(), **actual}
            if actual != value:
                raise ValueError(f"Checkpoint field {key!r} differs from the current frozen run")
        initialization = state.get("special_token_initialization")
        if self.args.reset_unseen_specials:
            validate_reset_provenance(initialization, self.settings["special_token_ablation"])

        progress = Progress(**state["progress"])
        if any(type(value) is not int or value < 0 for value in asdict(progress).values()):
            raise ValueError("Checkpoint progress fields must be nonnegative integers")
        if progress.updates > self.planned_updates:
            raise ValueError("Checkpoint progress exceeds the optimizer update budget")
        examples = min(self.planned_examples, progress.updates * self.capacity)
        if self.order is None:
            tokens = min(self.planned_tokens, progress.updates * self.capacity)
            input_tokens = tokens
            padded = 0
            if progress.updates == self.planned_updates:
                last_micro = tokens % (self.args.micro_batch * self.config.max_seq_len)
                tail = last_micro % self.config.max_seq_len
                if last_micro > self.config.max_seq_len and tail:
                    padded = self.config.max_seq_len - tail
        else:
            consumed = [self.train[index] for index in self.order[:examples]]
            tokens = sum(row[2] for row in consumed)
            lengths = [len(row[0]) - 1 for row in consumed]
            input_tokens = sum(lengths)
            padded = 0
            for start in range(0, examples, self.args.micro_batch):
                micro_lengths = lengths[start:start + self.args.micro_batch]
                padded += max(micro_lengths) * len(micro_lengths) - sum(micro_lengths)
        if (progress.examples, progress.tokens, progress.input_tokens, progress.padding_tokens) != (
            examples, tokens, input_tokens, padded
        ):
            raise ValueError("Checkpoint cursor does not match its completed optimizer updates")

        schedule_position = tokens if self.schedule.unit == "tokens" else progress.updates
        if state["scheduler_position"] != schedule_position:
            raise ValueError("Checkpoint scheduler position differs from the completed cursor")
        scheduled_lr = self.schedule.learning_rate(schedule_position)
        if any(group["lr"] != scheduled_lr for group in state["optimizer"]["param_groups"]):
            raise ValueError("Checkpoint optimizer LR differs from its schedule position")

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        self.progress = progress
        self.checkpoint_step = progress.updates
        self.base_checkpoint = state["base_checkpoint"]
        self.special_token_initialization = initialization
        self.validations = state["validations"]
        self.resume_history = [*state["resume_history"], {"checkpoint": str(path), "step": progress.updates}]
        timing = state["timing"]
        metrics_path = self.run_dir / "metrics.json"
        if metrics_path.exists():
            previous_metrics = read_json(metrics_path)
            if (previous_metrics.get("checkpoint") == str(path)
                    and previous_metrics.get("completed_updates") == progress.updates):
                timing = previous_metrics
        self.prior_elapsed = timing["elapsed_seconds"]
        self.prior_eval = timing["evaluation_seconds"]
        self.prior_save = timing["checkpoint_seconds"]
        self.prior_training = timing["training_seconds"]

    def timing(self):
        return {
            "elapsed_seconds": self.elapsed(),
            "training_seconds": self.prior_training + self.training_seconds,
            "evaluation_seconds": self.prior_eval + self.eval_seconds,
            "checkpoint_seconds": self.prior_save + self.save_seconds,
        }

    def checkpoint_state(self):
        progress = self.progress
        return {
            "schema_version": CHECKPOINT_SCHEMA, "stage": self.args.stage,
            "model": self.model.state_dict(), "config": asdict(self.config),
            "optimizer": self.optimizer.state_dict(), "scheduler": asdict(self.schedule),
            "scheduler_position": progress.tokens if self.schedule.unit == "tokens" else progress.updates,
            "scaler": None, "rng": capture_rng(self.device), "progress": asdict(progress),
            "settings": self.settings, "data_identity": self.data_identity, "sft_order": self.order,
            "tokenizer_sha256": self.tokenizer_hash, "resume_history": self.resume_history,
            "base_checkpoint": self.base_checkpoint, "validations": self.validations,
            "timing": self.timing(),
            **self.ablation_metadata(),
        }

    def ablation_metadata(self):
        metadata = {}
        if "special_token_ablation" in self.settings:
            metadata["special_token_ablation"] = self.settings["special_token_ablation"]
        if self.special_token_initialization is not None:
            metadata["special_token_initialization"] = self.special_token_initialization
        return metadata

    def save(self, force=False):
        if self.checkpoint_step == self.progress.updates and not force:
            return
        verify_file_identity(self.data_identity)
        started = perf_counter()
        self.checkpoint = write_checkpoint(self.run_dir, self.progress.updates, self.checkpoint_state()).resolve()
        self.checkpoint_step = self.progress.updates
        duration = perf_counter() - started
        self.save_seconds += duration
        emit({"event": "checkpoint", "step": self.checkpoint_step, "checkpoint": str(self.checkpoint),
              "checkpoint_seconds": duration, "elapsed_seconds": self.elapsed()}, self.log_path)

    def validate(self, phase):
        started = perf_counter()
        if self.order is None:
            cap = FINAL_VALIDATION_TOKENS if phase == "final" else VALIDATION_TOKENS
            count = min(cap, len(self.validation.tokens) - 1)
            batches = self.validation.batches(0, count, self.args.micro_batch)
            scope = {"max_loss_tokens": cap}
        else:
            batches = sft_batches(self.validation, range(len(self.validation)), self.args.micro_batch)
            scope = {"validation_examples": len(self.validation)}
        result = evaluate(self.model, batches, self.device)
        duration = perf_counter() - started
        self.eval_seconds += duration
        self.record_validation(phase, {**result, **scope, "evaluation_seconds": duration})
        if result["nonfinite"]:
            raise ValueError(f"Non-finite {phase} validation loss")

    def record_validation(self, phase, result):
        self.validations[phase] = {**result, "step": self.progress.updates}
        emit({"event": "validation", "phase": phase, **self.validations[phase],
              "elapsed_seconds": self.elapsed()}, self.log_path)

    def validate_final(self):
        final = self.validations.get("final")
        comparable = self.validations.get("final_comparable")
        if final is None or final["nonfinite"]:
            self.validate("final")
        if comparable is None or comparable["nonfinite"]:
            if self.order is not None or len(self.validation.tokens) - 1 <= VALIDATION_TOKENS:
                result = {**self.validations["final"], "evaluation_seconds": 0.0, "reused_from": "final"}
                if self.order is None:
                    result["max_loss_tokens"] = VALIDATION_TOKENS
                self.record_validation("final_comparable", result)
            else:
                self.validate("final_comparable")
        return (self.validations["final"], self.validations["final_comparable"]) != (final, comparable)

    def update(self):
        started = perf_counter()
        self.boundary_rng = capture_rng(self.device)
        if self.order is None:
            count = min(self.capacity, self.planned_tokens - self.progress.tokens)
            batches = list(self.train.batches(self.progress.tokens, count, self.args.micro_batch))
            examples = 0
        else:
            indices = self.order[self.progress.examples:self.progress.examples + self.capacity]
            batches = list(sft_batches(self.train, indices, self.args.micro_batch))
            examples = len(indices)
        next_progress = self.progress.advanced(batches, examples)
        targets = next_progress.tokens - self.progress.tokens
        units = next_progress.tokens if self.schedule.unit == "tokens" else next_progress.updates
        lr = self.schedule.learning_rate(units)

        loss = accumulate_gradients(self.model, batches, targets, self.device)
        if not math.isfinite(loss):
            raise ValueError("Non-finite training loss")
        diagnostics = {}
        if self.args.special_ablation_diagnostics and next_progress.updates in GRADIENT_STEPS:
            diagnostics["embedding_gradient_norms"] = embedding_gradient_norms(self.model)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        if not math.isfinite(grad_norm):
            raise ValueError("Non-finite training gradient norm")

        self.optimizer_in_progress = self.progress
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.progress = next_progress
        self.boundary_rng = None
        self.optimizer_in_progress = None

        duration = perf_counter() - started
        self.training_seconds += duration
        peak_memory = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
        emit({
            "event": "update", "step": self.progress.updates, "tokens_this_update": targets,
            "cumulative_valid_loss_tokens": self.progress.tokens, "examples_this_update": examples,
            "cumulative_examples": self.progress.examples, "micro_batches": len(batches),
            "input_tokens": sum(batch.input_tokens for batch in batches),
            "padding_tokens": sum(batch.padding_tokens for batch in batches),
            "loss": loss, "lr": lr, "grad_norm": grad_norm, "nonfinite": False,
            "elapsed_seconds": self.elapsed(), "update_seconds": duration,
            "effective_tokens_per_second": targets / duration, "peak_memory_bytes": peak_memory,
            **diagnostics,
        }, self.log_path)

    def run(self):
        if self.checkpoint is not None and not (self.run_dir / "latest.json").exists():
            publish_latest(self.run_dir, self.checkpoint, self.checkpoint_step)
        emit({"event": "resume" if self.checkpoint else "start", "stage": self.args.stage,
              "pid": os.getpid(), "step": self.progress.updates, "planned_loss_tokens": self.planned_tokens,
              "planned_examples": self.planned_examples, "planned_updates": self.planned_updates,
              "device": self.device.type, "tokenizer_sha256": self.tokenizer_hash,
              **self.ablation_metadata()}, self.log_path)
        invocation_start = self.progress.updates
        reason, error, recovery_error = self.completion_reason, None, None
        try:
            if "start" not in self.validations or self.validations["start"]["nonfinite"]:
                self.validate("start")
            while self.progress.updates < self.planned_updates:
                self.update()
                step = self.progress.updates
                if step % self.args.eval_every == 0:
                    self.validate("periodic")
                if step % self.args.checkpoint_every == 0 and step != self.planned_updates:
                    self.save()
                if (self.args.max_updates and step - invocation_start >= self.args.max_updates
                        and step < self.planned_updates):
                    reason = "interrupted"
                    break
            if reason == self.completion_reason:
                if (self.progress.tokens, self.progress.examples) != (self.planned_tokens, self.planned_examples):
                    raise RuntimeError("Training ended with a cursor different from the planned budget")
                validations_changed = self.validate_final()
                self.save(force=self.checkpoint_step == self.progress.updates and validations_changed)
            else:
                self.save()
        except (Exception, KeyboardInterrupt) as failure:
            error = f"{type(failure).__name__}: {failure}"
            reason = "interrupted" if isinstance(failure, KeyboardInterrupt) else "failure"
            if self.optimizer_in_progress is None:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.boundary_rng is not None:
                        restore_rng(self.boundary_rng)
                    self.save()
                except (Exception, KeyboardInterrupt) as recovery_failure:
                    recovery_error = f"{type(recovery_failure).__name__}: {recovery_failure}"
            else:
                self.progress = self.optimizer_in_progress

        metrics = {
            "stage": self.args.stage, "stop_reason": reason, "device": self.device.type,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "checkpoint_step": self.checkpoint_step if self.checkpoint else None,
            "completed_updates": self.progress.updates,
            "cumulative_valid_loss_tokens": self.progress.tokens, "cumulative_examples": self.progress.examples,
            "cumulative_input_tokens": self.progress.input_tokens,
            "cumulative_padding_tokens": self.progress.padding_tokens,
            "planned_loss_tokens": self.planned_tokens, "planned_examples": self.planned_examples,
            "planned_updates": self.planned_updates, "tokenizer_sha256": self.tokenizer_hash,
            "validation": self.validations, "resume_history": self.resume_history,
            "base_checkpoint": self.base_checkpoint, "error": error, "recovery_error": recovery_error,
            "optimizer_state_unsafe": self.optimizer_in_progress is not None, **self.timing(),
            **self.ablation_metadata(),
        }
        atomic_json(self.run_dir / "metrics.json", metrics)
        emit({"event": "stop", **metrics}, self.log_path)
        if reason == "failure":
            return 1
        return 130 if reason == "interrupted" and error else 0


def handle_shutdown(signum, _frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main(argv=None):
    args = parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    previous_handler = signal.signal(signal.SIGTERM, handle_shutdown)
    try:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with lock_run(args.run_dir):
            resume = resolve_resume(args.run_dir, args.resume)
            try:
                trainer = Trainer(args, resume)
            except (Exception, KeyboardInterrupt) as error:
                result = {"stage": args.stage,
                          "stop_reason": "interrupted" if isinstance(error, KeyboardInterrupt) else "failure",
                          "error": f"{type(error).__name__}: {error}"}
                if resume is None:
                    atomic_json(args.run_dir / "metrics.json", result)
                emit({"event": "startup_stopped", **result})
                return 130 if isinstance(error, KeyboardInterrupt) else 1
            return trainer.run()
    except (Exception, KeyboardInterrupt) as error:
        emit({"event": "error", "stage": args.stage, "error": f"{type(error).__name__}: {error}"})
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
