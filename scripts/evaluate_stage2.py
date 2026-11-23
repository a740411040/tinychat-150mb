"""Compare Stage 2 bases or prepare a blind natural-chat review after SFT."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from math import exp
from pathlib import Path
import random
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
LEGACY_DATA = ROOT / "data/stage1_v1"
NATURAL = ROOT / "evaluation/natural_sft_v3"
CONTEXT = 512
MAX_NEW = 96
VOCAB_SIZE = 16_384
LM_TARGETS = 131_072
LM_BATCH = 8
BLIND_SEED = 42
LABELS = ("old_100m", "new_500m")
GREEDY = {"strategy": "greedy", "context_tokens": CONTEXT, "max_new_tokens": MAX_NEW,
          "batch_size": 1, "stop_tokens": ["<|eos|>", "<|end_turn|>"]}


def write_json_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def notify(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def load_branch(source, device, tokenizer_sha256):
    from tinychat.inference import load_model
    from tinychat.training_state import file_identity

    weights = source / "model.safetensors" if source.is_dir() else source
    identity = file_identity(weights)
    model, tokenizer = load_model(source, LEGACY_DATA / "tokenizer.json", device)
    if (model.config.max_seq_len, model.config.vocab_size) != (CONTEXT, VOCAB_SIZE):
        raise ValueError(f"Expected the fixed context/vocabulary: {source}")
    if model.tokenizer_sha256 != tokenizer_sha256 or file_identity(weights) != identity:
        raise ValueError(f"Unexpected tokenizer or weights changed while loading: {source}")
    return model, tokenizer, {
        "source": str(source.resolve()), "weights": {"path": str(weights.resolve()), **identity},
        "tokenizer_sha256": model.tokenizer_sha256, "config": asdict(model.config),
        "dtype": str(next(model.parameters()).dtype), "device": device,
    }


def plain_prefixes():
    from scripts.diagnose_chat_v2 import PLAIN
    from scripts.diagnose_stage1_forward import CASES

    return [(f"diagnosis_v1:diag_{index:02}", prefix) for index, (prefix, _, _) in enumerate(CASES, 1)] + [
        (f"diagnosis_v2:plain_{index}", prefix) for index, prefix in enumerate(PLAIN, 1)
    ]


def plain_generate(model, tokenizer, prefix):
    import torch

    prompt = tokenizer.encode(prefix, add_special_tokens=False).ids
    if not prompt or len(prompt) + MAX_NEW > CONTEXT:
        raise ValueError("Plain prefix is empty or exceeds the fixed context")
    stop_ids = {tokenizer.token_to_id(token) for token in GREEDY["stop_tokens"]}
    device = next(model.parameters()).device
    ids = []
    started = perf_counter()
    with torch.inference_mode():
        for _ in range(MAX_NEW):
            inputs = torch.tensor([prompt + ids], dtype=torch.long, device=device)
            token = int(model(inputs)[0, -1].argmax().item())
            ids.append(token)
            if token in stop_ids:
                break
    return {"text": tokenizer.decode(ids, skip_special_tokens=True),
            "raw_text": tokenizer.decode(ids, skip_special_tokens=False), "generated_ids": ids,
            "input_tokens": len(prompt), "output_tokens": len(ids),
            "stop_reason": "end_token" if ids[-1] in stop_ids else "max_new_tokens",
            "elapsed_seconds": perf_counter() - started}


def run_base(args, sources):
    import torch
    from tinychat.evaluate_stage1 import diagnostics, summarize
    from tinychat.train_stage1 import PretrainStream, loss_sum
    from tinychat.train_stage2 import evaluate, read_pretrain_manifest
    from tinychat.training_state import read_frozen_manifest

    old, old_identity = read_frozen_manifest(LEGACY_DATA, "pretrain", VOCAB_SIZE)
    mixed, mixed_identity = read_pretrain_manifest(args.mixed_data, VOCAB_SIZE)
    if (mixed["stage2"]["profile"] != "full" or old["corpus_id"] != "stage1_v1"
            or old["dtype"] != mixed["dtype"] or old["tokenizer_sha256"] != mixed["tokenizer_sha256"]
            or mixed["stage2"]["general_validation_origin"] != old_identity["files"]["validation.bin"]):
        raise ValueError("Expected the formal mixed corpus and the original FineWeb general validation")
    windows = {"general": old_identity["files"]["validation.bin"],
               "mixed": mixed_identity["files"]["validation.bin"]}
    streams = {domain: PretrainStream(window["path"], CONTEXT, VOCAB_SIZE) for domain, window in windows.items()}
    if any(len(stream.tokens) <= LM_TARGETS for stream in streams.values()):
        raise ValueError("A frozen validation corpus is too short for the specified LM window")
    prefixes = plain_prefixes()
    if len(prefixes) != 12 or len({prefix for _, prefix in prefixes}) != len(prefixes):
        raise ValueError("Expected the twelve existing plain prefixes")

    args.output.mkdir(parents=True, exist_ok=False)
    for label, source in sources.items():
        model, tokenizer, model_info = load_branch(source, args.device, old["tokenizer_sha256"])
        scores = {}
        for domain, stream in streams.items():
            notify("lm_validation_start", model=label, domain=domain, targets=LM_TARGETS)
            result = evaluate(model, stream.batches(0, LM_TARGETS, LM_BATCH), torch.device(args.device), loss_sum)
            if result["nonfinite"] or result["valid_loss_tokens"] != LM_TARGETS:
                raise ValueError(f"Invalid full-vocabulary LM evaluation: {label}/{domain}")
            scores[domain] = {**result, "perplexity": exp(result["loss"])}
            notify("lm_validation_complete", model=label, domain=domain, **scores[domain])

        rows = []
        with (args.output / f"{label}.continuation.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for index, (case_id, prefix) in enumerate(prefixes, 1):
                row = {"id": case_id, "split": "diagnostic", "category": "plain", "prefix": prefix,
                       **plain_generate(model, tokenizer, prefix)}
                row["diagnostics"] = diagnostics(row["text"], row["raw_text"])
                write_row(handle, row)
                handle.flush()
                rows.append(row)
                notify("continuation_progress", model=label, completed=index, total=len(prefixes))
        write_json_new(args.output / f"{label}.summary.json", {
            "model": model_info, "lm": {"logits": "full vocabulary", "window_start": 0,
            "targets_per_domain": LM_TARGETS, "context_tokens": CONTEXT, "batch_size": LM_BATCH,
            "loss_units": "nats per target", "manifests": {"general": old_identity["manifest"],
            "mixed": mixed_identity["manifest"]}, "files": windows, "metrics": scores},
            "continuation": {"decoding": GREEDY, "prompt_encoding": "plain text; no added special tokens",
                             "metrics": summarize(rows), "limitations": "Twelve prior diagnostic prefixes."},
        })
        del model


def natural_prompts(split):
    from tinychat.chat_format import DEFAULT_SYSTEM

    path = NATURAL / "prompts.jsonl"
    manifest = json.loads((NATURAL / "prompt_manifest.json").read_text(encoding="utf-8"))
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if (manifest["frozen"] is not True or manifest["freeze_version"] != 2
            or digest != manifest["prompts_sha256"] or manifest["system"] != DEFAULT_SYSTEM
            or manifest["tokenizer"] != "data/stage1_v1/tokenizer.json"
            or manifest["chat_format"] != "tinychat.chat_format.encode_conversation(generation_prompt=True)"
            or manifest["context_tokens"] != CONTEXT or manifest["primary"] != {
                "split": "test", "strategy": "greedy", "batch_size": 1,
                "max_new_tokens": MAX_NEW, "runs_per_model": 1}):
        raise ValueError("Expected the Stage 2 frozen natural_sft_v3 v2 evaluation contract")
    prompts = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    rows = [record for record in prompts if record["split"] == split]
    categories = Counter(record["category"] for record in rows)
    if (len(rows) != manifest[f"{split}_count"] or len({row["id"] for row in rows}) != len(rows)
            or categories != {key: value[split] for key, value in manifest["category_counts"].items()}
            or sum("machine_check" in row for row in rows) != manifest["machine_check_counts"][split]):
        raise ValueError(f"Frozen natural prompts have unexpected counts for {split}")
    for row in rows:
        first = row["messages"][0]
        if first["role"] == "system" and first["content"] != manifest["system"]:
            raise ValueError(f"Frozen natural system prompt differs: {row['id']}")
    return rows, {"path": str(path), "sha256": digest, "freeze_version": manifest["freeze_version"],
                  "split": split, "system": manifest["system"], "chat_format": manifest["chat_format"]}


def blind_review(prompts, answers):
    from scripts.freeze_natural_sft_prompts import score_machine

    rng = random.Random(BLIND_SEED)
    packet, key = [], {}
    for prompt in prompts:
        prompt_id = prompt["id"]
        if "machine_check" in prompt:
            key[prompt_id] = {"category": prompt["category"], "machine_success": {
                label: score_machine(prompt, answers[label][prompt_id]) for label in LABELS}}
            continue
        labels = list(LABELS)
        rng.shuffle(labels)
        mapping = dict(zip(("A", "B"), labels, strict=True))
        key[prompt_id] = {"category": prompt["category"], "answer_labels": mapping}
        packet.append({"id": prompt_id, "messages": prompt["messages"], "rubric": prompt["rubric"],
                       "answers": {answer_id: answers[label][prompt_id] for answer_id, label in mapping.items()}})
    rng.shuffle(packet)
    return packet, key


def run_chat(args, sources):
    from tinychat.chat_format import encode_conversation
    from tinychat.evaluate_stage1 import diagnostics, summarize
    from tinychat.inference import generate_batch

    prompts, prompt_source = natural_prompts(args.split)
    legacy = json.loads((LEGACY_DATA / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    answers = {}
    for label, source in sources.items():
        model, tokenizer, model_info = load_branch(source, args.device, legacy["tokenizer_sha256"])
        for prompt in prompts:
            length = len(encode_conversation(tokenizer, prompt["messages"], generation_prompt=True)["input_ids"])
            if length != prompt["prompt_tokens"] or length + MAX_NEW > CONTEXT:
                raise ValueError(f"Frozen chat prompt tokenization changed: {prompt['id']}")
        rows = []
        with (args.output / f"{label}.chat.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for index, prompt in enumerate(prompts, 1):
                generated, = generate_batch(model, tokenizer, [prompt["messages"]], MAX_NEW)
                row = {field: prompt[field] for field in ("id", "split", "category")}
                row.update(generated, diagnostics=diagnostics(generated["text"], generated["raw_text"]))
                write_row(handle, row)
                handle.flush()
                rows.append(row)
                if index % 10 == 0 or index == len(prompts):
                    notify("chat_progress", model=label, split=args.split, completed=index, total=len(prompts))
        answers[label] = {row["id"]: row["text"] for row in rows}
        write_json_new(args.output / f"{label}.summary.json", {
            "model": model_info, "prompts": prompt_source, "decoding": GREEDY, "metrics": summarize(rows),
        })
        del model

    packet, key = blind_review(prompts, answers)
    write_json_new(args.output / "private_review_key.json", {"seed": BLIND_SEED, "cases": key})
    with (args.output / "blind_packet.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in packet:
            write_row(handle, row)
    notify("blind_packet_ready", split=args.split, semantic_pairs=len(packet), machine_pairs=len(key) - len(packet))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    base = modes.add_parser("base", help="Full-vocabulary LM loss and plain continuation")
    base.add_argument("--mixed-data", required=True, type=Path)
    chat = modes.add_parser("chat", help="Natural-chat diagnostics and anonymous review packet")
    chat.add_argument("--split", choices=("dev", "test"), required=True)
    for mode in (base, chat):
        mode.add_argument("--old-model", required=True, type=Path)
        mode.add_argument("--new-model", required=True, type=Path)
        mode.add_argument("--output", required=True, type=Path, help="A fresh output directory")
        mode.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not args.old_model.exists() or not args.new_model.exists():
        parser.error("Both model sources must exist")
    if args.old_model.samefile(args.new_model):
        parser.error("Old and new model sources must be distinct")
    if args.output.exists():
        parser.error("Evaluation output directory already exists")
    sources = dict(zip(LABELS, (args.old_model, args.new_model), strict=True))
    (run_base if args.mode == "base" else run_chat)(args, sources)


if __name__ == "__main__":
    main()
