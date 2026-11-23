"""Bounded forward-only diagnosis of existing weights; no training or weight writes."""
import argparse
import gc
import json
import random
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from tinychat.chat_format import encode_conversation
from tinychat.evaluate_stage1 import diagnostics
from tinychat.inference import load_model
from tinychat.model import ModelConfig, TinyLM


CASES = [
    ("春天到了，公园里的树木", "请接着写：春天到了，公园里的树木", "长出了嫩绿的叶子。"),
    ("氧气的化学式是", "氧气的化学式是什么？", "O2。"),
    ("9加6等于", "9加6等于多少？", "15。"),
    ("你好！很高兴", "请用一句话和我打招呼。", "你好，很高兴认识你！"),
    ("三种常见水果：苹果、香蕉、", "请列出三种常见水果，每种单独一行。", "苹果\n香蕉\n橙子"),
    ("如果外面正在下雨，出门可以", "下雨天出门可以带什么？", "可以带雨伞。"),
    ("礼貌地请求关窗：", "把“把窗户关上”改成礼貌的请求，只输出改写结果。", "请帮忙关一下窗户，谢谢。"),
    ("我叫小雨。我的名字是", "我刚才说自己叫什么？", "你叫小雨。"),
]


def cases(tokenizer):
    result = []
    for index, (prefix, user, reference) in enumerate(CASES):
        history = ([{"role": "user", "content": "我叫小雨。"},
                    {"role": "assistant", "content": "你好，小雨。"}] if index == 7 else [])
        messages = history + [{"role": "user", "content": user}]
        encoded = encode_conversation(tokenizer, messages + [{"role": "assistant", "content": reference}])
        result.append({"id": f"diag_{index + 1:02d}", "plain_prefix": prefix, "messages": messages,
                       "reference": reference, "input_ids": encoded["input_ids"], "labels": encoded["labels"]})
    return result


def sft_sample(path, seed):
    rng = random.Random(seed)
    pools, seen = {}, {}
    with path.open(encoding="utf-8") as file:
        for row in map(json.loads, file):
            key = row["source"]
            cap = 12 if "multiturn" in key else 116
            pool = pools.setdefault(key, [])
            seen[key] = seen.get(key, 0) + 1
            if len(pool) < cap:
                pool.append(row)
            else:
                index = rng.randrange(seen[key])
                if index < cap:
                    pool[index] = row
    selected = [row for key in sorted(pools) for row in pools[key]]
    if len(selected) != 128:
        raise ValueError("Expected a stratified 128-case SFT sample")
    return selected


def windows(path, low, high, count=64):
    stream = np.memmap(path, dtype="<u2", mode="r")
    high = min(high, len(stream) - 513)
    if low > high:
        raise ValueError("Empty diagnostic token range")
    offsets = np.linspace(low, high, count, dtype=np.int64)
    return [{"id": f"{path.name}:{int(offset)}", "input_ids": stream[offset:offset + 513].astype(np.int64).tolist(),
             "labels": stream[offset:offset + 513].astype(np.int64).tolist()} for offset in offsets]


def condition_pairs(rows, tokenizer):
    singles = [r for r in rows if len(r["messages"]) == 2][:64]
    real, mismatched = [], []
    for index, row in enumerate(singles):
        messages = [{"role": "user", "content": singles[(index + 1) % len(singles)]["messages"][0]["content"]},
                    row["messages"][1]]
        altered = encode_conversation(tokenizer, messages)
        if len(altered["input_ids"]) > 512:
            continue
        real.append(row)
        mismatched.append({"id": row["id"], **altered})
    return real, mismatched


def batch_tensors(rows):
    length = max(len(r["input_ids"]) - 1 for r in rows)
    inputs = torch.zeros((len(rows), length), dtype=torch.long, device="cuda")
    labels = torch.full_like(inputs, -100)
    for index, row in enumerate(rows):
        n = len(row["input_ids"]) - 1
        inputs[index, :n] = torch.tensor(row["input_ids"][:-1], device="cuda")
        labels[index, :n] = torch.tensor(row["labels"][1:], device="cuda")
    return inputs, labels


def context(master):
    return torch.autocast("cuda", dtype=torch.bfloat16) if master else nullcontext()


@torch.inference_mode()
def compare(master, exported, rows, assistant_only):
    metrics = {"examples": len(rows), "targets": 0, "top1_agreement_count": 0,
               "master_ce_sum": 0., "export_ce_sum": 0., "master_correct": 0, "export_correct": 0,
               "first_targets": 0, "master_first_ce_sum": 0., "export_first_ce_sum": 0.,
               "master_first_rank_sum": 0, "export_first_rank_sum": 0}
    for start in range(0, len(rows), 8):
        inputs, labels = batch_tensors(rows[start:start + 8])
        mask = labels != -100
        with context(True):
            logits_a = master(inputs)
        logits_b = exported(inputs)
        if not torch.isfinite(logits_a).all() or not torch.isfinite(logits_b).all():
            raise ValueError("Non-finite diagnostic logits")
        pred_a, pred_b = logits_a.argmax(-1), logits_b.argmax(-1)
        metrics["targets"] += int(mask.sum())
        metrics["top1_agreement_count"] += int(((pred_a == pred_b) & mask).sum())
        for name, logits, pred in (("master", logits_a, pred_a), ("export", logits_b, pred_b)):
            metrics[f"{name}_ce_sum"] += float(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                   labels.reshape(-1), ignore_index=-100, reduction="sum"))
            metrics[f"{name}_correct"] += int(((pred == labels) & mask).sum())
            if assistant_only:
                first = mask & F.pad(~mask[:, :-1], (1, 0), value=True)
                loc = first.nonzero(as_tuple=True)
                selected = logits[loc].float()
                gold = labels[loc]
                metrics[f"{name}_first_ce_sum"] += float(F.cross_entropy(selected, gold, reduction="sum"))
                rank = 1 + (selected > selected.gather(-1, gold[:, None])).sum(-1)
                metrics[f"{name}_first_rank_sum"] += int(rank.sum())
        if assistant_only:
            metrics["first_targets"] += int(first.sum())
        del logits_a, logits_b
    count = metrics["targets"]
    if not count:
        raise ValueError("No effective targets in diagnostic group")
    metrics["top1_agreement"] = metrics["top1_agreement_count"] / count
    for name in ("master", "export"):
        metrics[f"{name}_loss"] = metrics[f"{name}_ce_sum"] / count
        metrics[f"{name}_token_accuracy"] = metrics[f"{name}_correct"] / count
        if metrics["first_targets"]:
            metrics[f"{name}_first_loss"] = metrics[f"{name}_first_ce_sum"] / metrics["first_targets"]
            metrics[f"{name}_first_mean_rank"] = metrics[f"{name}_first_rank_sum"] / metrics["first_targets"]
    metrics["export_minus_master_loss"] = metrics["export_loss"] - metrics["master_loss"]
    return metrics


@torch.inference_mode()
def decode(model, tokenizer, prompt, use_master=False, sample=False, seed=20261122):
    tokens = list(prompt)
    generated = []
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if len(tokens) + 48 > 512:
        raise ValueError("Diagnostic generation would exceed context")
    reason = "max_new_tokens"
    for _ in range(48):
        with context(use_master):
            logits = model(torch.tensor([tokens], device="cuda"))[0, -1].float()
        if sample:
            logits = logits / 0.8
            ordered, indices = logits.sort(descending=True)
            probs = ordered.softmax(-1)
            remove = probs.cumsum(-1) - probs > 0.9
            probs[remove] = 0
            token = int(indices[torch.multinomial(probs, 1, generator=generator)])
        else:
            token = int(logits.argmax())
        generated.append(token)
        tokens.append(token)
        if token in (2, 6):
            reason = "end_token"
            break
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"text": text, "raw_text": raw, "generated_ids": generated, "stop_reason": reason,
            "diagnostics": diagnostics(text, raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["base", "sft"], required=True)
    args = parser.parse_args()
    output = args.output / f"{args.stage}_forward.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This diagnostic requires the established native BF16 CUDA environment")
    started = perf_counter()
    root = args.root
    run = "pretrain_100m_v1" if args.stage == "base" else "sft_stage1_v1"
    artifact = "base_100m_v1" if args.stage == "base" else "chat_stage1_v1"
    checkpoint_path = Path(json.loads((root / f"runs/{run}/metrics.json").read_text(encoding="utf-8"))["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    progress = checkpoint["progress"].copy()
    digest = checkpoint["tokenizer_sha256"]
    with torch.device("meta"):
        master = TinyLM(ModelConfig(**checkpoint["config"]), activation_checkpointing=False)
    master.load_state_dict(checkpoint["model"], strict=True, assign=True)
    master = master.to(device="cuda", dtype=torch.float32).eval().requires_grad_(False)
    del checkpoint
    gc.collect()
    exported, tokenizer = load_model(root / f"artifacts/{artifact}")
    exported.requires_grad_(False)
    if digest != exported.tokenizer_sha256:
        raise ValueError("Checkpoint and export use different tokenizers")
    export_report = json.loads((root / f"artifacts/{artifact}/export_report.json").read_text(encoding="utf-8"))
    if Path(export_report["checkpoint"]).resolve() != checkpoint_path.resolve():
        raise ValueError("Export was not produced from the compared checkpoint")
    fixed_cases = cases(tokenizer)
    training = sft_sample(root / "data/sft_stage1_v1/train.jsonl", 20261122)
    validation = sft_sample(root / "data/sft_stage1_v1/validation.jsonl", 20261123)
    real, mismatched = condition_pairs(validation, tokenizer)
    pt = root / "data/stage1_v1"
    groups = {
        "pt_consumed": windows(pt / "train.bin", 0, 100_000_000 - 512),
        "pt_unconsumed": windows(pt / "train.bin", 100_000_000, 128_238_640 - 513),
        "pt_validation": windows(pt / "validation.bin", 0, 1_225_935 - 513),
        "sft_train": training, "sft_validation": validation,
        "condition_real": real, "condition_wrong_user": mismatched, "diagnostic_references": fixed_cases,
    }
    measured = {}
    for name, rows in groups.items():
        measured[name] = compare(master, exported, rows, not name.startswith("pt_"))
        measured[name]["sample_ids"] = [row["id"] for row in rows]
        print(json.dumps({"stage": args.stage, "group": name, "targets": measured[name]["targets"],
                          "master_loss": measured[name]["master_loss"], "export_loss": measured[name]["export_loss"],
                          "top1_agreement": measured[name]["top1_agreement"]}), flush=True)
    generated = []
    for index, case in enumerate(fixed_cases):
        messages = ([{"role": "user", "content": case["plain_prefix"]}]
                    if args.stage == "base" else case["messages"])
        chat = encode_conversation(tokenizer, messages, generation_prompt=True)["input_ids"]
        plain = tokenizer.encode(case["plain_prefix"], add_special_tokens=False).ids
        variants = [("plain_greedy", exported, plain, False, False), ("chat_greedy", exported, chat, False, False),
                    ("plain_nucleus", exported, plain, False, True)] if args.stage == "base" else [
                    ("chat_greedy", exported, chat, False, False), ("master_chat_greedy", master, chat, True, False),
                    ("chat_nucleus", exported, chat, False, True)]
        for variant, model, prompt, is_master, sample in variants:
            row = {"case_id": case["id"], "variant": variant, "plain_prefix": case["plain_prefix"],
                   "messages": messages if "chat" in variant else None,
                   "actual_prompt_ids": prompt, "actual_prompt_text": tokenizer.decode(prompt, skip_special_tokens=False),
                   **decode(model, tokenizer, prompt, is_master, sample, 20261122 + index)}
            generated.append(row)
        print(json.dumps({"stage": args.stage, "generation_case": index + 1}), flush=True)
    result = {"stage": args.stage, "checkpoint": str(checkpoint_path), "checkpoint_progress": progress,
              "native_bf16": True, "gradient_enabled": torch.is_grad_enabled(),
              "any_parameter_requires_grad": any(p.requires_grad for p in master.parameters()) or any(p.requires_grad for p in exported.parameters()),
              "models_in_training_mode": [master.training, exported.training],
              "optimizer_created": False, "tokenizer_sha256": digest,
              "sampling_seed": 20261122, "metrics": measured, "generation": generated,
              "generation_limit": 48, "nucleus_settings": {"temperature": 0.8, "top_p": 0.9},
              "elapsed_seconds": perf_counter() - started,
              "limitations": "Small diagnostic sets, not a new held-out benchmark. Teacher forcing supplies correct earlier answer tokens. Nucleus changes decoding only; it cannot demonstrate improved knowledge. Source range/window and user length differ across groups."}
    with output.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
