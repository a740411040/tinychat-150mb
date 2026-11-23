"""Evaluate both fixed prompt sets without tuning generation on held-out answers."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

from tinychat.inference import generate_batch, load_model


def diagnostics(text, raw_text, expected=None):
    compact = re.sub(r"\s+", "", text)
    grams = [compact[i:i + 4] for i in range(max(0, len(compact) - 3))]
    repetition = 1 - len(set(grams)) / len(grams) if grams else 0.0
    cjk = sum("\u3400" <= c <= "\u9fff" for c in compact)
    leaked = any(marker in raw_text for marker in ["<|pad|>", "<|bos|>", "<|system|>", "<|user|>", "<|assistant|>"])
    return {
        "empty": not compact,
        "role_or_padding_leak": leaked,
        "repetition_4gram_fraction": repetition,
        "severe_repetition": bool(re.search(r"(.)\1{7,}", compact)) or (len(grams) >= 20 and repetition >= 0.5),
        "replacement_character_count": text.count("\ufffd"),
        "cjk_character_fraction": cjk / len(compact) if compact else 0.0,
        "reference_substring_present": (re.sub(r"\s+", "", str(expected)) in compact) if expected is not None else None,
    }


def summarize(rows):
    groups = {"all": rows}
    for key in ("split", "category"):
        for value in sorted({r[key] for r in rows}):
            groups[f"{key}:{value}"] = [r for r in rows if r[key] == value]
    result = {}
    for name, group in groups.items():
        n = len(group)
        result[name] = {
            "prompts": n,
            "empty_rate": sum(r["diagnostics"]["empty"] for r in group) / n,
            "severe_repetition_rate": sum(r["diagnostics"]["severe_repetition"] for r in group) / n,
            "role_or_padding_leak_rate": sum(r["diagnostics"]["role_or_padding_leak"] for r in group) / n,
            "max_length_rate": sum(r["stop_reason"] == "max_new_tokens" for r in group) / n,
            "mean_output_tokens": sum(r["output_tokens"] for r in group) / n,
            "mean_repetition_4gram_fraction": sum(r["diagnostics"]["repetition_4gram_fraction"] for r in group) / n,
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--batch-size", type=int, choices=range(1, 17), default=8)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    started = perf_counter()
    if not 1 <= args.max_new_tokens <= 256:
        p.error("--max-new-tokens must be within 1..256")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {args.output}")
    prompt_bytes = args.prompts.read_bytes()
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
    prompt_manifest = json.loads((args.prompts.parent / "prompt_manifest.json").read_text(encoding="utf-8"))
    if not prompt_manifest.get("frozen") or prompt_hash != prompt_manifest["prompts_sha256"]:
        raise ValueError("Evaluation prompts differ from the frozen set")
    prompts = [json.loads(line) for line in prompt_bytes.decode("utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("Frozen evaluation prompts are empty")
    if len({r["id"] for r in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt IDs")
    model, tokenizer = load_model(args.model, args.tokenizer, args.device)
    # Check all lengths before writing any outputs; do not silently change prompts.
    from tinychat.chat_format import encode_conversation
    for record in prompts:
        n = len(encode_conversation(tokenizer, record["messages"], generation_prompt=True)["input_ids"])
        if n + args.max_new_tokens > model.config.max_seq_len:
            raise ValueError(f"Evaluation prompt too long: {record['id']} ({n})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        for offset in range(0, len(prompts), args.batch_size):
            batch = prompts[offset:offset + args.batch_size]
            results = generate_batch(model, tokenizer, [record["messages"] for record in batch], args.max_new_tokens)
            batch_id = offset // args.batch_size + 1
            for record, generated in zip(batch, results, strict=True):
                row = {"id": record["id"], "split": record["split"], "category": record["category"],
                       "model_label": args.label, "batch_id": batch_id,
                       "timing_scope": "concurrent_batch_wall_until_row_finished", **generated}
                row["diagnostics"] = diagnostics(row["text"], row["raw_text"], record.get("expected"))
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                rows.append(row)
            out.flush()
            completed = len(rows)
            if completed // 10 != offset // 10 or completed == len(prompts):
                print(json.dumps({"event": "evaluation_progress", "model": args.label,
                                  "completed": completed, "total": len(prompts)}, ensure_ascii=False), flush=True)
    summary = {
        "model_label": args.label,
        "model_source": str(args.model.resolve()),
        "prompts": str(args.prompts.resolve()),
        "prompts_sha256": prompt_hash,
        "tokenizer_sha256": model.tokenizer_sha256,
        "batch_size": args.batch_size,
        "decoding": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens, "context": model.config.max_seq_len},
        "metrics": summarize(rows),
        "limitations": "Automatic metrics detect empty/repetition/template artifacts only; reference-substring presence is diagnostic, not correctness. Human/blind rubric assessment is reported separately. This custom prompt set is not a standardized benchmark.",
    }
    summary["elapsed_seconds_total"] = perf_counter() - started
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "overall": summary["metrics"]["all"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
