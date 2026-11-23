"""One CUDA BF16 parity gate using development prompts only, before test inference."""
import argparse
import json
import time
from pathlib import Path

import torch

from tinychat.inference import generate, generate_batch, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=Path("evaluation/stage1_v1/prompts.jsonl"))
    p.add_argument("--output", type=Path, default=Path("reports/stage1_batch_gate.json"))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("The native batch gate is already recorded")
    records = {row["id"]: row for row in (json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip())}
    # Predetermined coverage of chat, facts, arithmetic, format, extraction,
    # summarization, reasoning and multi-turn context; no final-test prompts.
    ids = [f"dev_{n:03d}" for n in (1, 21, 41, 61, 81, 101, 141, 161)]
    selected = [records[key] for key in ids]
    if any(row["split"] != "dev" for row in selected):
        raise ValueError("Batch gate must only use development cases")
    conversations = [row["messages"] for row in selected]
    report = {"prompt_ids": ids, "device": "cuda", "precision": "bfloat16", "max_new_tokens": 16,
              "selection_rule": "Use B8 for both formal evaluations only if every checked native token sequence matches B1 for both models; otherwise use B1 for both. No answer-reference scoring is used to select batching.", "models": {}}
    all_equal = True
    for label, source in (("base", args.base), ("sft", args.sft)):
        model, tokenizer = load_model(source)
        if next(model.parameters()).dtype != torch.bfloat16:
            raise ValueError("Native gate expects the exported BF16 model")
        started = time.perf_counter()
        single = [generate(model, tokenizer, messages, 16) for messages in conversations]
        single_seconds = time.perf_counter() - started
        started = time.perf_counter()
        batch = generate_batch(model, tokenizer, conversations, 16)
        batch_seconds = time.perf_counter() - started
        cases = [{"id": key, "ids_equal": a["generated_ids"] == b["generated_ids"], "single_ids": a["generated_ids"], "batch_ids": b["generated_ids"]} for key, a, b in zip(ids, single, batch)]
        matched = len(batch) == len(ids) and all(row["ids_equal"] for row in cases)
        all_equal = all_equal and matched
        report["models"][label] = {"tokenizer_sha256": model.tokenizer_sha256, "all_checked_ids_equal": matched, "single_wall_seconds": single_seconds, "batch_wall_seconds": batch_seconds, "cases": cases}
        print(json.dumps({"model": label, "matched": matched, "single_seconds": single_seconds, "batch_seconds": batch_seconds}), flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()
    report["adopted_batch_size"] = 8 if all_equal else 1
    report["limitations"] = "This is a targeted parity check, not a guarantee of bitwise agreement across every prefix, GPU or kernel. Timings are a short functional probe, not a warmed inference benchmark."
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"adopted_batch_size": report["adopted_batch_size"], "report": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
