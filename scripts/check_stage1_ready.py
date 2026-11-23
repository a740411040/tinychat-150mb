"""Record one pretraining gate after data/tokenizer are frozen."""
import argparse
import hashlib
import json
from pathlib import Path

from tinychat.chat_format import encode_conversation
from tinychat.inference import checked_tokenizer
from tinychat.training_state import read_frozen_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/stage1_v1"))
    p.add_argument("--prompts", type=Path, default=Path("evaluation/stage1_v1/prompts.jsonl"))
    p.add_argument("--output", type=Path, default=Path("reports/stage1_data_gate.json"))
    args = p.parse_args()
    manifest, identity = read_frozen_manifest(args.data_dir.resolve(), "pretrain", 16384)
    if manifest["splits"]["train"]["tokens"] < 100_000_001:
        raise ValueError("Insufficient unique stream capacity for exact100M targets")
    tokenizer, digest, _ = checked_tokenizer(manifest["tokenizer_path"], manifest["tokenizer_sha256"], 16384)
    examples = ["中文对话测试。", "Hello, world! 123.45", "你好🙂，temperature=26°C；一二三。\n第二行"]
    for example in examples:
        if tokenizer.decode(tokenizer.encode(example).ids) != example:
            raise ValueError(f"Byte-level roundtrip failed: {example!r}")
    prompt_bytes = args.prompts.read_bytes()
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
    prompt_manifest = json.loads((args.prompts.parent / "prompt_manifest.json").read_text(encoding="utf-8"))
    if (prompt_manifest.get("frozen") is not True or prompt_hash != prompt_manifest["prompts_sha256"]
            or prompt_hash != (manifest.get("evaluation_screen") or {}).get("sha256")):
        raise ValueError("Frozen evaluation input and corpus contamination screen identity differ")
    prompts = [json.loads(line) for line in prompt_bytes.decode("utf-8").splitlines() if line.strip()]
    if (len(prompts) != 300 or len({r["id"] for r in prompts}) != 300
            or sum(r["split"] == "dev" for r in prompts) != 200
            or sum(r["split"] == "test" for r in prompts) != 100):
        raise ValueError("Expected unique frozen200dev+100test prompts")
    lengths = [(r["id"], len(encode_conversation(tokenizer, r["messages"], generation_prompt=True)["input_ids"])) for r in prompts]
    too_long = [(key, length) for key, length in lengths if length + 96 > 512]
    if too_long:
        raise ValueError(f"Frozen prompts do not fit: {too_long}")
    chars, tokens, docs = 0, 0, 0
    with (args.data_dir / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line)["text"]
            chars += len(text)
            tokens += len(tokenizer.encode(text).ids)
            docs += 1
            if docs == 100:
                break
    report = {"ready": True, "tokenizer_sha256": digest, "prompts_sha256": prompt_hash, "train_tokens_available": manifest["splits"]["train"]["tokens"], "required_stream_tokens": 100_000_001, "prompt_count": len(prompts), "max_prompt_tokens": max(length for _, length in lengths), "max_generation_tokens": 96, "context_tokens": 512, "roundtrip_cases": examples, "roundtrip_pass": True, "tokenizer_sample_diagnostics": {"train_documents": docs, "characters": chars, "tokens": tokens, "tokens_per_character": tokens / chars}, "frozen_data_identity": identity}
    if args.output.exists():
        raise FileExistsError("Gate already recorded; do not rerun unchanged acceptance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frozen_data_identity"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
