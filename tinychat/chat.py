"""Generate one response from an exported stage-1 model or a local checkpoint."""
import argparse
import json
from pathlib import Path

from tinychat.inference import generate, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--messages-file", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    messages = json.loads(args.messages_file.read_text(encoding="utf-8")) if args.messages_file else [{"role": "user", "content": args.prompt}]
    model, tokenizer = load_model(args.model, args.tokenizer, args.device)
    print(json.dumps(generate(model, tokenizer, messages, args.max_new_tokens), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
