"""Export a trusted local pilot checkpoint into compact bfloat16 weights."""
import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
import torch

from tinychat.model import ModelConfig, TinyLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trusted checkpoint produced by this project")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = ModelConfig(**checkpoint["config"])
    weights = {name: value.to(torch.bfloat16).contiguous()
               for name, value in checkpoint["model"].items()}
    count = sum(value.numel() for value in weights.values())
    if count != 74_400_640 or "lm_head.weight" in weights:
        raise ValueError(f"Unexpected parameter layout/count: {count}")
    path = args.output / "model.safetensors"
    save_file(weights, str(path), metadata={"format": "pt", "purpose": "short-training-pilot"})
    size = path.stat().st_size
    if size > 150_000_000:
        raise ValueError(f"Export exceeds 150 MB: {size}")
    (args.output / "config.json").write_text(json.dumps(checkpoint["config"], indent=2), encoding="utf-8")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.save(str(args.output / "tokenizer.json"))
    loaded = load_file(str(path), device="cpu")
    for name in weights:
        if not torch.equal(weights[name], loaded[name]):
            raise RuntimeError(f"Serialization changed weight: {name}")
    del checkpoint, weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device("meta"):
        model = TinyLM(config, activation_checkpointing=False)
    model.load_state_dict(loaded, assign=True)
    inference_dtype = (torch.bfloat16 if device == "cuda" and
                       torch.cuda.is_bf16_supported(including_emulation=False) else torch.float32)
    model = model.to(device=device, dtype=inference_dtype).eval()
    del loaded
    prompt = "你好，今天的天气"
    tokens = tokenizer.encode(prompt).ids
    with torch.inference_mode():
        for _ in range(24):
            inputs = torch.tensor([tokens[-config.max_seq_len:]], dtype=torch.long, device=device)
            tokens.append(int(model(inputs)[0, -1].argmax()))
    report = {"parameter_count": count, "weight_bytes": size, "weight_MB": size / 1e6,
              "dtype": "bfloat16", "reloaded_and_generated": True,
              "sample": tokenizer.decode(tokens),
              "capability_note": "Engineering pilot only; not a trained conversation model"}
    (args.output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
