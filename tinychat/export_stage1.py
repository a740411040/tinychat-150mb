"""Export only stage-1 bfloat16 weights, with the frozen tokenizer and config."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tinychat.chat_format import manifest_final_eos
from tinychat.inference import check_checkpoint, generate, load_model
from tinychat.special_token_ablation import ablation_settings, validate_reset_provenance
from tinychat.training_state import read_frozen_manifest


def export_manifest(checkpoint, data_dir, stage):
    manifest, identity = read_frozen_manifest(
        data_dir, "pretrain" if stage == "base" else "sft", checkpoint["config"]["vocab_size"]
    )
    if identity != checkpoint["data_identity"]:
        raise ValueError("Export data identity differs from the checkpoint training data")
    return manifest


def export_chat_metadata(checkpoint, manifest):
    final_eos = manifest_final_eos(manifest)
    settings = checkpoint.get("settings", {})
    factor = settings.get("special_token_ablation", ablation_settings())
    if not isinstance(factor, dict) or type(factor.get("reset_unseen_specials")) is not bool:
        raise ValueError("Invalid checkpoint special-token ablation settings")
    seed = factor.get("reset_seed")
    if (factor["reset_unseen_specials"] and (type(seed) is not int or not 0 <= seed < 2**32)
            or type(factor.get("final_eos")) is not bool
            or factor != ablation_settings(final_eos, factor["reset_unseen_specials"], seed)):
        raise ValueError("Checkpoint special-token ablation settings disagree with export format")

    chat_format = {
        "special_token_ids": manifest["special_token_ids"],
        "implementation": "tinychat.chat_format.encode_conversation",
        "context_tokens": checkpoint["config"]["max_seq_len"], "padding": "right",
        "final_eos": final_eos,
        "labels": "assistant content/end_turn/final eos only" if final_eos else "assistant content/end_turn only",
    }
    metadata = {"final_eos": final_eos}
    if "special_token_ablation" in settings:
        metadata["special_token_ablation"] = factor
    initialization = checkpoint.get("special_token_initialization")
    if factor["reset_unseen_specials"]:
        validate_reset_provenance(initialization, factor)
    if initialization is not None:
        metadata["special_token_initialization"] = initialization
    return chat_format, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["base", "sft"], required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite export {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    manifest = export_manifest(checkpoint, args.data_dir, args.stage)
    chat_format, ablation_metadata = export_chat_metadata(checkpoint, manifest)
    tokenizer_path = Path(manifest["tokenizer_path"])
    tokenizer, digest = check_checkpoint(checkpoint, tokenizer_path, completed_stage="pretrain" if args.stage == "base" else "sft")
    if digest != manifest["tokenizer_sha256"]:
        raise ValueError("Export manifest and checkpoint tokenizer identity differ")
    weights = {name: value.to(dtype=torch.bfloat16, device="cpu").contiguous() for name, value in checkpoint["model"].items()}
    params = sum(t.numel() for t in weights.values())
    if params != 74_400_640 or "lm_head.weight" in weights:
        raise ValueError(f"Unexpected parameter count or shared-weight layout: {params}")
    config = checkpoint["config"]
    args.output.mkdir(parents=True)
    path = args.output / "model.safetensors"
    save_file(weights, str(path), metadata={"format": "pt", "purpose": f"stage1-{args.stage}", "tokenizer_sha256": manifest["tokenizer_sha256"]})
    size = path.stat().st_size
    if size > 150_000_000:
        raise ValueError(f"Weight file exceeds 150 MB: {size}")
    loaded = load_file(str(path), device="cpu")
    if any(not torch.equal(weights[name], loaded[name]) for name in weights):
        raise RuntimeError("Serialized weights changed")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Copy bytes: the frozen tokenizer is never re-serialized or retrained here.
    (args.output / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    (args.output / "chat_format.json").write_text(json.dumps(chat_format, ensure_ascii=False, indent=2), encoding="utf-8")
    del checkpoint, weights, loaded, tokenizer
    model, tokenizer = load_model(args.output)
    sample = generate(model, tokenizer, [{"role": "user", "content": "你好，请简单介绍一下你自己。"}], max_new_tokens=48)
    report = {
        "stage": args.stage, "checkpoint": str(args.checkpoint.resolve()),
        "parameter_count": params, "weight_bytes": size, "weight_MB": size / 1e6,
        "dtype": "bfloat16", "tokenizer_sha256": manifest["tokenizer_sha256"],
        "serialized_weights_equal": True, "reloaded_and_generated": True,
        "sample": sample,
        "capability_note": "100M-token stage-1 experiment; consult held-out dialogue evaluation, not this single sample.",
        "data_manifest": str((args.data_dir / "manifest.json").resolve()),
        **ablation_metadata,
        "distribution_note": "Data sources and usage notes are recorded in the actual data manifest; "
                             "the stage name alone does not identify the corpus. " + manifest.get("usage", ""),
    }
    (args.output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
