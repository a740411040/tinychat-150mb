"""Prepare and instrument the fixed stage-1 special-token/EOS ablation."""

import argparse
import json
import math
from pathlib import Path

import torch

from .chat_format import IGNORE_LABEL, SPECIAL_IDS, manifest_final_eos
from .training_state import (
    SPLITS, atomic_json, file_identity, load_json, read_json, read_frozen_manifest, verify_file_identity,
)


DEFAULT_RESET_SEED = 1729
RESET_RECIPE = "ordinary_mean_capped_10pct_orthogonal_cpu_gaussian_v1"
GRADIENT_STEPS = frozenset({1, 2, 25, 100, 250, 313})
PAD = SPECIAL_IDS["<|pad|>"]
EOS = SPECIAL_IDS["<|eos|>"]
END_TURN = SPECIAL_IDS["<|end_turn|>"]
FIRST_ORDINARY = len(SPECIAL_IDS)
RESET_IDS = tuple(index for index in SPECIAL_IDS.values() if index not in (PAD, EOS))


def ablation_settings(final_eos=True, reset_unseen_specials=False, reset_seed=DEFAULT_RESET_SEED):
    return {
        "final_eos": final_eos,
        "reset_unseen_specials": reset_unseen_specials,
        "reset_seed": reset_seed if reset_unseen_specials else None,
        "reset_recipe": RESET_RECIPE if reset_unseen_specials else None,
    }


def _special_geometry(rows):
    norms = rows.norm(dim=1)
    normalized = torch.nn.functional.normalize(rows, dim=1)
    return {"row_norms": norms.tolist(), "pairwise_cosines": (normalized @ normalized.T).tolist()}


def _finite_values(values):
    return (isinstance(values, list) and len(values) == FIRST_ORDINARY
            and all(type(value) in (float, int) and math.isfinite(value) for value in values))


def validate_reset_provenance(record, factor):
    error = "Checkpoint is missing valid frozen special-token reset provenance"
    if (not isinstance(record, dict) or type(record.get("seed")) is not int
            or record["seed"] != factor["reset_seed"] or record.get("recipe") != factor["reset_recipe"]):
        raise ValueError(error)
    for phase in ("before", "after"):
        geometry = record.get(phase)
        if not isinstance(geometry, dict) or not _finite_values(geometry.get("row_norms")):
            raise ValueError(error)
        cosines = geometry.get("pairwise_cosines")
        if (not isinstance(cosines, list) or len(cosines) != FIRST_ORDINARY
                or not all(_finite_values(row) for row in cosines)):
            raise ValueError(error)
    required_ids = [PAD, *RESET_IDS] if record["before"]["row_norms"][PAD] else list(RESET_IDS)
    changed_ids = record.get("changed_ids")
    if (not isinstance(changed_ids, list) or any(type(index) is not int for index in changed_ids)
            or changed_ids != required_ids):
        raise ValueError(error)


def embedding_geometry(model):
    weights = model.token_embedding.weight.detach().cpu()
    return {
        **_special_geometry(weights[:FIRST_ORDINARY].to(torch.float64)),
        "ordinary_mean_row_norm": weights[FIRST_ORDINARY:].float().norm(dim=1).mean().item(),
    }


@torch.no_grad()
def reset_unseen_specials(model, seed=DEFAULT_RESET_SEED):
    """Reset the six unseen tied rows with a private CPU RNG and return their initial geometry."""
    weight = model.token_embedding.weight
    if weight.dtype != torch.float32 or weight.shape[0] <= FIRST_ORDINARY:
        raise ValueError("Special-token reset requires FP32 embeddings with ordinary rows")
    ordinary = weight[FIRST_ORDINARY:].detach().cpu()
    target_norm = ordinary.norm(dim=1).mean().item()
    if not 0 < target_norm < float("inf"):
        raise ValueError("Special-token reset requires a finite positive ordinary mean row norm")
    mean = ordinary.mean(dim=0, dtype=torch.float64)
    mean_norm = mean.norm().item()
    common = mean
    basis = []
    if mean_norm:
        mean_direction = mean / mean_norm
        basis.append(mean_direction)
        common = mean_direction * min(mean_norm, 0.1 * target_norm)
    if weight.shape[1] < len(basis) + len(RESET_IDS):
        raise ValueError("Embedding dimension too small for orthogonal special-token reset")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = (target_norm ** 2 - common.square().sum().item()) ** 0.5
    before = weight[:FIRST_ORDINARY].detach().to(device="cpu", dtype=torch.float64)
    for index in RESET_IDS:
        direction = torch.randn(mean.shape, generator=generator, dtype=torch.float64)
        for axis in basis:
            direction = direction - direction.dot(axis) * axis
        direction = torch.nn.functional.normalize(direction, dim=0)
        basis.append(direction)
        weight[index].copy_(common + scale * direction)
    weight[PAD].zero_()

    after = weight[:FIRST_ORDINARY].detach().to(device="cpu", dtype=torch.float64)
    return {
        "recipe": RESET_RECIPE,
        "seed": seed,
        "changed_ids": [index for index in range(FIRST_ORDINARY) if not torch.equal(before[index], after[index])],
        "common_norm": common.norm().item(),
        "ordinary_mean_row_norm": target_norm,
        "before": _special_geometry(before),
        "after": _special_geometry(after),
    }


def embedding_gradient_norms(model):
    gradient = model.token_embedding.weight.grad.detach()
    return {
        "special_rows": gradient[:FIRST_ORDINARY].norm(dim=1).tolist(),
        "ordinary": gradient[FIRST_ORDINARY:].norm().item(),
    }


def prepare_no_eos_data(source, output):
    """Derive frozen SFT rows by removing only the final supervised EOS pair."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Derived SFT output must be outside the frozen source directory")
    manifest, identity = read_frozen_manifest(source, "sft", read_json(source / "manifest.json")["vocab_size"])
    if not manifest_final_eos(manifest):
        raise ValueError("Expected a frozen SFT source with supervised final EOS")
    output.mkdir(parents=True)

    for split in SPLITS:
        details = manifest["splits"][split]
        filename = f"{split}.jsonl"
        source_path, target_path = source / filename, output / filename
        documents = tokens = assistant_tokens = 0
        min_context = max_context = None
        with source_path.open(encoding="utf-8") as original, target_path.open("x", encoding="utf-8") as derived:
            for line_number, line in enumerate(original, 1):
                row = load_json(line)
                ids, labels = row["input_ids"], row["labels"]
                if len(ids) != len(labels) or ids[-2:] != [END_TURN, EOS] or labels[-2:] != [END_TURN, EOS]:
                    raise ValueError(f"Invalid final supervised EOS at {source_path}:{line_number}")
                row["input_ids"], row["labels"] = ids[:-1], labels[:-1]
                length = len(row["input_ids"])
                documents += 1
                tokens += length
                assistant_tokens += sum(label != IGNORE_LABEL for label in row["labels"][1:])
                min_context = length if min_context is None else min(min_context, length)
                max_context = length if max_context is None else max(max_context, length)
                derived.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if (documents, tokens + documents, assistant_tokens + documents) != (
            details["documents"], details["tokens"], details["assistant_tokens"]
        ):
            raise ValueError(f"Source SFT totals differ from frozen manifest: {source_path}")
        details.update(tokens=tokens, assistant_tokens=assistant_tokens,
                       min_context=min_context, max_context=max_context)
        manifest["files"][filename] = file_identity(target_path)

    verify_file_identity(identity)
    original_corpus = manifest["corpus_id"]
    manifest["corpus_id"] = output.name
    manifest["chat_format"] = {**manifest.get("chat_format", {}), "final_eos": False}
    manifest["derived_from"] = {"manifest": identity["manifest"], "corpus_id": original_corpus}
    manifest["supervision"] = "Assistant content and assistant end markers per Stage 1 format"
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_no_eos_data(args.source, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "final_eos": False, "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
