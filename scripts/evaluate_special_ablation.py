"""Bounded, frozen local evaluation for the special-token/EOS factorial experiment."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
from time import perf_counter

import torch
from torch.nn import functional as F

from tinychat.chat_format import IGNORE_LABEL, encode_conversation
from tinychat.evaluate_stage1 import diagnostics
from tinychat.inference import generate, load_model
from tinychat.train_stage1 import sft_batch
from tinychat.verified_sft_checks import verify_dialogue
from scripts.freeze_natural_sft_prompts import score_machine

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/special_ablation_v1"
ARMS = ("A_keep_eos", "B_reset_eos", "C_keep_noeos", "D_reset_noeos")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def freeze():
    verified = read_jsonl(ROOT / "data/sft_verified_v2/test.jsonl")
    families = sorted({r["task_family"] for r in verified})
    if len(verified) != 500 or len(families) != 20:
        raise ValueError("Unexpected original test split")
    selected = []
    for family in families:
        selected.extend(sorted((r for r in verified if r["task_family"] == family), key=lambda r: r["id"])[:5])
    prompts = [{"id": "verified:" + r["id"], "source_id": r["id"], "kind": "verified",
                "category": r["task_family"], "messages": r["messages"][:-1],
                "reference": r["messages"][-1]["content"]} for r in selected]
    natural = sorted((r for r in read_jsonl(ROOT / "evaluation/natural_sft_v3/prompts.jsonl")
                      if r["split"] == "dev"), key=lambda r: r["id"])
    if len(natural) != 100 or set(Counter(r["category"] for r in natural).values()) != {10}:
        raise ValueError("Unexpected natural development split")
    for row in natural:
        prompts.append({**row, "source_id": row["id"], "id": "natural:" + row["id"], "kind": "natural"})
    probes = json.loads((ROOT / "reports/diagnosis_v2_resume/probes.json").read_text(encoding="utf-8"))
    if len(probes) != 24:
        raise ValueError("Unexpected historical probe set")
    prompts.extend({**r, "source_id": r["id"], "id": "probe:" + r["id"], "kind": "probe",
                    "category": r["group"]} for r in probes)
    if len(prompts) != 224 or len({r["id"] for r in prompts}) != 224:
        raise ValueError("Prompt selection count/IDs inconsistent")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / "data/stage1_v1/tokenizer.json"))
    lengths = [len(encode_conversation(tokenizer, r["messages"], generation_prompt=True)["input_ids"])
               for r in prompts]
    if max(lengths) + 96 > 512:
        raise ValueError("Frozen prompt exceeds generation context")
    review_ids = [r["id"] for category in sorted({r["category"] for r in prompts if r["kind"] == "natural"})
                  for r in sorted((r for r in prompts if r["kind"] == "natural" and r["category"] == category),
                                  key=lambda r: r["id"])[:2]]
    packet = {"frozen": True, "counts": dict(Counter(r["kind"] for r in prompts)),
              "selection": "verified first5 sorted IDs/family; all natural DEV; existing24 probes",
              "max_prompt_tokens": max(lengths), "decoding": {"greedy": True, "max_new_tokens": 96,
              "batch_size": 1, "stop_ids": [2, 6]}, "review_ids": review_ids,
              "natural_test_used": False, "prompts": prompts}
    write_new(REPORT / "frozen_evaluation.json", packet)
    print(json.dumps({k: v for k, v in packet.items() if k not in {"prompts", "review_ids"}}, ensure_ascii=False))


def valid_generated_structure(generated_ids):
    content = generated_ids[:-1] if generated_ids and generated_ids[-1] in {2, 6} else generated_ids
    return all(token >= 7 for token in content)


def score_verified(source, text, generated_ids=None):
    if generated_ids is not None and not valid_generated_structure(generated_ids):
        return False
    trial = copy.deepcopy(source["messages"])
    trial[-1]["content"] = text.strip()
    try:
        verify_dialogue(trial, source["split"], source["task_family"], source["style"], source["prompt_identity"])
        return True
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError):
        return False


def loss_masks(inputs, labels):
    valid = labels != IGNORE_LABEL
    content = valid & (labels != 2) & (labels != 6)
    return {"content": content, "end_turn": valid & (labels == 6),
            "final_eos": valid & (labels == 2), "first_content": content & (inputs == 5)}


@torch.inference_mode()
def validation_components(model):
    # Every arm uses the ORIGINAL labels here, regardless of its training format.
    original = read_jsonl(ROOT / "data/sft_verified_v2/validation.jsonl")
    rows = [(torch.tensor(r["input_ids"]), torch.tensor(r["labels"]),
             sum(label != IGNORE_LABEL for label in r["labels"][1:])) for r in original]
    totals = {name: [0.0, 0] for name in ("content", "end_turn", "final_eos", "first_content")}
    device = next(model.parameters()).device
    for offset in range(0, len(rows), 8):
        batch = sft_batch(rows[offset:offset + 8])
        inputs, labels = batch.inputs.to(device), batch.labels.to(device)
        logits = model(inputs)
        nll = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none",
                              ignore_index=IGNORE_LABEL).reshape_as(labels)
        for name, mask in loss_masks(inputs, labels).items():
            totals[name][0] += float(nll[mask].sum())
            totals[name][1] += int(mask.sum())
        del logits, nll
    return {name: {"loss": total / count, "tokens": count} for name, (total, count) in totals.items()}


def embedding_stats(model):
    weight = model.token_embedding.weight.detach().float().cpu()
    ids = [0, 1, 3, 4, 5, 6]
    vectors = F.normalize(weight[ids], dim=-1)
    cos = vectors @ vectors.T
    return {"ordinary_mean_norm": float(weight[7:].norm(dim=-1).mean()),
            "special_norms": weight[:7].norm(dim=-1).tolist(),
            "unused_ids": ids, "unused_cosine_matrix": cos.tolist(),
            "assistant_end_turn_cosine": float(F.cosine_similarity(weight[5:6], weight[6:7]).item())}


def generate_probed(model, tokenizer, messages):
    first = {}

    def hidden_hook(_module, _inputs, output):
        if "hidden_cosine" not in first:
            first["hidden_cosine"] = float(F.cosine_similarity(output[0, -2].float(),
                                                              output[0, -1].float(), dim=0))

    def output_hook(_module, _inputs, output):
        if "eos_probability" not in first:
            logits = output[0, -1].float()
            probabilities = logits.softmax(-1)
            first.update({"eos_probability": float(probabilities[2]),
                          "end_turn_probability": float(probabilities[6]),
                          "eos_rank": int((logits > logits[2]).sum()) + 1,
                          "end_turn_rank": int((logits > logits[6]).sum()) + 1,
                          "top_id": int(logits.argmax()), "eos_logit": float(logits[2])})

    handles = [model.final_norm.register_forward_hook(hidden_hook), model.register_forward_hook(output_hook)]
    try:
        result = generate(model, tokenizer, messages, max_new_tokens=96)
    finally:
        for handle in handles:
            handle.remove()
    return result, first


def aggregate(rows):
    groups = {"all": rows}
    for kind in sorted({r["kind"] for r in rows}):
        groups[kind] = [r for r in rows if r["kind"] == kind]
    for kind, category in sorted({(r["kind"], r["category"]) for r in rows}):
        groups[f"{kind}:{category}"] = [r for r in rows if r["kind"] == kind and r["category"] == category]
    summary = {}
    for name, group in groups.items():
        n = len(group)
        scored = [r for r in group if r["correct"] is not None]
        summary[name] = {"count": n, "empty": sum(r["diagnostics"]["empty"] for r in group),
                         "repetition": sum(r["diagnostics"]["severe_repetition"] for r in group),
                         "role_leak": sum(r["diagnostics"]["role_or_padding_leak"] for r in group),
                         "length_limit": sum(r["stop_reason"] == "max_new_tokens" for r in group),
                         "first_eos": sum(r["first"]["top_id"] == 2 for r in group),
                         "first_end_turn": sum(r["first"]["top_id"] == 6 for r in group),
                         "mean_first_eos_probability": sum(r["first"]["eos_probability"] for r in group) / n,
                         "mean_output_tokens": sum(r["output_tokens"] for r in group) / n,
                         "scored": len(scored), "correct": sum(r["correct"] for r in scored)}
    return summary


def evaluate_arm(arm):
    target = REPORT / arm
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation {target}")
    frozen = json.loads((REPORT / "frozen_evaluation.json").read_text(encoding="utf-8"))
    if frozen.get("frozen") is not True:
        raise ValueError("Evaluation was not frozen")
    sources = {r["id"]: r for r in read_jsonl(ROOT / "data/sft_verified_v2/test.jsonl")}
    artifact = ROOT / "artifacts/special_ablation_v1" / arm
    model, tokenizer = load_model(artifact)
    target.mkdir(parents=True)
    started = perf_counter()
    geometry = embedding_stats(model)
    validation = validation_components(model)
    rows = []
    with (target / "outputs.jsonl").open("x", encoding="utf-8", newline="\n") as file:
        for index, prompt in enumerate(frozen["prompts"], 1):
            generated, first = generate_probed(model, tokenizer, prompt["messages"])
            if prompt["kind"] == "verified":
                correct = score_verified(sources[prompt["source_id"]], generated["text"], generated["generated_ids"])
            elif prompt["kind"] == "natural":
                correct = score_machine(prompt, generated["text"])
            else:
                reference = prompt.get("reference")
                correct = generated["text"].strip() == reference.strip() if reference is not None else None
            if correct is not None and not valid_generated_structure(generated["generated_ids"]):
                correct = False
            row = {"id": prompt["id"], "kind": prompt["kind"], "category": prompt["category"],
                   **generated, "first": first, "correct": correct,
                   "diagnostics": diagnostics(generated["text"], generated["raw_text"])}
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()
            rows.append(row)
            if index % 25 == 0 or index == len(frozen["prompts"]):
                print(json.dumps({"event": "evaluation_progress", "arm": arm, "done": index,
                                  "total": len(frozen["prompts"]), "elapsed_seconds": perf_counter() - started}), flush=True)
    summary = {"arm": arm, "artifact": str(artifact), "decoding": frozen["decoding"],
               "metrics": aggregate(rows), "common_validation": validation, "embedding": geometry,
               "elapsed_seconds": perf_counter() - started,
               "limitations": "Single-seed diagnostic; natural semantic cases are not auto-scored; probe exact-match is diagnostic."}
    write_new(target / "summary.json", summary)


def effects(values):
    a, b, c, d = [values[arm] for arm in ARMS]
    return {"reset_main_effect": ((b - a) + (d - c)) / 2,
            "remove_eos_main_effect": ((c - a) + (d - b)) / 2,
            "interaction": d - c - b + a}


def summarize_all():
    frozen = json.loads((REPORT / "frozen_evaluation.json").read_text(encoding="utf-8"))
    summaries = {arm: json.loads((REPORT / arm / "summary.json").read_text(encoding="utf-8")) for arm in ARMS}
    rows = {arm: {r["id"]: r for r in read_jsonl(REPORT / arm / "outputs.jsonl")} for arm in ARMS}
    expected = {r["id"] for r in frozen["prompts"]}
    if any(set(group) != expected for group in rows.values()):
        raise ValueError("Arms do not share complete identical evaluation IDs")
    contrasts = {}
    for kind in ("all", "verified", "natural", "probe"):
        contrasts[kind] = {}
        for metric in ("empty", "repetition", "role_leak", "first_eos", "mean_first_eos_probability"):
            values = {arm: summaries[arm]["metrics"][kind][metric] for arm in ARMS}
            if metric != "mean_first_eos_probability":
                values = {arm: value / summaries[arm]["metrics"][kind]["count"] for arm, value in values.items()}
            contrasts[kind][metric] = {"arm_rates": values, **effects(values)}
    values = {arm: summaries[arm]["metrics"]["verified"]["correct"] / 100 for arm in ARMS}
    contrasts["verified"]["correct"] = {"arm_rates": values, **effects(values)}
    result = {"arms": summaries, "factorial_effects": contrasts,
              "interpretation": "Raw signed differences: lower empty/repetition is better, higher correctness is better; single seed, descriptive only."}
    write_new(REPORT / "comparison.json", result)
    prompts = {r["id"]: r for r in frozen["prompts"]}
    rng = random.Random(20261122)
    packet, mapping = [], {}
    for prompt_id in frozen["review_ids"]:
        arm_order = list(ARMS)
        rng.shuffle(arm_order)
        for arm in arm_order:
            review_id = f"review-{len(packet)+1:03d}"
            prompt, answer = prompts[prompt_id], rows[arm][prompt_id]
            packet.append({"review_id": review_id, "prompt_id": prompt_id, "category": prompt["category"],
                           "messages": prompt["messages"], "rubric": prompt["rubric"], "answer": answer["text"],
                           "raw_answer": answer["raw_text"]})
            mapping[review_id] = {"arm": arm, "prompt_id": prompt_id}
    write_new(REPORT / "blind_review_packet.json", packet)
    write_new(REPORT / "blind_review_map.json", mapping)
    print(json.dumps({arm: summaries[arm]["metrics"]["natural"] for arm in ARMS}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "model", "summarize"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "model":
        if args.arm is None:
            parser.error("model requires --arm")
        evaluate_arm(args.arm)
    else:
        summarize_all()


if __name__ == "__main__":
    main()
