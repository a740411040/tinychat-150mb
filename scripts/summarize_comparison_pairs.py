"""Paired exploratory statistics for two models on the same frozen test cases."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np


def paired_statistics(rows, old="old_v2", new="clean_v2", replicates=10000):
    by_model = {model: {r["id"]: r for r in rows if r["model"] == model} for model in (old, new)}
    counts = Counter(r["model"] for r in rows)
    ids = sorted(by_model[old])
    if len(ids) != 100 or set(ids) != set(by_model[new]) or counts != Counter({old: 100, new: 100}):
        raise ValueError("Expected exactly100 uniquely paired scores for each named model")
    if any(by_model[old][key]["category"] != by_model[new][key]["category"] for key in ids):
        raise ValueError("Paired categories differ")
    categories = sorted({r["category"] for r in by_model[old].values()})
    groups = [np.array([i for i, key in enumerate(ids) if by_model[old][key]["category"] == category])
              for category in categories]
    rng = np.random.default_rng(20261122)
    indices = np.concatenate([rng.choice(group, (replicates, len(group)), replace=True) for group in groups], axis=1)
    result = {"models": {"old": old, "new": new}, "cases": 100, "bootstrap_replicates": replicates,
              "bootstrap_method": "paired case bootstrap stratified by task category; percentile95%; fixed seed20261122",
              "limitations": "Exploratory custom diagnostic. Model-judge uncertainty and correlated tasks are not captured. Sparse or zero discordance can produce degenerate bootstrap intervals; these do not establish equivalence or population certainty."}
    for name, value in (("success", lambda r: r["score"] >= 3), ("readable", lambda r: r["readable"]),
                        ("severe_failure", lambda r: r["severe_failure"]), ("score", lambda r: r["score"])):
        previous = np.array([value(by_model[old][key]) for key in ids], dtype=float)
        current = np.array([value(by_model[new][key]) for key in ids], dtype=float)
        differences = current - previous
        ci = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975]).tolist()
        metric = {"old_mean": float(previous.mean()), "new_mean": float(current.mean()),
                  "new_minus_old": float(differences.mean()), "exploratory_paired_ci95": ci,
                  "new_higher_cases": int((differences > 0).sum()), "old_higher_cases": int((differences < 0).sum()),
                  "ties": int((differences == 0).sum())}
        if name == "success":
            wins, losses = metric["new_higher_cases"], metric["old_higher_cases"]
            discordant = wins + losses
            metric["mcnemar_exact_two_sided_p"] = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2 ** discordant)) if discordant else 1.0
        result[name] = metric
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("evaluation/comparison_v2"))
    args = p.parse_args()
    path = args.root / "blind_review/unblinded_scores.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = paired_statistics(rows)
    output = args.root / "blind_review/paired_statistics.json"
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "success": result["success"]}))


if __name__ == "__main__":
    main()
