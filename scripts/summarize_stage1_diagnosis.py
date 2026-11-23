"""Summarize preserved read-only diagnostics and project measured training cost."""
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path.cwd()
    folder = root / "reports/diagnosis_v1"
    forward = {s: read(folder / f"{s}_forward.json") for s in ("base", "sft")}
    groups = tuple(forward["base"]["metrics"])
    for group in groups:
        if forward["base"]["metrics"][group]["sample_ids"] != forward["sft"]["metrics"][group]["sample_ids"]:
            raise ValueError(f"Different comparison sample in {group}")
    result = {"forward": {}, "learning_curves": {}, "cost": {}}
    for stage, report in forward.items():
        variant_counts = {}
        for row in report["generation"]:
            count = variant_counts.setdefault(row["variant"], {"cases": 0, "severe_repetition": 0, "max_length": 0})
            count["cases"] += 1
            count["severe_repetition"] += row["diagnostics"]["severe_repetition"]
            count["max_length"] += row["stop_reason"] == "max_new_tokens"
        metrics = report["metrics"]
        pairs = []
        if stage == "sft":
            pairs = [r["case_id"] for r in report["generation"] if r["variant"] == "chat_greedy" and
                     r["generated_ids"] != next(s["generated_ids"] for s in report["generation"]
                     if s["case_id"] == r["case_id"] and s["variant"] == "master_chat_greedy")]
        result["forward"][stage] = {
            "max_absolute_group_ce_difference": max(abs(m["export_minus_master_loss"]) for m in metrics.values()),
            "pt_validation_minus_consumed_ce": metrics["pt_validation"]["master_loss"] - metrics["pt_consumed"]["master_loss"],
            "sft_validation_minus_train_ce": metrics["sft_validation"]["master_loss"] - metrics["sft_train"]["master_loss"],
            "wrong_user_minus_real_user_ce": metrics["condition_wrong_user"]["master_loss"] - metrics["condition_real"]["master_loss"],
            "condition_pair_count": metrics["condition_real"]["examples"],
            "generation_counts": variant_counts, "precision_generation_different_ids": pairs,
            "no_training_guard": {k: report[k] for k in ("gradient_enabled", "any_parameter_requires_grad", "models_in_training_mode", "optimizer_created")},
        }
    for run in ("pretrain_100m_v1", "sft_stage1_v1"):
        events = [json.loads(line) for line in (root / f"runs/{run}/steps.jsonl").read_text(encoding="utf-8").splitlines()]
        updates = [r for r in events if r["event"] == "update"]
        validation = [{k: r.get(k) for k in ("phase", "step", "loss", "valid_loss_tokens")} for r in events if r["event"] == "validation"]
        norms = np.array([r["grad_norm"] for r in updates])
        result["learning_curves"][run] = {"validation": validation, "updates": len(updates),
            "grad_norm_percentiles_p0_p50_p95_p100": np.percentile(norms, [0, 50, 95, 100]).tolist(),
            "updates_clipped_fraction": float((norms > 1).mean()),
            "nonfinite_updates": sum(bool(r["nonfinite"]) for r in updates),
            "last_100_updates_target_weighted_loss": sum(r["loss"] * r["tokens_this_update"] for r in updates[-100:]) / sum(r["tokens_this_update"] for r in updates[-100:]),
            "first_lr": updates[0]["lr"], "last_lr": updates[-1]["lr"]}
    pt = read(root / "runs/pretrain_100m_v1/metrics.json")
    sft = read(root / "runs/sft_stage1_v1/metrics.json")
    speed = pt["cumulative_valid_loss_tokens"] / pt["elapsed_seconds"]
    result["cost"] = {"measured_wall_targets_per_second": speed,
        "parameters": 74400640, "current_tokens_per_parameter": 100000000 / 74400640,
        "scenario_speed_floor": 18000,
        "range_basis": "Baseline uses measured whole-run wall speed including validation/checkpoints; slower bound assumes18k targets/s. These are estimates, not measured future runs. Data preparation excluded.",
        "scenarios": [{"total_tokens": total, "additional_tokens": total - 100000000,
                       "hours_at_measured_speed": (total - 100000000) / speed / 3600,
                       "hours_at_18k_speed": (total - 100000000) / 18000 / 3600}
                      for total in (300000000, 500000000, 1000000000, 1500000000)],
        "sft_20000_example_one_epoch_minutes_measured": sft["elapsed_seconds"] / 60,
        "sft_20000_example_two_epochs_minutes_linear": sft["elapsed_seconds"] / 30}
    with (folder / "diagnosis_summary.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"forward": result["forward"], "cost": result["cost"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
