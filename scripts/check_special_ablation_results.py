"""Check completed fixed budgets and summarize observed training evidence once."""
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from scripts.evaluate_special_ablation import ARMS, ROOT, REPORT, read_jsonl, write_new
from tinychat.model import ModelConfig


def main():
    state = json.loads((REPORT / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "awaiting_independent_results_review":
        raise ValueError("Four-arm execution and evaluation are not complete")
    if state["protected_files_before"] != state["protected_files_after"]:
        raise ValueError("Original file identities changed")
    expected_phases = [(phase, arm) for arm in ARMS for phase in ("train", "export")]
    expected_phases += [("evaluate", arm) for arm in ARMS] + [("summarize", "all")]
    actual_phases = [(p["phase"], p["arm"]) for p in state["completed_phases"]]
    if actual_phases != expected_phases or any(p["exit_code"] for p in state["completed_phases"]):
        raise ValueError("A required phase is missing or failed")
    expected_config = asdict(ModelConfig(**json.loads((ROOT / "configs/pilot.json").read_text(encoding="utf-8"))))
    expected_order = np.random.default_rng(42).permutation(20000)
    base_path = str((ROOT / "runs/pretrain_100m_v1/step_00003052.pt").resolve())
    rows = {}
    shared_scheduler = shared_groups = shared_lrs = None
    for arm in ARMS:
        directory = ROOT / "runs/special_ablation_v1" / arm
        checkpoint = torch.load(directory / "step_00000313.pt", map_location="cpu", mmap=True, weights_only=False)
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        events = read_jsonl(directory / "steps.jsonl")
        updates = [e for e in events if e["event"] == "update"]
        reset = arm.startswith(("B_", "D_"))
        eos = arm.startswith(("A_", "B_"))
        targets = 265622 if eos else 245622
        settings, progress = checkpoint["settings"], checkpoint["progress"]
        factor = settings["special_token_ablation"]
        assert checkpoint["config"] == expected_config
        assert checkpoint["base_checkpoint"] == base_path
        assert np.array_equal(checkpoint["sft_order"], expected_order)
        assert settings["seed"] == 42 and settings["micro_batch"] == settings["grad_accum"] == 8
        assert settings["device"] == "cuda" and settings["checkpoint_every"] == settings["eval_every"] == 250
        assert settings["planned_examples"] == progress["examples"] == 20000
        assert settings["planned_updates"] == progress["updates"] == 313
        assert settings["planned_tokens"] == progress["tokens"] == targets
        assert factor["reset_unseen_specials"] is reset and factor["final_eos"] is eos
        assert not metrics["error"] and not metrics["optimizer_state_unsafe"]
        assert [r["step"] for r in updates] == list(range(1, 314))
        assert all(r["examples_this_update"] == 64 and r["micro_batches"] == 8 for r in updates[:-1])
        assert updates[-1]["examples_this_update"] == 32 and updates[-1]["micro_batches"] == 4
        assert sum(r["tokens_this_update"] for r in updates) == targets
        groups = [{k: v for k, v in group.items() if k != "params"} for group in checkpoint["optimizer"]["param_groups"]]
        assert checkpoint["scheduler"] == {"unit": "updates", "budget": 313, "warmup": 10, "peak": 3e-5, "minimum": 3e-6}
        assert [g["weight_decay"] for g in groups] == [0.1, 0.0]
        assert all(g["betas"] == (0.9, 0.95) and g["eps"] == 1e-8 for g in groups)
        lrs = [r["lr"] for r in updates]
        if shared_scheduler is None:
            shared_scheduler, shared_groups, shared_lrs = checkpoint["scheduler"], groups, lrs
        else:
            assert checkpoint["scheduler"] == shared_scheduler and groups == shared_groups and lrs == shared_lrs
        artifact = ROOT / "artifacts/special_ablation_v1" / arm
        exported = json.loads((artifact / "export_report.json").read_text(encoding="utf-8"))
        assert exported["parameter_count"] == 74400640 and exported["weight_bytes"] <= 150000000
        assert exported["serialized_weights_equal"] is True
        assert json.loads((artifact / "chat_format.json").read_text(encoding="utf-8"))["final_eos"] is eos
        valid = [e for e in events if e["event"] == "validation"]
        rows[arm] = {"reset": reset, "final_eos": eos, "updates": len(updates), "examples": progress["examples"],
                     "supervised_tokens": targets, "first_grad_norm": updates[0]["grad_norm"],
                     "median_grad_norm": statistics.median(e["grad_norm"] for e in updates),
                     "clipped_updates": sum(e["grad_norm"] > 1 for e in updates),
                     "last50_training_loss_mean": statistics.mean(e["loss"] for e in updates[-50:]),
                     "training_format_validation": [{k: e[k] for k in ("phase", "step", "loss", "valid_loss_tokens")} for e in valid],
                     "gradient_groups": [{k: e[k] for k in ("step", "grad_norm", "embedding_gradient_norms")}
                                         for e in updates if "embedding_gradient_norms" in e],
                     "initialization": checkpoint["special_token_initialization"],
                     "elapsed_seconds": metrics["elapsed_seconds"], "training_seconds": metrics["training_seconds"],
                     "peak_torch_allocated_bytes": max(e["peak_memory_bytes"] for e in updates),
                     "weight_bytes": exported["weight_bytes"], "tokenizer_sha256": exported["tokenizer_sha256"]}
        del checkpoint
    assert len({r["tokenizer_sha256"] for r in rows.values()}) == 1
    # Historical logs are a reproducibility reference, not a substitute for arm A.
    old = [r for r in read_jsonl(ROOT / "runs/sft_oldbase_v2/steps.jsonl") if r["event"] == "update"]
    new = [r for r in read_jsonl(ROOT / "runs/special_ablation_v1/A_keep_eos/steps.jsonl") if r["event"] == "update"]
    baseline = {"same_lr_and_token_counts": all(a["lr"] == b["lr"] and a["tokens_this_update"] == b["tokens_this_update"]
                                                for a, b in zip(old, new, strict=True)),
                "max_step_loss_absolute_difference": max(abs(a["loss"] - b["loss"]) for a, b in zip(old, new, strict=True)),
                "max_grad_norm_absolute_difference": max(abs(a["grad_norm"] - b["grad_norm"]) for a, b in zip(old, new, strict=True))}
    write_new(REPORT / "execution_check.json", {"passed": True, "arms": rows,
              "shared_scheduler": shared_scheduler, "shared_optimizer_groups": shared_groups,
              "historical_control": baseline, "original_files_preserved": True,
              "notes": "Same examples/order/updates/LR; dropping final EOS intentionally removes 20k targets, so total training CE is not cross-arm comparable."})
    print(json.dumps({arm: {k: r[k] for k in ("first_grad_norm", "median_grad_norm", "clipped_updates", "elapsed_seconds")}
                      for arm, r in rows.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
