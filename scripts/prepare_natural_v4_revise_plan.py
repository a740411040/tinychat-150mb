"""Prepare only five already-public practice examples for an independent plan."""
from pathlib import Path
from scripts.natural_v4_scale_queue import DATA, read
from scripts.natural_v4_rewrite_pilot import write_new

ISSUES = {
    "scale-prefix-082192": "自然类误收材料问答；据文答复增加了材料未写的预警/可再生能源做法。",
    "scale-prefix-080318": "自然类误收明确文本转换指令；没有完整执行标点替换，漏了‘两周，包括’中的逗号。",
    "scale-prefix-014945": "虽有‘刚才提到’的措辞，末问自己完整写出儒家文化，泛定义并不依赖前文；不能计真实多轮。",
    "scale-prefix-097320": "Smoltalk源问题存在混乱混语/粘贴污染；被作为可用多轮接受。",
    "scale-prefix-094118": "Smoltalk源问题存在混乱混语/粘贴污染；被作为可用多轮接受。",
}
rows = {}
for path in sorted((DATA / "queue/probe_v1").glob("batch_*/result.json")):
    for row in read(path)["rows"]:
        if row["id"] in ISSUES:
            rows[row["id"]] = {
                "id": row["id"], "route_hint": row["category"],
                "original_public_messages": [{"role": message["role"], "content": message["content"]}
                                             for message in row["original_messages"]],
                "rewritten_messages": [{"role": message["role"], "content": message["content"]}
                                       for message in row["messages"]],
                "independent_review_issue": ISSUES[row["id"]],
            }
if set(rows) != set(ISSUES):
    raise ValueError("Missing a selected public example; do not send an incomplete packet")
target = Path("planning/train_150mb/natural_sft_v4_scale20k_revise_plan_input.txt")
write_new(target, {"scope": "Only five public-corpus practice questions and their teacher-authored rewrites; no private model evaluation or secrets.",
                   "records": list(rows.values())})
print({"examples": len(rows), "bytes": target.stat().st_size})
