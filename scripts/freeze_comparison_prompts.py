"""Validate and freeze independently authored test cases plus separate development cases."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer
from tinychat.chat_format import encode_conversation

CATEGORIES = ("everyday_chat", "factual_qa", "arithmetic", "instruction_following", "text_rewrite",
              "information_extraction", "classification", "multi_turn", "clarification", "grounded_reasoning")


def development_rows():
    cases = [
        ("everyday_chat", "借了同事的保温袋，归还时说一句礼貌的话。", "表达归还和感谢，中文自然。", None),
        ("everyday_chat", "我刚开始养一盆薄荷，想把每次浇水记下来。给一个简单记录办法。", "给出可操作的记录办法，不编造养护条件。", None),
        ("factual_qa", "看一本普通纸质书时，封面后面一张张供阅读的纸通常叫什么？", "回答书页或页面，接受同义表达。", "书页"),
        ("factual_qa", "水结成冰，是从液态变成什么状态？", "回答固态。", "固态"),
        ("arithmetic", "仓库原有46个纸箱，运走17个，还剩几个？只写数字。", "仅回答29。", "29"),
        ("arithmetic", "每盒有8支蜡笔，4盒一共多少支？", "正确回答32支。", "32"),
        ("instruction_following", "用两行写出‘准备’和‘完成’，每行只写一个词。", "第一行准备，第二行完成，不添加其他文字。", "准备\n完成"),
        ("instruction_following", "回答中只允许一个汉字：白天看到的太阳属于恒星还是行星？恒星答‘恒’，行星答‘行’。", "仅回答恒。", "恒"),
        ("text_rewrite", "把‘立刻把清单给我’改成礼貌的请求，意思不变。", "保留立即索取清单的意思，改成礼貌请求，如请现在把清单给我，可以吗。", None),
        ("text_rewrite", "把‘小岚关上了窗户’改为被字句。", "窗户被小岚关上了，语义不变。", "窗户被小岚关上了。"),
        ("information_extraction", "通知：手工课于周四下午三点在二楼画室开始。只提取地点。", "仅回答二楼画室。", "二楼画室"),
        ("information_extraction", "订单：收件人许禾，物品茶杯，数量6。请只写数量。", "仅回答6。", "6"),
        ("classification", "将‘毛巾、芒果、枕头’中属于水果的词写出来。", "仅识别芒果为水果。", "芒果"),
        ("classification", "‘这张椅子坐着很舒服’表达的是满意还是不满？", "回答满意。", "满意"),
        ("multi_turn", [{"role": "user", "content": "我的钥匙放在绿色盒子里。"}, {"role": "assistant", "content": "好的，钥匙在绿色盒子里。"}, {"role": "user", "content": "我说钥匙在哪儿？"}], "依据历史回答绿色盒子里。", "绿色盒子"),
        ("multi_turn", [{"role": "user", "content": "请记住：这次活动叫‘松果日’，我负责海报。"}, {"role": "assistant", "content": "记住了，你负责松果日的海报。"}, {"role": "user", "content": "我负责哪项工作？"}], "依据历史回答海报。", "海报"),
        ("clarification", "帮我把那份文件改一下。", "请求提供文件和修改要求，不假装已经修改。", None),
        ("clarification", "替我查一下它的尺寸。", "先询问所指物品，不凭空给尺寸。", None),
        ("grounded_reasoning", "规定：只有佩戴访客证才能进入展厅。方宁没有访客证。按规定她能进入吗？", "不能进入，依据给定规则。", "不能"),
        ("grounded_reasoning", "甲箱比乙箱轻，乙箱比丙箱轻。哪个箱子最重？", "回答丙箱。", "丙箱"),
    ]
    return [{"id": f"v2_dev_{i:03d}", "split": "dev", "category": category,
             "messages": content if isinstance(content, list) else [{"role": "user", "content": content}],
             "rubric": rubric, "expected": expected}
            for i, (category, content, rubric, expected) in enumerate(cases, 1)]


def user_key(row):
    return re.sub(r"\s+", "", "\n".join(m["content"] for m in row["messages"] if m["role"] == "user"))


def editorial_corrections(rows):
    """Resolve independent review findings before any prompt set is frozen."""
    replacements = {
        "v2_test_006": ("铅笔里的笔芯主要由石墨还是金属铅制成？", "回答石墨；接受说明还混有黏土，不应回答金属铅。", "石墨", "Replace exact reuse of old evaluation question."),
        "v2_test_009": ("蜘蛛有几条腿？", "准确回答八条腿。", "八条", "Avoid ambiguous bamboo woody/herbaceous classification."),
        "v2_test_063": ("原有24张贴纸，送走7张，还剩多少张？", None, None, "Supply initial quantity rather than consumed quantity."),
        "v2_test_059": ("钟面上，分针走完一整圈通常经过多少分钟？", "正确回答60分钟或一小时。", "60分钟", "Replace exact reuse of old evaluation question."),
        "v2_test_082": ("“花盆土太湿，该减少浇水吗？”属于“浇水问题、工具清点、座位安排”中的哪一类？只答类别。", None, None, "Make topic explicit without inferring cause from yellow leaves."),
        "v2_test_095": ("帮我把这个长度换算一下。", None, None, "Make unit conversion intent explicit."),
        "v2_test_096": ("图书角有甲、乙两车。甲车只放待修补书，乙车只放待归架书。《纸船》封面破损，等待修补；《谷仓》已修好，等待归架。问：两本书分别放哪车？", None, None, "State both books' pending handling status explicitly."),
        "v2_test_097": ("两盆植物的摆放要求是：石莲放明亮窗边，蕨草放半阴散射光处。现在石莲在半阴处，蕨草在明亮窗边。怎样调整才能符合要求？", "按给定要求互换两盆植物的位置。", "石莲放明亮窗边，蕨草放半阴散射光处。", "Replace unsupported dark-cabinet horticulture with explicit grounded requirements."),
        "v2_test_100": ("长桌最多只能放80厘米长的托盘。现有60厘米和90厘米的托盘各一个，不能叠放。问：哪个能单独放在长桌上？", None, None, "State maximum length instead of an ambiguous example capacity."),
    }
    edits = []
    for row in rows:
        if row["id"] in {"v2_test_042", "v2_test_072"}:
            old = json.loads(json.dumps(row, ensure_ascii=False))
            if row["id"] == "v2_test_042":
                row["rubric"] = "先澄清票种等必要信息，不假装订好票；接受分步澄清，不强制所有票都问行程。"
            else:
                row["expected"] = "光照充足让这株茉莉开了花。"
            edits.append({"id": row["id"], "old": old, "new": json.loads(json.dumps(row, ensure_ascii=False)), "reason": "Align rubric/reference with permissible correct responses."})
        if row["id"] == "v2_test_037":
            old = row["messages"][-1]["content"]
            row["messages"][-1]["content"] = "我原定的窑场行程具体是什么时间？只答时间。"
            edits.append({"id": row["id"], "old": old, "new": row["messages"][-1]["content"], "reason": "Align requested precision with reference."})
        if row["id"] in replacements:
            content, rubric, expected, reason = replacements[row["id"]]
            old = json.loads(json.dumps(row, ensure_ascii=False))
            row["messages"] = [{"role": "user", "content": content}]
            if rubric is not None:
                row["rubric"] = rubric
                row["expected"] = expected
            edits.append({"id": row["id"], "old": old, "new": json.loads(json.dumps(row, ensure_ascii=False)), "reason": reason})
    return edits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, nargs=2, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / "evaluation/comparison_v2"
    if (target / "prompts.jsonl").exists():
        raise FileExistsError("Evaluation already frozen")
    tests, provenance = [], []
    for result_path in args.results:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result["taskCompleted"] or not result["executionPass"] or result.get("providerErrors"):
            raise ValueError(f"Author did not complete successfully: {result_path}")
        rows = [json.loads(line) for line in result["finalText"].strip().splitlines() if line.strip()]
        if len(rows) != 50:
            raise ValueError(f"Expected exactly50 authored rows: {len(rows)}")
        if any(row["split"] != "test" for row in rows):
            raise ValueError("Author output must contain test rows only")
        tests.extend(rows)
        provenance.append({"task_id": result["taskId"], "result_path": str(result_path)})
    if {r["id"] for r in tests} != {f"v2_test_{i:03d}" for i in range(1, 101)}:
        raise ValueError("Incorrect test ID coverage")
    if Counter(r["category"] for r in tests) != Counter({c: 10 for c in CATEGORIES}):
        raise ValueError("Incorrect category counts")
    corrections = editorial_corrections(tests)
    rows = development_rows() + sorted(tests, key=lambda r: r["id"])
    tokenizer = Tokenizer.from_file(str(root / "data/stage1_v1/tokenizer.json"))
    old_rows = [json.loads(line) for line in (root / "evaluation/stage1_v1/prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    old_keys = {user_key(r) for r in old_rows}
    keys, lengths = set(), []
    duplicates = []
    for row in rows:
        if row["split"] not in {"dev", "test"} or not isinstance(row["rubric"], str) or not row["rubric"].strip():
            raise ValueError(f"Invalid row {row['id']}")
        if not row["messages"] or row["messages"][-1]["role"] != "user":
            raise ValueError(f"Invalid generation conversation: {row['id']}")
        for index, message in enumerate(row["messages"]):
            if message["role"] != ("user" if index % 2 == 0 else "assistant") or not message["content"].strip():
                raise ValueError(f"Invalid role/content: {row['id']}")
        if (row["category"] == "multi_turn") != (len(row["messages"]) >= 3):
            raise ValueError(f"Incorrect multi-turn category: {row['id']}")
        key = user_key(row)
        if key in keys or key in old_keys:
            duplicates.append(row["id"])
        keys.add(key)
        length = len(encode_conversation(tokenizer, row["messages"], generation_prompt=True)["input_ids"])
        if length + 96 > 512:
            raise ValueError(f"Prompt too long: {row['id']}")
        lengths.append(length)
    if duplicates:
        raise ValueError(f"Duplicated or old prompts: {duplicates}")
    target.mkdir(parents=True, exist_ok=True)
    path = target / "prompts.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"frozen": True, "dev_count": 20, "test_count": 100,
                "test_author": "Kimi via approved read-only entry", "test_author_provenance": provenance,
                "pre_freeze_editorial_corrections": "pre_freeze_corrections.json; independently reviewed ambiguity fixes before any training/evaluation",
                "dev_author": "root curated independent development prompts", "max_prompt_tokens": max(lengths),
                "decoding": {"greedy": True, "max_new_tokens": 96, "batch_size": 1},
                "quality_target": {"test_readable_relevant_main_instruction_rate_min": 0.6,
                                   "severe_corruption_empty_repetition_rate_max": 0.05},
                "note": "Custom diagnostic, not standardized benchmark. Frozen before training; no test-driven tuning. No exact user-prompt reuse from old evaluation. Machine data-conflict filtering only; semantic task overlap remains possible.",
                "prompts_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (target / "prompt_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "pre_freeze_corrections.json").write_text(json.dumps(corrections, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "dev": 20, "test": 100, "max_prompt_tokens": max(lengths)}))


if __name__ == "__main__":
    main()
