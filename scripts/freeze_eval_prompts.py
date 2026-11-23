"""Create fixed 200 development + 100 independently authored test prompts."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def development():
    rows = []
    def add(category, prompt, rubric, expected=None, history=None):
        rows.append({"id": f"dev_{len(rows)+1:03d}", "split": "dev", "category": category, "messages": [*(history or []), {"role": "user", "content": prompt}], "rubric": rubric, "expected": expected})
    chat = ["第一次参加读书会有点紧张，怎样开场？", "周末想在家休息，请给两个简单活动。", "朋友帮我搬了书，帮我写一句感谢的话。", "今天任务太多，怎样先排出顺序？", "我想每天读十分钟书，给我一个容易坚持的方法。", "室友总忘记关灯，怎样礼貌提醒？", "我要向老师请教问题，第一句话怎么说？", "下午容易走神，给我两个不花钱的办法。", "邻居送来水果，写一句自然的感谢。", "朋友临时取消聚会，我有些失望，怎样回复？", "想把书桌收拾整齐，从哪里开始？", "我想学习做饭，怎样选第一道菜？", "明天要介绍自己，怎样开头比较自然？", "最近总拖到最后才交作业，给一个具体建议。", "我准备给朋友寄明信片，写一句问候。", "一周没运动了，怎样安排一次轻松的活动？", "给刚搬家的朋友写一句祝福。", "我想减少看手机的时间，先做哪一步？", "朋友考试前紧张，帮我写一句安慰。", "午休只有半小时，怎样安排比较轻松？"]
    for text in chat:
        add("everyday_chat", text, "中文自然、切题，给出简短可执行建议或符合场景的话，不捏造用户经历。")
    facts = [("一年通常有多少个月？", "12"), ("一周有多少天？", "7"), ("太阳通常从哪个方向升起？", "东"), ("地球的天然卫星叫什么？", "月球"), ("水在标准大气压下通常多少摄氏度结冰？", "0"), ("中国的首都是什么城市？", "北京"), ("三角形有几条边？", "3"), ("正方形有几条边？", "4"), ("植物进行光合作用主要需要哪种光源？", "太阳"), ("一小时是多少分钟？", "60"), ("一米等于多少厘米？", "100"), ("红绿灯中，通常哪种颜色表示停止？", "红"), ("蜜蜂采集花蜜主要来自植物的哪个部分？", "花"), ("冰融化后变成什么？", "水"), ("人用什么器官听声音？", "耳"), ("企鹅是鸟类还是鱼类？", "鸟"), ("地球是行星还是恒星？", "行星"), ("长江最终流入哪个海？", "东海"), ("汉字“木”一般有几画？", "4"), ("四季中，春季后面是什么季节？", "夏")]
    # Keep development facts distinct from the independently authored test facts.
    facts[0] = ("闰年的二月有多少天？", "29")
    facts[1] = ("一天有多少小时？", "24")
    facts[4] = ("蜡烛燃烧需要空气中的哪种气体？", "氧")
    facts[5] = ("大熊猫主要吃哪种植物？", "竹")
    for prompt, answer in facts:
        add("factual_basic", prompt, f"基本事实正确；参考答案：{answer}（文字数字同样可接受）。", answer)
    for i in range(20):
        a, b = 13 + i * 3, 2 + i
        add("arithmetic", f"请计算 {a} 加 {b}，只写结果。", f"结果为{a+b}，不添加解释。", str(a+b))
    nouns = ["苹果", "书包", "桌子", "河流", "月亮", "树叶", "铅笔", "小路", "海风", "花园", "茶杯", "书店", "星星", "雨伞", "公交车", "竹子", "纸船", "晨光", "窗户", "白云"]
    for noun in nouns:
        add("instruction_format", f"请用“{noun}”造两个简短句子，分两行，每行以数字序号开头。", f"恰好两行有效句子，分别有数字序号，两句都包含{noun}。")
    for i, item in enumerate(nouns):
        add("extraction", f"记录：物品={item}；编号={301+i}；颜色=蓝色。请只回答编号。", f"只输出{301+i}。", str(301+i))
    places = ["图书馆", "公园", "学校", "社区", "书店", "广场", "教室", "小区", "河边", "车站"]
    for i in range(20):
        place, day = places[i % 10], "星期六" if i < 10 else "星期日"
        add("summarization", f"用一句话概括：{day}上午，志愿者在{place}整理图书。他们把旧书分类，方便大家借阅。活动在中午结束。", f"一句话包含志愿者在{place}整理图书的主要事件，不添加原文没有的事实。")
    raw = ["把窗户关上。", "给我借一下这本书。", "你快点回复。", "把音量调小。", "帮我看一下这段文字。", "给我一张纸。", "告诉我会议时间。", "你坐过去一点。", "把门打开。", "帮我把杯子递过来。", "你把这个地方打扫一下。", "给我讲一遍步骤。", "发我一下地址。", "把桌上的书收好。", "你别挡住通道。", "帮我确认一下名字。", "告诉我怎么走。", "把文件再发一遍。", "给我留一个座位。", "你等我两分钟。"]
    for sentence in raw:
        add("rewriting", f"把下面这句话改得礼貌一些，保持原意：{sentence}", "保留原本请求，语气更礼貌、自然，不添加无关背景。")
    for i in range(20):
        names = [f"小{c}" for c in ["林", "陈", "周"]]
        add("simple_reasoning", f"{names[0]}有{9+i}张卡片，{names[1]}比{names[0]}多2张，{names[2]}比{names[1]}少1张。谁的卡片最多？", "回答小陈，比较关系正确。", "小陈")
    for i, item in enumerate(nouns):
        add("multi_turn", "我刚才说要买什么，预算是多少？请简短回答。", f"记住物品{item}和预算{20+i*5}元；不能只重复助手的话。", history=[{"role": "user", "content": f"我准备买{item}，预算是{20+i*5}元。"}, {"role": "assistant", "content": "好的，我记住你的购买计划了。"}])
    unknowns = ["我桌上现在放着几本书？", "我明天那场面试几点开始？", "我家门口现在下雨了吗？", "你知道我上周见了谁吗？", "我现在穿的衣服是什么颜色？", "我把钥匙放到哪里了？", "请告诉我今天楼下餐馆的菜单。", "我的快递现在到哪里了？", "我手机剩余多少电量？", "我刚才没发出来的那条消息写了什么？", "他这样做是什么意思？", "帮我改一下。", "那个地方好不好？", "这件事应该选哪个？", "明天适合出门吗？", "按刚才的要求写，但我还没告诉你要求。", "帮我安排一段行程，但我还没说目的地。", "请评价这篇文章，但我还没有贴文章。", "解释这张图片，但我没有发送图片。", "请比较这两本书，但我没有给出书名。"]
    for prompt in unknowns:
        add("uncertainty", prompt, "明确缺少哪些信息或提出相关澄清问题，不编造看见的内容、实时数据或用户私密信息。")
    assert len(rows) == 200
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("evaluation/stage1_v1"))
    p.add_argument("--test-author-task", required=True)
    args = p.parse_args()
    tests = [json.loads(line) for line in args.test_input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tests) != 100 or any(x["split"] != "test" for x in tests):
        raise ValueError("Expected exactly100 held-out test prompts")
    categories = collections.Counter(x["category"] for x in tests)
    if len(categories) != 10 or set(categories.values()) != {10}:
        raise ValueError(f"Test category imbalance: {categories}")
    rows = development() + tests
    if len({r["id"] for r in rows}) != 300:
        raise ValueError("Duplicate IDs")
    signatures = [json.dumps(r["messages"], ensure_ascii=False, sort_keys=True) for r in rows]
    if len(set(signatures)) != len(rows):
        raise ValueError("Duplicate conversations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prompts.jsonl").open("x", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    metadata = {"frozen": True, "dev_count": 200, "test_count": 100, "test_author": "Kimi via approved read-only safe entry", "test_author_task": args.test_author_task, "dev_author": "deterministic curated project script", "decoding": {"greedy": True, "max_new_tokens": 96}, "quality_target": {"test_readable_relevant_main_instruction_rate_min": 0.6, "severe_corruption_empty_repetition_rate_max": 0.05}, "note": "Custom small-model diagnostics, not a standardized benchmark. Some development cases are correlated templates; category results and blind test grading must be separate. Test prompts are frozen before SFT and are never used to select hyperparameters in this stage."}
    metadata["prompts_sha256"] = hashlib.sha256((args.output_dir / "prompts.jsonl").read_bytes()).hexdigest()
    (args.output_dir / "prompt_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"prompts": len(rows), "test_categories": dict(categories)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
