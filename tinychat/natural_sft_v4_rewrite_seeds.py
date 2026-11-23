"""Materialize and verify manually selected local questions for the 64-item pilot."""

import argparse
import json
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from itertools import combinations, islice
import sqlite3

from tokenizers import Tokenizer

from .chat_format import check_special_ids
from .natural_sft_v4_semantic_pool import (
    CMRC_PREFIX, CMRC_SEPARATOR, MAX_TOKENS, OUTPUT as POOL_OUTPUT, ROOT, TOKENIZER,
    json_rows, parquet_rows, relative, verify_example, write_json,
)
from .stage1_data import normalized_content


REPORT = ROOT / "reports/natural_sft_v4"
DEST = ROOT / "data/natural_sft_v4_rewrite_pilot/seeds.jsonl"
SEED_REPORT = REPORT / "rewrite_pilot/seed_manifest.json"
PROMPT_BANKS = tuple(ROOT / "evaluation/natural_sft_v3" / name for name in (
    "prompts_for_overlap.jsonl", "freeze_v1/prompts_for_overlap.jsonl",
))
COUNTS = {"natural": 22, "grounded": 16, "multi": 16, "instruction": 10}


@dataclass(frozen=True)
class Choice:
    prefix: int
    constraints: tuple[str, ...]
    anchors: tuple[str, ...] = ()
    subtype: str | None = None
    evidence: tuple[str, str, tuple[str, ...]] | None = None
    regression: str | None = None


CHOICES = {
    "natural": [
        Choice(4352, ("给出可制作的新意礼物和实际做法；题目没有预算、对象和场合信息。",), regression="quality_round2_review.txt:15"),
        Choice(92454, ("回应反复忘带钥匙的日常困扰，建议应能实施。",)),
        Choice(94703, ("回应用户跟小明去公园玩后的开心心情；不要声称在场。",)),
        Choice(28204, ("礼貌并明确地婉拒邀请，可给自然话术；题目没有要求虚构理由。",), regression="quality_round2_review.txt:16"),
        Choice(92481, ("照顾想养宠物和长辈担心卫生的两边，提供有条件的家庭商量建议。",)),
        Choice(92655, ("回应备考压力，给一个可执行的日常步骤；不作诊断或成绩保证。",)),
        Choice(18808, ("说明移动支付的实际付款过程，包括常见扫码方式与核对确认；避免依赖某版本按钮位置。",), regression="semantic_round2_review.txt:19"),
        Choice(92786, ("回应搬新家的开心和参观邀请；不要声称能到场。",)),
        Choice(92888, ("围绕几个好友在家聚会，提供简便安排，不假设具体人数。",)),
        Choice(93126, ("回应单位洗手间没有人打扫的实际困扰；仅凭这个现象不能判断原清洁人员是否离职。",)),
        Choice(93593, ("回应模拟联合国准备时的紧张，建议聚焦当前活动的准备动作。",)),
        Choice(93196, ("回应对工作成就感的担忧，建议基于真实经历梳理进展；不承诺结果。",)),
        Choice(10238, ("围绕初学做饭，给容易上手的练习路径；不假定已经有特定厨具。",)),
        Choice(95322, ("回应因为忙而少见朋友的失落，建议兼顾时间少的现实。",)),
        Choice(95480, ("回应开始编织的兴趣，介绍可上手的内容；不要编造自己的体验。",)),
        Choice(6656, ("给适用于一般同事冲突的沟通步骤；不假定谁对谁错。",)),
        Choice(95413, ("用户吉他练习遇到瓶颈，给可执行的练习调整；不保证练习成效。",)),
        Choice(96541, ("回应重逢聊不到一起的尴尬，解释可能的情境变化；不读心或定性关系。",)),
        Choice(98086, ("回应摄影学习中的比较和灰心，提供可实践的反馈或练习办法。",)),
        Choice(92759, ("建议携带方便的野餐食物；不要假定可以现场加热或冷藏。",)),
        Choice(94185, ("回应降温时是否换季整理衣橱，考虑过渡天气。",)),
        Choice(92353, ("回应累到不想做晚饭的具体处境，给低负担选择，避免医嘱。",)),
    ],
    "grounded": [
        Choice(34415, ("只依据故事。文中只说衣服无法脱下，没有尺寸、损坏或其他机制的信息，不能判定原因。",),
               evidence=("文本内容：", "\n问题：", ("他穿上了一件衣服，却发现自己无法脱下来。",))),
        Choice(84786, ("从《绿宝石失窃案》的故事梗概回答疑团的触发事件。",)),
        Choice(34457, ("从本文呈现的方法作答，覆盖各项；不要加材料以外的方法。",),
               evidence=("给一段文本内容：", "\n根据这个文本内容回答问题", ("以下是四项小技巧",))),
        Choice(82616, ("定位白荡湖水产的产量比较；不外推到其他产地。",)),
        Choice(34828, ("按三问顺序回答。角色及事件可由描述确定；描述没有片名或唯一标识，不凭类似情节猜片名。",),
               evidence=("描述：", "\n问题1:", ("医生发现他可以在某种程度上预知未来",))),
        Choice(84098, ("绑定到专辑《大海》的音乐风格；不把商业表现当风格。",)),
        Choice(34831, ("区分有兴趣、考虑和已承诺；保留 Lucy 在对话末尾的态度。",),
               evidence=("", "\n根据以上这段对话内容回答", ("看起来很有趣，我会考虑一下。",))),
        Choice(85973, ("根据材料解释太保指什么；区分职官与具体担任者。",)),
        Choice(34948, ("从对话中小明对新同学的举动判断性格特点；可引用行为，不推测长期性格。",),
               evidence=("对话内容：", "\n问题：", ("我可以带你一起学习哦。",))),
        Choice(85994, ("提取原文举例的基建项目组成部分；区分例子与工程划分级别。",)),
        Choice(35040, ("沿用小李对小明的建议，避免补写材料外的课程或学习计划。",),
               evidence=("", "\n根据以上这段对话内容回答", ("你应该把理论和实践结合起来。",))),
        Choice(88659, ("据材料说明罗弗敦群岛出名的原因，避免列一串与问句无关的数据。",)),
        Choice(35407, ("找回叙述者吃蛋糕时许的完整愿望；保持叙述对象。",),
               evidence=("给一段文本内容：", "\n根据这个文本内容回答问题", ("我许了一个愿望",))),
        Choice(88661, ("依据首句说明赛道的位置；不把测试者或车厂所在国当地点。",)),
        Choice(35500, ("定位与当地人交流这条建议的原有理由；不补充未经材料支持的效果。",),
               evidence=("给一段文本内容：", "\n根据这个文本内容回答问题", ("2. 和当地人交流。",))),
        Choice(91928, ("绑定非洲猬目的别名，区分同段中的候选分类名及分支名称。",)),
    ],
    "multi": [
        Choice(4575, (
            "提出同一趟旅行的三地安排；不要编造时长、价格或个人出发地。",
            "在三地要求下按时间和预算限制提出较现实的上海、杭州、苏州版本；题面没有金额、天数、出发地和日期，假设须明示。",
            "沿用上海、杭州、苏州安排，给可定位的住宿区域选择；明确条件，不把示例预算当实时交通或房价。",
        ), ("从首轮的三地行程出发，第二问要求按时间和预算缩小范围。未给金额或天数。", "“这个计划”指第二轮的上海、杭州、苏州行程；未给日期和出发地，不得假定实际价格。"),
               "daily", regression="semantic_round2_review.txt:20"),
        Choice(94718, (
            "回答缺少量勺时如何提高调味一致性；第一答必须介绍矿泉水瓶盖作为可校准的临时量具。",
            "瓶盖尺寸和实际装填量未知，盐、酱油的体积转质量还取决于调料；说明如何对手边这个盖实测，不填通用克数或毫升数。"
        ), ("“你说的矿泉水瓶盖”要求第一答明确说到矿泉水瓶盖。",), "daily"),
        Choice(92576, ("回应用户描述的早冷午热，并留下可供后问回应的穿衣建议。",
                       "按同一天的较大温差和只想要一件衣服的新约束提供可行建议，说明现实取舍。"),
               ("“听着真麻烦”接第一轮的应对方案；“从早到晚的温差”来自首问。",), "daily"),
        Choice(28842, ("写一篇虚构的宠物狗故事，正文需要有可提取的宠物狗名。",
                       "只提取你刚写的故事里实际出现的全部宠物狗名。"),
               ("“这篇文章”指第一答撰写的宠物狗故事；提取目标必须能在新正文找到。",), "document"),
        Choice(92622, ("回应阴雨下的晾衣困扰，第一答提到烘干设备，避免假装观察用户当地天气。",
                       "按阴雨和大件久晾的条件提供换气、晾法等可执行建议；不保证具体干燥时间。"),
               ("“只能靠速干机”承接第一答的烘干设备，长时间晾不干源自首问的阴雨。",), "daily"),
        Choice(92867, ("用户只记得一点旋律；先提出可试音乐识别，不能凭空猜歌。",
                       "给识别工具未能找到时的后备路径，围绕有限的旋律线索，不重复同一步。"),
               ("“我来试试用音乐识别软件”要求第一答明确提到这个途径；后问是路径失败后的分支。",), "daily"),
        Choice(93051, ("回应餐厅有一道菜感觉异常的用户，不杜撰自己的亲历或原因；可问当时怎样处理。",
                       "给下次在餐厅遇到同样情况的具体沟通步骤或话术，不推定食品问题的原因。"),
               ("“不太好意思说”“这种情况”依赖前文的菜品体验及是否向店家开口。",), "daily"),
        Choice(31648, ("写完整短科幻故事，落实遇到外星文明并建立联系；留下可辨识的核心角色。",
                       "根据第一答描述主要角色并提炼主题，两部分都完成。"),
               ("“这篇文章”指第一答所写科幻故事；只使用新故事的情节与角色。",), "document"),
        Choice(93071, ("回应工作压力，给实用放松建议，不诊断或保证。",
                       "在不便出门且想更省力的约束下，调整前轮计划，给马上可做的具体小动作。"),
               ("“你说的都挺好”承接第一答的放松方式；第二答是在其基础上降低行动门槛。",), "daily"),
        Choice(94194, ("围绕天气不错想外出给简便穿搭思路，并给活动地点留出分支。",
                       "落实公园休闲和只有白运动鞋两项约束；回应想穿得有特色的顾虑。"),
               ("用户说“我决定去公园”时，前一答应已经把去处作为穿搭条件。",), "daily"),
        Choice(98876, ("围绕未来家居演讲推荐能吸引听众的具体应用切入点。",
                       "在场地设备有限的情况下调整手机控制智能设备的互动构想；不给出未经确认的平台或设备。"),
               ("“互动环节”由首轮的智能家居演讲延伸，第二轮需承接前一答的场景。",), "daily"),
        Choice(71297, ("写父亲因谋生而与儿子异地多年、最后重逢的短篇，围绕亲情，并写入可直接摘录的主题句。",
                       "从刚写的文章里原样提取一句主题句。"),
               ("“这篇文章”指第一答的故事，第二答只能摘该正文中存在的句子。",), "document"),
        Choice(24958, ("交付介绍学校课程安排的页面设计，清楚呈现课程信息。",
                       "在原页加入便于查找的联系方式，不虚构学校真实号码或地址，保留课程安排功能。"),
               ("“还想加上”要求延续前一轮的学校课程页面，不能交付孤立联系方式清单。",), "daily"),
        Choice(19496, ("给增强写作能力的方法，第一答需确实涉及读书和练习。",
                       "按用户排除已有的读书和练习，提供不同的方法，不以改名方式重复。"),
               ("“除了读书、练习”引用第一答的两项方法并设为排除项。",), "daily"),
        Choice(26791, ("给满三种利用技术提高日常工作效率的具体用法，不预测收益。",
                       "按用户不愿花太多时间学习的新条件，给前述技术的低门槛做法或明确替换关系。"),
               ("“这些技术”“它们”必须指向第一答列出的三项技术及其使用任务。",), "daily"),
        Choice(2036, ("以寓言形式完整演绎《老鼠嫁女儿》，保持可辨识的故事；避免额外断言唯一版本。",
                      "基于第一答实际发生的故事细节提供启发。"),
               ("“这个故事”指上一轮的寓言正文，启发必须能回指本次故事情节。",), "document"),
    ],
    "instruction": [
        Choice(5505, ("把给定句子改成正式汉语，保留现在的不确定性和将尝试的承诺。",)),
        Choice(27671, ("只把原陈述改成一般疑问句，保留对象与属性。",)),
        Choice(661, ("按表中姓名栏提取所有员工姓名，保持拼写，不混入部门。",)),
        Choice(21647, ("从给定数列选出最大的三个数，不补写题外数字。",)),
        Choice(36577, ("合成一个连贯句子并对至少一个子句使用同义表达，保留散步、天气和阳光信息。",)),
        Choice(36845, ("更简洁地表达想换新手机且旧机仍可用的转折关系。",)),
        Choice(2491, ("用中文概括会议的焦点和达成的意见；不补写会议结果。",)),
        Choice(2506, ("把英语短句译成自然中文；表达很快到达，不需要字面拆解。",)),
        Choice(35323, ("给所示歌词拟一个合适的短标题；不要补写歌词或声称识别出原曲。",)),
        Choice(35821, ("只依据所列条件输出可直接用的购物清单；覆盖规格、数量和尺码。",)),
    ],
}


def batch_choices():
    positions = Counter()
    for batch in range(1, 9):
        increments = (3, 2, 2, 1) if batch <= 6 else (2, 2, 2, 2)
        for category, amount in zip(COUNTS, increments):
            start = positions[category]
            positions[category] += amount
            for choice in CHOICES[category][start:start + amount]:
                yield batch, category, choice
    if positions != Counter(COUNTS) or any(len(CHOICES[category]) != count for category, count in COUNTS.items()):
        raise ValueError("Manual selections differ from pilot quotas")


def selected_pool_rows(choices):
    wanted = {choice.prefix for _, _, choice in choices}
    rows = {}
    for row in json_rows(POOL_OUTPUT / "pool.jsonl"):
        key = int(row["id"].removeprefix("prefix-"))
        if key in wanted:
            rows[key] = row
    if len(wanted) != 64 or set(rows) != wanted:
        raise ValueError(f"Pool selectors missing or repeated: {sorted(wanted - rows.keys())}")
    return rows


def evidence_offsets(user, source, choice):
    transform = source["transform"]
    if transform["type"] == "cmrc_verbatim_full_context_and_question_wrapper":
        start = len(CMRC_PREFIX)
        end = start + transform["passage_chars_zero_based_half_open"][1]
        offset_left, offset_right = transform["answer_chars_zero_based_half_open"]
        return {
            "user_message_index": 0, "material_chars_zero_based_half_open": [start, end],
            "anchor_chars_zero_based_half_open": [[start + offset_left, start + offset_right]],
            "source_type": "cmrc_complete_context",
        }
    start_mark, end_mark, anchors = choice.evidence
    start = user.index(start_mark) + len(start_mark)
    end = user.index(end_mark, start)
    spans = []
    for anchor in anchors:
        left = user.index(anchor, start, end)
        spans.append([left, left + len(anchor)])
    return {
        "user_message_index": 0, "material_chars_zero_based_half_open": [start, end],
        "anchor_chars_zero_based_half_open": spans,
        "source_type": "verbatim_user_supplied_material",
    }


def make_seed(index, batch, category, choice, row):
    turns = len(choice.constraints)
    option = row["prefix_options"][turns - 1]
    options = row["prefix_options"][:turns]
    messages = row["messages"][:option["message_count"]]
    assistants = [i for i, item in enumerate(messages) if item["role"] == "assistant"]
    users = [item for item in messages if item["role"] == "user"]
    fixed_tokens = option["tokens"] - option["supervised_tokens"] + turns
    if len(users) != turns or len(assistants) != turns or len(choice.anchors) != turns - 1:
        raise ValueError(f"Turn coverage is inconsistent for {row['id']}")
    if fixed_tokens > 350:
        raise ValueError(f"Fixed prompt exceeds 350 tokens for {row['id']}: {fixed_tokens}")
    source = row["sources"][0]
    seed = {
        "id": f"rewrite-{index:03}", "pilot_batch": batch, "category": category,
        "source_group": row["group_id"], "pool_prefix_id": row["id"], "sources": row["sources"],
        "source_license": source.get("license", source.get("usage")),
        "original_messages": messages, "user_messages": users,
        "selected_original_message_range_zero_based_half_open": [0, len(messages)],
        "rewrite_assistant_turn_indices": assistants,
        "constraints": [{"assistant_message_index": i, "review_focus": note} for i, note in zip(assistants, choice.constraints)],
        "dependency_anchors": [
            {"user_message_index": assistants[i] - 1, "history_through_assistant_message_index": assistants[i - 1], "anchor": note}
            for i, note in enumerate(choice.anchors, 1)
        ],
        "max_context": MAX_TOKENS, "final_eos": False, "original_prefix_tokens": option["tokens"],
        "tokens_without_assistant_content": fixed_tokens, "assistant_content_token_budget": MAX_TOKENS - fixed_tokens,
        "regression_seed": choice.regression is not None, "old_assistant_text_status": "unverified_reference_only",
    }
    if choice.regression:
        seed["regression_report"] = f"reports/natural_sft_v4/{choice.regression}"
    if choice.subtype:
        seed["multi_subtype"] = choice.subtype
    if category == "grounded":
        seed["grounded_evidence"] = evidence_offsets(users[0]["content"], source, choice)
    return seed, options


def locator_key(source):
    repo = source.get("repo")
    if repo == "opencsg/smoltalk-chinese" and source.get("source_row_1_based") is not None:
        return repo, str(source["source_row_1_based"] - 1)
    for locator in ("source_row_id", "cache_ordinal", "source_byte_offset", "ordinal_1_based", "old_train_id"):
        value = source.get(locator)
        if value is not None:
            return repo, str(value)
    raise ValueError(f"Missing source locator for {repo}")


def prior_review_locations():
    files = (
        ("data_pilot", "samples"), ("data_round2_pilot", "samples"),
        ("quality_pilot_review", "reviews"), ("quality_round2_review", "reviews"),
        ("semantic_pilot_review", "reviews"), ("semantic_round2_review", "reviews"),
        ("local_capacity_probe", "records"), ("local_source_pilot", "records"),
        ("smoltalk_source_probe", "records"),
    )
    seen = defaultdict(set)
    for name, key in files:
        path = REPORT / f"{name}.json"
        records = json.loads(path.read_text(encoding="utf-8-sig"))[key]
        for record in records:
            if name == "smoltalk_source_probe":
                identity = "opencsg/smoltalk-chinese", str(record["source_row_index_0_based"])
            elif source := record.get("source_record"):
                identity = locator_key(source)
            elif record.get("source_row_id") is not None:
                identity = record["source"], str(record["source_row_id"])
            else:
                continue
            seen[identity].add(relative(path))
    return seen, [f"reports/natural_sft_v4/{name}.json" for name, _ in files]


def prompt_terms(text):
    normalized = normalized_content(text)
    return normalized, set(zip(normalized, normalized[1:], normalized[2:]))


def load_prompt_bank():
    bank = {}
    stats = Counter()
    for path in PROMPT_BANKS:
        for row in json_rows(path):
            for item in row["messages"]:
                if item["role"] != "user":
                    continue
                text = item["content"]
                key, terms = prompt_terms(text)
                if key:
                    bank.setdefault(key, (row["id"], text, terms))
                    stats[relative(path)] += 1
    return bank, dict(stats)


def near_match(left, right):
    return len(left & right) / len(left | right) if left and right else 0.0


def prompt_checks(seeds):
    bank, stats = load_prompt_bank()
    firsts = {}
    known_users = defaultdict(set)
    review = []
    for seed in seeds:
        for index, user in enumerate(seed["user_messages"]):
            content = user["content"]
            components = [content]
            if content.startswith(CMRC_PREFIX):
                components.append(content.rsplit(CMRC_SEPARATOR, 1)[1])
            for component in components:
                key, terms = prompt_terms(component)
                if not key:
                    raise ValueError(f"Blank normalized user prompt in {seed['id']}")
                if key in bank:
                    raise ValueError(f"Frozen prompt exact overlap: {seed['id']} and {bank[key][0]}")
                known_users[key].add(seed["id"])
                max_score, nearest = max((near_match(terms, entry[2]), entry[0]) for entry in bank.values())
                if max_score >= 0.22:
                    review.append({"seed": seed["id"], "user_turn": index + 1, "other": nearest, "score": round(max_score, 3)})
            if index == 0:
                firsts[seed["id"]] = prompt_terms(components[-1])
    duplicate_users = {key: sorted(ids) for key, ids in known_users.items() if len(ids) > 1}
    if duplicate_users:
        raise ValueError(f"Duplicate normalized user messages between seeds: {duplicate_users}")
    for (left, (_, terms_left)), (right, (_, terms_right)) in combinations(firsts.items(), 2):
        score = near_match(terms_left, terms_right)
        if score >= 0.22:
            review.append({"seed": left, "user_turn": 1, "other": right, "score": round(score, 3)})
    return {
        "inputs_only_user_messages_read": stats, "unique_normalized_frozen_user_inputs": len(bank),
        "normalized_exact_frozen_matches": 0, "normalized_exact_cross_seed_user_matches": 0,
        "tri_character_jaccard_review_queue": sorted(review, key=lambda hit: -hit["score"]),
    }


def verify_local_sources(seeds, options_by_id, tokenizer):
    files = defaultdict(list)
    for seed in seeds:
        pool_view = {"id": seed["id"], "group_id": seed["source_group"],
                     "messages": seed["original_messages"], "prefix_options": options_by_id[seed["id"]]}
        for source in seed["sources"]:
            files[ROOT / source["local_file"]].append((pool_view, source))
    verified = Counter()
    for path, selected in files.items():
        if path.suffix == ".sqlite3":
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
                for pool_view, source in selected:
                    payload, = database.execute(
                        "SELECT payload FROM candidates WHERE repo=? AND revision=? AND ordinal=?",
                        (source["repo"], source["revision"], source["cache_ordinal"]),
                    ).fetchone()
                    verify_example(pool_view, source, json.loads(payload), tokenizer)
                    verified[relative(path)] += 1
            continue
        is_parquet = path.suffix == ".parquet"
        positions = defaultdict(list)
        for pool_view, source in selected:
            locator = "source_row_1_based" if is_parquet else "jsonl_line_1_based"
            positions[source[locator]].append((pool_view, source))
        rows = parquet_rows(path) if is_parquet else json_rows(path)
        for number, original in enumerate(islice(rows, max(positions)), 1):
            for pool_view, source in positions.pop(number, ()):
                verify_example(pool_view, source, original, tokenizer)
                verified[relative(path)] += 1
        if positions:
            raise ValueError(f"Could not relocate selected source records in {path}: {sorted(positions)}")
    return dict(verified)


def build():
    choices = list(batch_choices())
    pool_rows = selected_pool_rows(choices)
    seeds, options_by_id = [], {}
    for index, (batch, category, choice) in enumerate(choices, 1):
        seed, options = make_seed(index, batch, category, choice, pool_rows[choice.prefix])
        seeds.append(seed)
        options_by_id[seed["id"]] = options
    if len({seed["source_group"] for seed in seeds}) != 64:
        raise ValueError("Pilot seeds contain duplicate candidate groups")
    if Counter(seed["category"] for seed in seeds) != Counter(COUNTS):
        raise ValueError("Pilot seeds differ from category quotas")
    cmrc_count = sum(seed.get("grounded_evidence", {}).get("source_type") == "cmrc_complete_context" for seed in seeds)
    multi_counts = Counter(seed["multi_subtype"] for seed in seeds if seed["category"] == "multi")
    if cmrc_count > 8 or multi_counts["document"] > 8 or multi_counts["daily"] < 8:
        raise ValueError("Pilot source or multi-turn strata exceed their limits")
    seen, prior_paths = prior_review_locations()
    overlaps = {}
    for seed in seeds:
        paths = set().union(*(seen.get(locator_key(source), ()) for source in seed["sources"]))
        if paths:
            overlaps[seed["id"]] = sorted(paths)
    regressions = {seed["id"]: seed["regression_report"] for seed in seeds if seed["regression_seed"]}
    if len(regressions) > 8:
        raise ValueError("Pilot exceeds the repair-control quota")
    stats = {
        "status": "seed_questions_only", "seeds": len(seeds), "categories": COUNTS,
        "batches": {batch: dict(Counter(seed["category"] for seed in seeds if seed["pilot_batch"] == batch))
                    for batch in range(1, 9)},
        "unique_source_groups": 64, "primary_inputs": dict(Counter(seed["sources"][0]["input"] for seed in seeds)),
        "grounded_cmrc": cmrc_count, "multi_subtypes": dict(multi_counts),
        "repair_controls": regressions, "new_seed_count": len(seeds) - len(regressions),
        "prior_review_inputs": prior_paths, "prior_review_source_locator_overlaps": overlaps,
        "unexpected_prior_review_overlaps": {k: v for k, v in overlaps.items() if k not in regressions},
        "largest_fixed_prompt": max(seed["tokens_without_assistant_content"] for seed in seeds),
        "smallest_total_assistant_content_budget": min(seed["assistant_content_token_budget"] for seed in seeds),
        "largest_original_prefix": max(seed["original_prefix_tokens"] for seed in seeds),
        "prompt_checks": prompt_checks(seeds),
        "chat_format": {"max_context": MAX_TOKENS, "final_eos": False, "tokenizer": relative(TOKENIZER)},
    }
    return seeds, options_by_id, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Print selection checks without writing files")
    args = parser.parse_args()
    if not args.check_only:
        for path in (DEST, SEED_REPORT):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
    seeds, options_by_id, stats = build()
    if args.check_only:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    if stats["unexpected_prior_review_overlaps"]:
        raise ValueError(f"Non-repair examples overlap a prior pilot: {stats['unexpected_prior_review_overlaps']}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_special_ids(tokenizer)
    stats["local_source_locators_verified"] = verify_local_sources(seeds, options_by_id, tokenizer)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    SEED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("x", encoding="utf-8", newline="\n") as output:
        for seed in seeds:
            output.write(json.dumps(seed, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(SEED_REPORT, stats)
    print(f"Verified and wrote {len(seeds)} seeds to {DEST}")


if __name__ == "__main__":
    main()
