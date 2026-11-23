"""Prepare task-verifiable Chinese SFT data for the paired comparison."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import cycle, islice
import json
from pathlib import Path
from random import Random

from tokenizers import Tokenizer

from .chat_format import IGNORE_LABEL, SPECIAL_IDS, check_special_ids, encode_conversation
from .comparison_sft_data import ConflictIndex, external_index, prompts, pt_chunks, pt_index
from .stage1_data import SPLITS, build_lock, file_identity, heartbeat, json_line, write_json
from .verified_sft_checks import (
    ENDS, PATCH_LINE, RESOURCE_LINE, RULES, STARTS, UPDATES, initial_record, prompt_identity, verify_encoded_row,
)


CORPUS_ID = "sft_verified_v2"
SOURCE = "local_verifiable_scenarios_v2"
SEED = 20261122
SAMPLE_SEED = 352679
MAX_TOKENS = 512
PARTITIONS = ("validation", "test", "train")
RESULT_COUNTS = {"train": 20_000, "validation": 500, "test": 500}
MULTI_COUNTS = {"train": 3000, "validation": 75, "test": 75}
ENTITY_PARTS = {
    "train": ("棠槿青清柏杉椿兰松棣梧桐桃杨银茶梓榆", "舟川江绣桥霞芒盏垒坪坞溪坪亭楼榭湾谷"),
    "validation": ("绛珀酥柚玄缃鹤鸿杉橙芝莓络铃桃白杏柳", "原岭麓岛田浦岩坡潭涧门闸廊坊庭梁坎廓"),
    "test": ("墨雪野泠霁寒静汐荧桦莲朔秋晴霜木兰竹", "枝芽蕊果叶环伞灯卷笺杯台盘匣炉囊箱筐"),
}
INTEGER_BOUNDS = {"train": (11, 419), "validation": (421, 529), "test": (531, 659)}
YEARS = {"train": (1991, 2055), "validation": (2081, 2119), "test": (2181, 2219)}
BOOL_RULES = ("三项都满足", "至少一项满足", "前两项中恰一项满足且第三项满足", "第一项满足且后两项不都满足")
EXTRACTION_FIELDS = ("归档点", "条目名称", "展签用色", "代号", "签收人", "保存处", "所用纸张", "活动称呼")
CLARIFICATION_SLOTS = (
    ("预约排练室", ("场地", "日期", "时段")), ("登记寄存", ("寄存人", "柜号", "件数")),
    ("记下领用需求", ("领用人", "物品", "数量")), ("填写导览安排", ("参加人", "集合处", "出发时段")),
)
SUPPORT_FIELDS = ("封面", "编号", "类别", "地点", "标签", "尺寸", "说明")
COLORS = ("浅灰", "松石绿", "米白", "深蓝", "赭红", "乳白", "靛蓝", "草绿")
PAPERS = ("铜版纸", "牛皮纸", "卡纸", "再生纸", "白板纸", "宣纸")
TIMES = ("09:00-10:00", "14:00-15:00", "15:30-16:30", "18:00-19:00")


def rank(text):
    return sha256(text.encode("utf-8")).hexdigest()


def compact(value, *, sort_keys=False):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)


def fields(payload):
    return [f"{name}：{compact(value)}" for name, value in payload.items()]


def unique_names(split, rng, count):
    heads, tails = ENTITY_PARTS[split]
    names = sorted({head + tail for head in heads for tail in tails})
    return rng.sample(names, count)


def slot_value(slot, name, split, rng):
    if slot == "日期":
        return f"{rng.randint(*YEARS[split]):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if slot in {"时段", "出发时段"}:
        return rng.choice(TIMES)
    if slot in {"件数", "数量"}:
        return rng.randint(1, 12)
    if slot == "物品":
        return rng.choice(("夹板", "记号笔", "收纳盒", "文件袋", "便签纸"))
    if slot == "柜号":
        return f"{name}柜{rng.randint(1, 18)}号"
    if slot == "场地":
        return name + "排练室"
    if slot == "集合处":
        return name + "入口"
    return name


def extraction_value(field, name, rng):
    if field == "展签用色":
        return rng.choice(COLORS)
    if field == "所用纸张":
        return rng.choice(PAPERS)
    if field in {"归档点", "保存处"}:
        return name + "资料柜"
    if field == "条目名称":
        return name + "展览记录"
    if field == "活动称呼":
        return name + "手工体验日"
    return name


def support_value(field, name, rng):
    if field == "尺寸":
        return f"{rng.randint(6, 195)}厘米"
    if field == "封面":
        return rng.choice(COLORS)
    if field == "地点":
        return name + "阅览室"
    if field == "说明":
        return rng.choice(("已装订", "待复核", "午后登记", "归还后登记"))
    if field == "类别":
        return rng.choice(("活动记录", "采购清单", "现场笔记", "往来信件"))
    return name


def fixture(family, split, rng):
    a, b, c, d, e, f, g, h, i, j = names = unique_names(split, rng, 10)
    match family:
        case "arithmetic":
            lower, upper = INTEGER_BOUNDS[split]
            left = rng.randint(lower, upper)
            right = rng.randint(2, 39)
            operation = rng.choice(("+", "−", "×", "÷"))
            if operation == "÷":
                left *= right
                update = {"运算": rng.choice(("+", "−", "×"))}
            else:
                update = {"右数": right + rng.randint(1, 13)}
            return {"左数": left, "右数": right, "运算": operation}, update

        case "extraction":
            keys = rng.sample(EXTRACTION_FIELDS, 4)
            text = "；".join(f"{key}：{extraction_value(key, name, rng)}"
                            for key, name in zip(keys, names[:4], strict=True)) + "。"
            return {"原文": text, "字段": keys[0]}, {"字段": keys[rng.randrange(1, len(keys))]}

        case "format":
            y, m, day = rng.randint(*YEARS[split]), rng.randint(1, 12), rng.randint(1, 28)
            mode = rng.choice(("短表", "ISO"))
            other = "ISO" if mode == "短表" else "短表"
            return {"年": y, "月": m, "日": day, "格式": mode}, rng.choice(({"格式": other}, {"日": day % 28 + 1}))

        case "clarification":
            task, slots = rng.choice(CLARIFICATION_SLOTS)
            values = [slot_value(slot, names[index + 1], split, rng) for index, slot in enumerate(slots)]
            missing = rng.sample(range(3), rng.randrange(1, 3))
            filled = {slot: "" if idx in missing else values[idx] for idx, slot in enumerate(slots)}
            correction = {slots[missing[0]]: values[missing[0]]}
            return {"事项": f"{a}小组{task}", "必填": list(slots), "填写": filled}, {"填写": correction}

        case "correction":
            key = rng.choice(("存放处", "负责人", "记录代号"))
            record = {"存放处": b + "室", "负责人": d, "记录代号": f}
            corrected = h + "室" if key == "存放处" else h
            replacement = i + "室" if key == "存放处" else i
            return {"记录": record, "已更正": {key: corrected}, "字段": key}, {"改写": {key: replacement}}

        case "filter_select":
            rows = [[name, rng.choice((a, b)), rng.randint(35, 99)] for name in names[2:8]]
            threshold = rng.randrange(35, 91)
            return {"表": rows, "组": a, "最低分": threshold}, rng.choice(({"组": b}, {"最低分": rng.randint(80, 110)}))

        case "count":
            low = rng.randint(2, 20)
            high = low + rng.randint(7, 27)
            numbers = [rng.randint(1, 61) for _ in range(13)]
            mode = rng.choice(("奇", "偶"))
            return {"数列": numbers, "下限": low, "上限": high, "奇偶": mode}, {"奇偶": "不限"}

        case "dedup":
            original = rng.choices(names[:5], k=8)
            added = rng.sample(names[5:], 3)
            return {"清单": original}, {"追加": [rng.choice(original), *added]}

        case "sort":
            rows = [[name, rng.randint(1, 4), rng.randrange(10, 80, 5)] for name in names[:5]]
            return {"行": rows}, {"追加": [[f, rng.randint(1, 4), rng.randrange(10, 80, 5)]]}

        case "set":
            floor = {"train": 7, "validation": 85, "test": 155}[split] + rng.randint(0, 19)
            left = rng.choices(range(floor, floor + 24), k=8)
            right = rng.choices(range(floor, floor + 24), k=7)
            modes = rng.sample(("并", "交", "差"), 2)
            return {"左": left, "右": right, "运算": modes[0]}, {"运算": modes[1]}

        case "classify":
            base = rng.randint(1, 25)
            bounds = [base, base + rng.randint(3, 14), base + rng.randint(15, 28), base + rng.randint(29, 52)]
            intervals = [[bounds[idx], bounds[idx + 1], name] for idx, name in enumerate((a, b, c))]
            first, second = rng.sample(range(3), 2)
            return {"测量值": rng.randrange(*bounds[first:first + 2]), "区间": intervals}, {"测量值": rng.randrange(*bounds[second:second + 2])}

        case "lookup":
            rows, columns = [a, b, c], [d, e, f]
            values = rng.sample(range(-70, 100), 9)
            table = [values[offset:offset + 3] for offset in (0, 3, 6)]
            row = rng.choice(rows)
            col = rng.choice(columns)
            return {"行名": rows, "列名": columns, "表": table, "行": row, "列": col}, {"列": rng.choice([x for x in columns if x != col])}

        case "map":
            mapping = dict(zip(names[:4], names[4:8], strict=True))
            units = rng.choices([a, b, c, d, i, j], k=6)
            return {"词序列": units, "替换": mapping}, {"追加": rng.sample([a, b, c, d, i, j], 2)}

        case "string":
            alphabet = {"train": "AbCdEfGhIjKmMnOpQrsTvYz", "validation": "BjDFHLMnpQRTvbWX", "test": "aCDgIKNQStUYz"}[split]
            text = "".join(rng.choices(alphabet + "172985-._#", k=rng.randint(9, 17)))
            mode = rng.choice(("大写", "小写"))
            delete_chars = rng.choice(("-._", "#1", "-#", "_7"))
            return {"字符串": text, "删除字符": delete_chars, "大小写": mode}, {"大小写": "小写" if mode == "大写" else "大写"}

        case "boolean":
            base = {"train": 4, "validation": 48, "test": 93}[split] + rng.randint(0, 14)
            inputs = [rng.randint(base, base + 40) for _ in range(3)]
            limits = [rng.randint(base, base + 40) for _ in range(3)]
            condition, new_condition = rng.sample(BOOL_RULES, 2)
            return {"给定值": inputs, "下限": limits, "条件": condition}, {"条件": new_condition}

        case "sequence":
            pattern = names[:rng.randrange(2, 6)]
            count = rng.randint(1, 5)
            return {"周期": pattern, "已出现数": rng.randint(1, 24), "接写数": count}, {"接写数": count + rng.randrange(1, 4)}

        case "reorder":
            items = names[:rng.randint(4, 8)]
            steps = rng.randint(1, len(items) - 1)
            operation = rng.choice(("逆序", "左移"))
            return {"清单": items, "操作": operation, "步数": steps}, {"操作": "左移" if operation == "逆序" else "逆序"}

        case "reference":
            sentences = [f"{a}：我先提到{c}展签，随后提到{d}图册。", f"{b}：我先提到{e}木盒，随后提到{f}色卡。"]
            item = rng.choice(("前者", "后者"))
            return {"原话": sentences, "说话人": a, "指代": item}, rng.choice(({"说话人": b}, {"指代": "后者" if item == "前者" else "前者"}))

        case "constraints":
            table = [[name, rng.choice((a, b)), rng.randint(1, 100), rng.randint(1, 20)] for name in names[2:9]]
            floor, ceiling = rng.randint(10, 45), rng.randint(10, 20)
            return {"表": table, "组": a, "最低分": floor, "最多件数": ceiling}, rng.choice((
                {"最低分": rng.randint(51, 95)}, {"最多件数": rng.randint(2, 9)},
            ))

        case "support":
            field_a, field_b, field_c = rng.sample(SUPPORT_FIELDS, 3)
            value_a, value_b, value_c = (support_value(field, name, rng)
                                         for field, name in ((field_a, d), (field_b, f), (field_c, h)))
            other = support_value(field_a, j, rng)
            if other == value_a:
                other = support_value(field_a, i, rng)
            records = [(a, field_a, value_a), (a, field_b, value_b), (b, field_c, value_c)]
            corpus = [f"{subject}的{field}：{value}。" for subject, field, value in records]
            statements = rng.sample([[a, field_a, value_a], [a, field_a, other], [c, field_a, other], [b, field_c, value_c]], 2)
            return {"原文": corpus, "陈述": statements[0]}, {"陈述": statements[1]}
    raise ValueError(f"Unknown generation family: {family}")


def generated_answer(family, data):
    """Produce a label. The independent verifier never imports this module."""
    match family:
        case "arithmetic":
            x, y = data["左数"], data["右数"]
            result = {"+": x + y, "−": x - y, "×": x * y, "÷": x // y}[data["运算"]]
        case "extraction":
            entries = (part.removesuffix("。").split("：", 1) for part in data["原文"].split("；"))
            return dict(entries)[data["字段"]]
        case "format":
            y, m, day = data["年"], data["月"], data["日"]
            return f"{y:04d}-{m:02d}-{day:02d}" if data["格式"] == "ISO" else compact([y, m, day])
        case "clarification":
            filled = data["填写"]
            omitted = "、".join(slot for slot in data["必填"] if not filled[slot])
            return f"请补充{omitted}。" if omitted else "资料已齐。"
        case "correction":
            record = data["记录"] | data["已更正"]
            return record[data["字段"]]
        case "filter_select":
            filtered = sorted((x for x in data["表"] if x[1] == data["组"] and x[2] >= data["最低分"]),
                              key=lambda x: x[2], reverse=True)
            return filtered[0][0] if filtered else "无"
        case "count":
            progression = range(data["下限"], data["上限"] + 1)
            parity = {"奇": {1}, "偶": {0}, "不限": {0, 1}}[data["奇偶"]]
            distribution = Counter(data["数列"])
            result = sum(distribution[n] for n in progression if n % 2 in parity)
        case "dedup":
            result = list(dict.fromkeys(data["清单"]))
        case "sort":
            descending = sorted(data["行"], key=lambda x: x[2], reverse=True)
            result = [item[0] for item in sorted(descending, key=lambda x: x[1])]
        case "set":
            left, right = set(data["左"]), set(data["右"])
            result = sorted({"并": left.union, "交": left.intersection, "差": left.difference}[data["运算"]](right))
        case "classify":
            measurement = data["测量值"]
            return next(label for low, _, label in reversed(data["区间"]) if measurement >= low)
        case "lookup":
            addressed = {(row_name, col_name): number
                         for row_name, row in zip(data["行名"], data["表"], strict=True)
                         for col_name, number in zip(data["列名"], row, strict=True)}
            result = addressed[data["行"], data["列"]]
        case "map":
            result = list(map(lambda word: data["替换"].get(word, word), data["词序列"]))
        case "string":
            cleaned = data["字符串"].translate(str.maketrans("", "", data["删除字符"]))
            return cleaned.upper() if data["大小写"] == "大写" else cleaned.lower()
        case "boolean":
            mask = sum(1 << position for position, value in enumerate(data["给定值"]) if value >= data["下限"][position])
            successful = ({7}, set(range(1, 8)), {5, 6}, {1, 3, 5})[BOOL_RULES.index(data["条件"])]
            return "是" if mask in successful else "否"
        case "sequence":
            source = cycle(data["周期"])
            result = list(islice(source, data["已出现数"], data["已出现数"] + data["接写数"]))
        case "reorder":
            original, steps = data["清单"], data["步数"]
            result = original[::-1] if data["操作"] == "逆序" else original[steps:] + original[:steps]
        case "reference":
            speaker = data["说话人"] + "：我先提到"
            phrase = next(s.removeprefix(speaker).removesuffix("。") for s in data["原话"] if s.startswith(speaker))
            first, second = phrase.split("，随后提到", 1)
            return first if data["指代"] == "前者" else second
        case "constraints":
            groups = {row[0] for row in data["表"] if row[1] == data["组"]}
            passing = {row[0] for row in data["表"] if row[2] >= data["最低分"]}
            quantities = {row[0] for row in data["表"] if row[3] <= data["最多件数"]}
            result = [row[0] for row in data["表"] if row[0] in groups & passing & quantities]
        case "support":
            subject, field, value = data["陈述"]
            sentence = f"{subject}的{field}：{value}。"
            if sentence in data["原文"]:
                return "支持"
            return "矛盾" if any(x.startswith(f"{subject}的{field}：") for x in data["原文"]) else "未提及"
        case _:
            raise ValueError(f"Unknown generation family: {family}")
    return compact(result) if isinstance(result, list) else str(result)


def edited_fixture(family, payload, change):
    result = deepcopy(payload)
    if family == "correction" and "改写" in change:
        result["已更正"].update(change["改写"])
    elif family == "clarification":
        result["填写"].update(change["填写"])
    elif "追加" in change:
        key = {"dedup": "清单", "map": "词序列", "sort": "行"}[family]
        result[key].extend(change["追加"])
    else:
        result.update(change)
    return result


def render_row(family, split, style, multi, ordinal, seed=SEED):
    scenario_seed = rank(f"{seed}:{split}:{family}:{style}:{int(multi)}:{ordinal}")
    rng = Random(int(scenario_seed[:16], 16))
    payload, change = fixture(family, split, rng)
    style_a, style_b = divmod(style, len(ENDS[split]))
    question = "\n".join((STARTS[split][style_a], RULES[family], RESOURCE_LINE, *fields(payload), ENDS[split][style_b]))
    first = generated_answer(family, payload)
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": first}]
    if multi:
        after = generated_answer(family, edited_fixture(family, payload, change))
        if after == first:
            return None
        lead = UPDATES[split][rng.randrange(len(UPDATES[split]))]
        if family == "constraints":
            lead += "原有的限制仍有效。"
        follow = "\n".join((lead, PATCH_LINE, *fields(change)))
        messages += [{"role": "user", "content": follow}, {"role": "assistant", "content": after}]
    scene_id = rank(compact([family, payload], sort_keys=True))
    identifier = prompt_identity(messages)
    row = {"id": identifier, "source": SOURCE, "split": split, "task_family": family,
           "template_family": family, "template_id": f"{split}:{family}:{style:02d}", "style": style,
           "scenario_seed": scenario_seed, "scene_id": scene_id, "prompt_identity": identifier,
           "messages": messages}
    return row


def plan_slots(split, family_index, probe):
    if probe:
        if split == "train":
            return [(family_index, 42, 8)]
        return [(family_index, 1, 0)] if family_index < 12 else []
    if split == "train":
        return [(style, 50 - (8 if style % 2 == 0 else 7), 8 if style % 2 == 0 else 7) for style in range(20)]
    double = {(family_index * 2 + offset) % 20 for offset in range(5)}
    multi = {(family_index * 3 + offset) % 20 for offset in range(4 if family_index < 15 else 3)}
    return [(style, 1 + (style in double) - (style in multi), int(style in multi)) for style in range(20)]


def prompt_keys(messages):
    pieces = prompts(messages)
    return (pieces[0], "|".join(pieces)) if len(pieces) > 1 else pieces


class Gate:
    def __init__(self, tokenizer, evaluation, old_sft, old_pt):
        self.tokenizer = tokenizer
        self.evaluation = evaluation
        self.old_sft = old_sft
        self.old_pt = old_pt
        self.heldout_indexes = {split: ConflictIndex() for split in ("validation", "test")}
        self.ids = set()
        self.seeds = set()
        self.scenes = set()
        self.local_keys = set()
        self.rejections = Counter()

    def screening(self, row):
        split = row["split"]
        messages = row["messages"]
        keys = prompt_keys(messages)
        if row["id"] in self.ids or row["scene_id"] in self.scenes or any(key in self.local_keys for key in keys):
            return "duplicate_initial_scene_or_prompt"
        for key in prompts(messages):
            if reason := self.evaluation.match(key):
                return f"evaluation_{reason}"
            if reason := self.old_sft.match(key):
                return f"old_sft_{reason}"
        for message in messages:
            for key in pt_chunks(message["content"]):
                if reason := self.old_pt.match(key):
                    return f"old_pretrain_{reason}"
        for peer, index in self.heldout_indexes.items():
            if peer != split:
                for key in keys:
                    if reason := index.match(key):
                        return f"cross_split_{reason}"
        return None

    def accept(self, row):
        if row is None:
            self.rejections["followup_unchanged_result"] += 1
            return False
        if reason := self.screening(row):
            self.rejections[reason] += 1
            return False
        encoded = encode_conversation(self.tokenizer, row["messages"])
        if len(encoded["input_ids"]) > MAX_TOKENS:
            self.rejections["would_exceed_context"] += 1
            return False
        row.update(encoded)
        verify_encoded_row(row, self.tokenizer, MAX_TOKENS)
        family, _, context = initial_record(row["messages"][0]["content"], row["split"])
        if row["scene_id"] != rank(compact([family, context], sort_keys=True)):
            raise ValueError("Visible scenario differs from its provenance")
        if row["scenario_seed"] in self.seeds:
            raise ValueError("Duplicate scenario seed")
        self.ids.add(row["id"])
        self.seeds.add(row["scenario_seed"])
        self.scenes.add(row["scene_id"])
        keys = prompt_keys(row["messages"])
        self.local_keys.update(keys)
        if row["split"] in self.heldout_indexes:
            for key in keys:
                self.heldout_indexes[row["split"]].add(key)
        return True


def collect(gate, probe, progress):
    selected = {split: [] for split in SPLITS}
    for split in PARTITIONS:
        for family_index, family in enumerate(RULES):
            start = len(selected[split])
            for style, single_count, multi_count in plan_slots(split, family_index, probe):
                for multi, count in ((False, single_count), (True, multi_count)):
                    obtained = 0
                    ordinal = 0
                    while obtained < count and ordinal < 3000:
                        attempt = render_row(family, split, style, multi, ordinal)
                        ordinal += 1
                        if gate.accept(attempt):
                            selected[split].append(attempt)
                            obtained += 1
                    if obtained != count:
                        raise RuntimeError(f"Exhausted {split}/{family}/{style} multi={multi}; selected {obtained}/{count} "
                                           f"with rejects {dict(gate.rejections)}")
            progress(f"{split} {family}: {len(selected[split]) - start} selected, {len(selected[split])} split total")
    return selected


def summarize(rows):
    answer_turns = sum(len(row["messages"]) // 2 for row in rows)
    return {
        "documents": len(rows),
        "tokens": sum(len(row["input_ids"]) for row in rows),
        "assistant_tokens": sum(label != IGNORE_LABEL for row in rows for label in row["labels"]),
        "assistant_turns_verified": answer_turns,
        "multi": sum(len(row["messages"]) > 2 for row in rows),
        "max_context": max((len(row["input_ids"]) for row in rows), default=0),
        "min_context": min((len(row["input_ids"]) for row in rows), default=0),
        "by_family": dict(Counter(row["task_family"] for row in rows)),
        "multi_by_family": dict(Counter(row["task_family"] for row in rows if len(row["messages"]) > 2)),
        "by_template": dict(Counter(row["template_id"] for row in rows)),
    }


def track_entity_partitions():
    inventories = {}
    for split, (heads, tails) in ENTITY_PARTS.items():
        inventories[split] = {head + tail for head in heads for tail in tails}
    for left in SPLITS:
        for right in SPLITS:
            if left < right and inventories[left] & inventories[right]:
                raise ValueError(f"Named-entity inventories overlap: {left} and {right}")
    return {split: sorted(values) for split, values in inventories.items()}


def audit_quotas(stats):
    for split, row in stats.items():
        if row["documents"] != RESULT_COUNTS[split] or row["multi"] != MULTI_COUNTS[split]:
            raise ValueError(f"Quota mismatch for {split}")
        expected_family_count = RESULT_COUNTS[split] // len(RULES)
        for family_index, family in enumerate(RULES):
            slots = plan_slots(split, family_index, probe=False)
            if (len(slots) != 20 or row["by_family"].get(family) != expected_family_count
                    or row["multi_by_family"].get(family) != sum(multi for _, _, multi in slots)):
                raise ValueError(f"Family distribution mismatch for {split}/{family}")
            for style, single, multi in slots:
                if not 1 <= single + multi <= 50 or row["by_template"].get(f"{split}:{family}:{style:02d}") != single + multi:
                    raise ValueError(f"Template quota mismatch for {split}/{family}/{style}")


def review_samples(rows):
    inventory = []
    for family in RULES:
        for multi, count in ((False, 3), (True, 2)):
            pool = (row for row in rows if row["task_family"] == family and (len(row["messages"]) > 2) == multi)
            candidates = sorted(pool, key=lambda row: rank(f"{SAMPLE_SEED}:{row['id']}"))
            if len(candidates) < count:
                raise ValueError(f"Insufficient samples for {family}")
            inventory.extend({"id": row["id"], "family": family, "messages": row["messages"]} for row in candidates[:count])
    return sorted(inventory, key=lambda row: rank(f"review-order:{SAMPLE_SEED}:{row['id']}"))


def load_frozen_tokenizer(base):
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    path = Path(manifest["tokenizer_path"])
    if (manifest["schema_version"] != 1 or manifest["corpus_id"] != "stage1_v1" or manifest["frozen"] is not True
            or not path.is_absolute() or path.resolve() != (base / "tokenizer.json").resolve()
            or manifest["vocab_size"] != 16384 or manifest["special_token_ids"] != SPECIAL_IDS
            or manifest["files"]["tokenizer.json"] != file_identity(path)
            or manifest["tokenizer_sha256"] != sha256(path.read_bytes()).hexdigest()):
        raise ValueError("Stage 1 tokenizer identity does not match")
    tokenizer = Tokenizer.from_file(str(path))
    check_special_ids(tokenizer)
    if tokenizer.get_vocab_size() != manifest["vocab_size"]:
        raise ValueError("Stage 1 tokenizer vocabulary size differs")
    return tokenizer, manifest


def prepare(args):
    tokenizer, base = load_frozen_tokenizer(args.base)
    args.reports.mkdir(parents=True, exist_ok=True)
    inventories = track_entity_partitions()
    with build_lock(args.output), heartbeat("indexing machine-only external holdout keys") as progress:
        if not args.probe and (args.output / "manifest.prepared.json").exists():
            raise FileExistsError("A prepared corpus already exists")
        evaluation, old_sft, identities = external_index(args.old_sft, args.eval_prompts)
        old_pt, pt_identities = pt_index(args.base)
        gate = Gate(tokenizer, evaluation, old_sft, old_pt)
        selected = collect(gate, args.probe, progress)
        stats = {split: summarize(rows) for split, rows in selected.items()}
        report = {"status": "probe_only" if args.probe else "prepared_for_independent_review", "splits": stats,
                  "seed": SEED, "quality_sample_seed": SAMPLE_SEED, "rejected_attempts": dict(gate.rejections),
                  "verified_dialogues": sum(len(rows) for rows in selected.values()),
                  "verified_assistant_turns": sum(row["assistant_turns_verified"] for row in stats.values()),
                  "screened_inputs": {"evaluation_and_old_sft": identities, "old_pretraining": pt_identities},
                  "partitioned_named_entity_inventory": inventories,
                  "scene_identity": "SHA-256 of task family and normalized visible initial fields, independent of presentation",
                  "aliasing": "The 20 task procedures are shared by design. Surface introductions/closings, follow-up frames, "
                              "named-entity inventories, scenario seeds and initial parameter combinations are separated. Shared "
                              "operators, prompt instructions and short common words remain deliberately similar.",
                  "training_update_budget": {"batch_size": 8, "gradient_accumulation": 8, "epochs": 1, "updates": 313},
                  "near_duplicate_scope": "Across partitions, normalized first prompt and full user-turn sequence screened "
                                          "with five-character set Jaccard >=0.8 and exact matching; later contextual patches are "
                                          "checked with their history. Repeated answer words and abstract task structures are expected.",
                  "external_screening": "All individual user turns compared to full old and new evaluation prompts for exact, "
                                        "normalized symmetric containment from 6 characters and near overlap, and to old SFT "
                                        "holdout user prompts. Both roles checked against normalized old pretrain holdout "
                                        "document fragments at least 40 characters.",
                  "checker": "Reparsed the user-visible initial fields and changes, independently solved every assistant turn, "
                             "re-encoded source messages and walked all labels, roles and end markers. No answer cache was consulted.",
                  "limitations": "Fictional task families share deterministic instruction and data grammar; surface template counts "
                                 "do not establish natural-dialogue diversity. Collision checks do not exclude all paraphrases."}
        if args.probe:
            write_json(args.reports / "verified_sft_probe.json", report)
            progress(f"probe done: {report['verified_dialogues']:,} dialogues checked")
            return
        audit_quotas(stats)
        files = {"tokenizer.json": base["files"]["tokenizer.json"]}
        for split, rows in selected.items():
            name = f"{split}.jsonl"
            path = args.output / name
            pending = args.output / f"{name}.partial"
            with pending.open("w", encoding="utf-8", newline="\n") as stream:
                for row in sorted(rows, key=lambda item: rank(f"{SEED}:output:{split}:{item['id']}")):
                    stream.write(json_line(row))
            pending.replace(path)
            files[name] = file_identity(path)
        manifest = {
            "schema_version": 1, "corpus_id": CORPUS_ID, "frozen": True, "vocab_size": base["vocab_size"],
            "seed": SEED, "max_tokens": MAX_TOKENS, "tokenizer_path": base["tokenizer_path"],
            "tokenizer_sha256": base["tokenizer_sha256"], "special_token_ids": SPECIAL_IDS,
            "counts": RESULT_COUNTS, "splits": stats, "files": files, "screened_inputs": report["screened_inputs"],
            "sources": [{"source": SOURCE, "generator": "tinychat/comparison_verified_sft.py",
                         "verifier": "tinychat/verified_sft_checks.py"}],
            "supervision": "Assistant content, assistant end markers and final EOS per frozen Stage 1 format",
            "quota": {"distinct_semantic_families": 20, "train_per_family": 1000, "multi": MULTI_COUNTS},
            "split_policy": report["aliasing"],
        }
        write_json(args.reports / "verified_sft_quality.json", report)
        write_json(args.reports / "verified_sft_samples.json", review_samples(selected["train"]))
        write_json(args.output / "manifest.prepared.json", manifest)
        progress(f"prepared {manifest['counts']} for independent review")


def publish_prepared(output, base, review_marker):
    """Publish the exact prepared manifest after the matching review passes."""
    from .training_state import load_json, read_frozen_manifest

    with build_lock(output):
        prepared_path = output / "manifest.prepared.json"
        prepared_bytes = prepared_path.read_bytes()
        review = load_json(review_marker.read_text(encoding="utf-8-sig"))
        if (not isinstance(review, dict) or review.get("decision") != "pass" or review.get("corpus_id") != CORPUS_ID
                or review.get("prepared_sha256") != sha256(prepared_bytes).hexdigest()):
            raise ValueError("Independent review does not approve this prepared SFT manifest")

        manifest = load_json(prepared_bytes.decode("utf-8"))
        if (manifest["schema_version"] != 1 or manifest["corpus_id"] != CORPUS_ID or manifest["frozen"] is not True
                or manifest["counts"] != RESULT_COUNTS or manifest["vocab_size"] != 16_384
                or manifest["special_token_ids"] != SPECIAL_IDS or manifest["max_tokens"] != MAX_TOKENS):
            raise ValueError("Prepared manifest differs from the verified SFT protocol")

        tokenizer_path = Path(manifest["tokenizer_path"])
        if not tokenizer_path.is_absolute() or tokenizer_path.resolve() != (base / "tokenizer.json").resolve():
            raise ValueError("Prepared manifest does not reference the original tokenizer")
        paths = {"tokenizer.json": tokenizer_path, **{f"{split}.jsonl": output / f"{split}.jsonl" for split in SPLITS}}
        if set(manifest["files"]) != set(paths):
            raise ValueError("Prepared manifest must cover exactly the splits and tokenizer")
        for name, path in paths.items():
            if file_identity(path) != manifest["files"][name]:
                raise ValueError(f"Prepared file changed before publication: {path}")

        published_path = output / "manifest.json"
        pending_path = output / "manifest.json.partial"
        pending_path.write_bytes(prepared_bytes)
        pending_path.replace(published_path)
        try:
            published, _ = read_frozen_manifest(output, "sft", 16_384)
        except BaseException:
            published_path.replace(pending_path)
            raise
        return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/stage1_v1"))
    parser.add_argument("--old-sft", type=Path, default=Path("data/sft_stage1_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/sft_verified_v2"))
    parser.add_argument("--reports", type=Path, default=Path("reports/comparison_v2"))
    parser.add_argument("--eval-prompts", type=Path, action="append")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--publish-prepared", action="store_true")
    args = parser.parse_args()
    if args.publish_prepared:
        published = publish_prepared(args.output, args.base, args.reports / "verified_sft_review_ready.txt")
        print(f"[verified SFT] frozen {published['counts']}", flush=True)
        return
    if args.eval_prompts is None:
        args.eval_prompts = [Path("evaluation/stage1_v1/prompts.jsonl"), Path("evaluation/comparison_v2/prompts.jsonl")]
    prepare(args)


if __name__ == "__main__":
    main()
