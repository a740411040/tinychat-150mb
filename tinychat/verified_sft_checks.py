"""Independently validate task answers from the visible Chinese dialogue."""

from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
import re

from .chat_format import DEFAULT_SYSTEM, IGNORE_LABEL, SPECIAL_IDS, encode_conversation


RULES = {
    "arithmetic": "按运算符计算两个整数。除法都能整除。只写结果。",
    "extraction": "从原文抄出所问字段冒号后的值，保留原字词，只写该值。",
    "format": "将记录里的年、月、日转为指定格式，只写结果。ISO 用 YYYY-MM-DD，短表用 [年,月,日] JSON 数组。",
    "clarification": "检查必填项。空字符串是缺项。缺项时按必填顺序回答‘请补充甲、乙。’；齐全时回答‘资料已齐。’。",
    "correction": "先记下原记录，再用已更正的值覆盖同名字段。之后的修订以最新值为准。只写所问字段现在的值。",
    "filter_select": "表每行依次是编号、组别、分数。选指定组达到最低分且分数最高的一项，并列取靠前者。只写编号，无符合项写‘无’。",
    "count": "统计数列中闭区间内且符合奇偶要求的项数。重复项逐个计数。只写整数。",
    "dedup": "把同名项的后续出现删掉，保留第一次出现的相对顺序。只输出 JSON 数组。",
    "sort": "每行依次是编号、组序、分值。按组序升序、分值降序排列，都相同的保持原顺序。只输出编号的 JSON 数组。",
    "set": "按给定运算处理两个集合，重复元素忽略。并是任一边有，交是两边都有，差是仅左边有。只输出升序 JSON 数组。",
    "classify": "每个区间依次给出下限、上限、标签。按左闭右开区间给测量值分组，只写标签。",
    "lookup": "表的横纵位置依次对应所列列名和行名。定位指定的行和列，只写单元格整数。",
    "map": "逐项使用给定词表替换；未列入词表的词原样保留。只输出替换后的 JSON 数组。",
    "string": "删掉字符串里指定的字符，保留其余字符顺序，再按要求统一英文字母大小写。只写结果。",
    "boolean": "给定值和下限按位置一一对应。给定值达到其下限即为满足，按给出的条件求真值。只写‘是’或‘否’。",
    "sequence": "已知序列从周期首项开始反复，按已出现项数往后接指定数量，只输出新接的 JSON 数组。",
    "reorder": "逆序时忽略步数；左移时把头部对应步数的项按原顺序移到末尾。只输出新顺序的 JSON 数组。",
    "reference": "按原话确定所问的人先提到和随后提到的物件。前者指先提到的，后者指随后提到的，只写物件。",
    "constraints": "表每行依次是编号、组别、分数、件数。筛出指定组、达到最低分、件数不超上限的行。只输出原顺序编号的 JSON 数组。",
    "support": "陈述依次给出对象、字段、值。原文中对象和字段的值相等写‘支持’，值不同写‘矛盾’，找不到该对象的字段写‘未提及’。",
}

STARTS = {
    "train": ("请处理下面的小任务。", "请按给定资料答复。", "请核对这份资料。", "请给出此题结果。"),
    "validation": ("来做一次记录核验。", "请解出下列练习。", "请推导这项结果。", "这里有一项核对。"),
    "test": ("帮我完成这项核算。", "想请你检查下面的内容。", "用下面的材料做一道题。", "下面是一个待处理的案例。"),
}
ENDS = {
    "train": ("按题内规定作答。", "答案用所要求的形式。", "依据这份数据作答。", "请遵循指定的输出形式。", "用本题所给内容即可。"),
    "validation": ("以这道题的定义为准。", "输出遵从上面的说明。", "本题只看当前材料。", "按字段描述提交结果。", "请照所列口径处理。"),
    "test": ("用规定的样式给出答案。", "请根据已给规则独立完成。", "只采用这则材料中的信息。", "用这里写明的方式回答。", "请严格按材料中的操作答复。"),
}
UPDATES = {
    "train": ("沿用上一轮资料，只改下面的内容，再按原规则回答。", "原题其他内容不变，请用这项修订重新作答。"),
    "validation": ("继续处理刚才的题目。按下列改动更新，原来的要求仍适用。", "下面只列要改的部分，其余沿用前一轮，再给结果。"),
    "test": ("接着上一问，把以下项目换成新值，随后按同样要求返回答案。", "本轮给出局部更正，前文未更正内容继续有效。请重新回答。"),
}

PATCH_KEYS = {
    "arithmetic": {"右数", "运算"}, "extraction": {"字段"}, "format": {"格式", "日"},
    "clarification": {"填写"}, "correction": {"改写", "字段"}, "filter_select": {"最低分", "组"},
    "count": {"下限", "上限", "奇偶"}, "dedup": {"追加"}, "sort": {"追加"}, "set": {"运算", "右"},
    "classify": {"测量值"}, "lookup": {"行", "列"}, "map": {"追加", "替换"},
    "string": {"删除字符", "大小写"}, "boolean": {"条件", "给定值"},
    "sequence": {"接写数", "已出现数"}, "reorder": {"操作", "步数"},
    "reference": {"说话人", "指代"}, "constraints": {"最低分", "最多件数"},
    "support": {"陈述"},
}

INITIAL_KEYS = {
    "arithmetic": ("左数", "右数", "运算"), "extraction": ("原文", "字段"),
    "format": ("年", "月", "日", "格式"), "clarification": ("事项", "必填", "填写"),
    "correction": ("记录", "已更正", "字段"), "filter_select": ("表", "组", "最低分"),
    "count": ("数列", "下限", "上限", "奇偶"), "dedup": ("清单",), "sort": ("行",),
    "set": ("左", "右", "运算"), "classify": ("测量值", "区间"), "lookup": ("行名", "列名", "表", "行", "列"),
    "map": ("词序列", "替换"), "string": ("字符串", "删除字符", "大小写"),
    "boolean": ("给定值", "下限", "条件"), "sequence": ("周期", "已出现数", "接写数"),
    "reorder": ("清单", "操作", "步数"), "reference": ("原话", "说话人", "指代"),
    "constraints": ("表", "组", "最低分", "最多件数"), "support": ("原文", "陈述"),
}

ROLE_CHAIN = ("user", "assistant")
RESOURCE_LINE = "资料"
PATCH_LINE = "改动"
RULE_TO_FAMILY = {rule: family for family, rule in RULES.items()}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def load_object(text):
    def reject_duplicate_keys(pairs):
        fields = dict(pairs)
        require(len(fields) == len(pairs), "Duplicate keys in a visible record")
        return fields

    def reject_nonfinite(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(text, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)


def equal_value(actual, expected):
    """JSON Boolean values must not compare as integers in task answers."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(equal_value(a, b) for a, b in zip(actual, expected))
    return actual == expected


def prompt_identity(messages):
    prompts = [record["content"] for record in messages if record["role"] == "user"]
    serialized = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def load_visible_fields(lines):
    fields = {}
    for line in lines:
        key, separator, value = line.partition("：")
        require(bool(separator) and bool(key) and key not in fields, "Malformed or duplicated visible field")
        fields[key] = load_object(value)
    return fields


def initial_record(prompt, partition):
    lines = prompt.split("\n")
    require(len(lines) >= 5 and lines[2] == RESOURCE_LINE, "Malformed initial task prompt")
    require(lines[0] in STARTS[partition] and lines[-1] in ENDS[partition], "Prompt template belongs to another split")
    require(lines[1] in RULE_TO_FAMILY, "Unknown or altered task instruction")
    family = RULE_TO_FAMILY[lines[1]]
    context = load_visible_fields(lines[3:-1])
    require(set(context) == set(INITIAL_KEYS[family]), "Unexpected visible scenario fields")
    style = STARTS[partition].index(lines[0]) * len(ENDS[partition]) + ENDS[partition].index(lines[-1])
    return family, style, context


def apply_visible_update(family, original, prompt, partition):
    lines = prompt.split("\n")
    closing = "原有的限制仍有效。" if family == "constraints" else ""
    require(len(lines) >= 3 and lines[0] in {lead + closing for lead in UPDATES[partition]} and lines[1] == PATCH_LINE,
            "Malformed follow-up")
    patch = load_visible_fields(lines[2:])
    require(bool(patch) and set(patch) <= PATCH_KEYS[family], "Unexpected follow-up fields")
    updated = dict(original)
    for field, value in patch.items():
        if field == "改写":
            require(type(value) is dict and bool(value) and set(value) <= set(original["记录"]), "Invalid record correction")
            updated["已更正"] = {**original["已更正"], **value}
        elif field == "填写":
            require(type(value) is dict and bool(value) and set(value) <= set(original["必填"]), "Invalid slot update")
            updated[field] = {**original[field], **value}
        elif field == "追加":
            list_field = {"dedup": "清单", "sort": "行", "map": "词序列"}[family]
            require(type(value) is list and bool(value), "Empty appended data")
            updated[list_field] = [*original[list_field], *value]
        elif family == "constraints":
            previous = original[field]
            require(type(value) is int and type(previous) is int, "Constraint bounds must be integers")
            updated[field] = max(previous, value) if field == "最低分" else min(previous, value)
        else:
            require(type(value) is type(original[field]), "Follow-up changed a visible field type")
            updated[field] = value
    return updated


def recompute(family, context):
    """Return the answer and its validation mode. No generator data is loaded."""
    match family:
        case "arithmetic":
            left, right = context["左数"], context["右数"]
            require(type(left) is type(right) is int, "Arithmetic requires integer operands")
            sign = context["运算"]
            require(sign in {"+", "−", "×", "÷"}, "Unknown arithmetic operator")
            operands = Decimal(left), Decimal(right)
            methods = {"+": Decimal.__add__, "−": Decimal.__sub__, "×": Decimal.__mul__, "÷": Decimal.__truediv__}
            result = methods[sign](*operands)
            require(result == result.to_integral_value(), "Arithmetic result must be an integer")
            return str(int(result)), "text"

        case "extraction":
            field = re.escape(context["字段"])
            hits = re.findall(rf"(?:^|[。；]){field}：([^。；]+)", context["原文"])
            require(len(hits) == 1, "Requested extraction field must occur once")
            return hits[0], "text"

        case "format":
            y, m, d = (context[key] for key in ("年", "月", "日"))
            expected = date(y, m, d)
            mode = context["格式"]
            require(mode in {"ISO", "短表"}, "Unsupported calendar format")
            return (expected, "date") if mode == "ISO" else ([y, m, d], "json")

        case "clarification":
            required, values = context["必填"], context["填写"]
            require(bool(required) and len(required) == len(set(required)) and set(values) == set(required), "Invalid required-slot list")
            require(all(type(value) in (str, int) and value != 0 for value in values.values()), "Invalid slot value")
            missing = [field for field in required if values[field] == ""]
            return ("请补充" + "、".join(missing) + "。" if missing else "资料已齐。"), "text"

        case "correction":
            changes = context["已更正"]
            require(type(changes) is dict and set(changes) <= set(context["记录"]), "Correction references an unknown field")
            key = context["字段"]
            return (changes[key] if key in changes else context["记录"][key]), "text"

        case "filter_select":
            group, threshold = context["组"], context["最低分"]
            best_id, best_score = "无", None
            for identifier, row_group, score in context["表"]:
                if row_group == group and score >= threshold and (best_score is None or score > best_score):
                    best_id, best_score = identifier, score
            return best_id, "text"

        case "count":
            selector = context["奇偶"]
            require(selector in {"奇", "偶", "不限"} and context["下限"] <= context["上限"], "Invalid range or parity")
            count = sum(1 for number in context["数列"] if context["下限"] <= number <= context["上限"]
                        and (selector == "不限" or number % 2 == (selector == "奇")))
            return str(count), "text"

        case "dedup":
            output = []
            for item in context["清单"]:
                if item not in output:
                    output.append(item)
            return output, "json"

        case "sort":
            remaining = list(context["行"])
            result = []
            require(len(remaining) == len({row[0] for row in remaining}), "Sort row identifiers must be distinct")
            while remaining:
                best = min(range(len(remaining)), key=lambda i: (remaining[i][1], -remaining[i][2], i))
                result.append(remaining.pop(best)[0])
            return result, "json"

        case "set":
            left, right = set(context["左"]), set(context["右"])
            operation = context["运算"]
            require(operation in {"并", "交", "差"}, "Unknown set operation")
            values = [item for item in left | right if operation == "并" or
                      (item in left and (item in right) == (operation == "交"))]
            return sorted(values), "json"

        case "classify":
            matches = [label for lo, hi, label in context["区间"] if lo <= context["测量值"] < hi]
            require(len(matches) == 1, "A measurement must belong to exactly one class")
            return matches[0], "text"

        case "lookup":
            rows, columns, table = context["行名"], context["列名"], context["表"]
            require(len(set(rows)) == len(rows) == len(table) and len(set(columns)) == len(columns)
                    and all(len(row) == len(columns) for row in table), "Malformed lookup table")
            return str(table[rows.index(context["行"])][columns.index(context["列"])]), "text"

        case "map":
            replacement = context["替换"]
            return [replacement[item] if item in replacement else item for item in context["词序列"]], "json"

        case "string":
            text, delete = context["字符串"], context["删除字符"]
            require(text.isascii() and delete.isascii() and context["大小写"] in {"大写", "小写"}, "Invalid ASCII normalization")
            output = []
            for char in text:
                if char not in delete:
                    output.append(char.upper() if context["大小写"] == "大写" else char.lower())
            return "".join(output), "text"

        case "boolean":
            flags = [value >= lower for value, lower in zip(context["给定值"], context["下限"], strict=True)]
            require(len(flags) == 3, "Boolean problem must define three conditions")
            first, second, third = flags
            checks = {"三项都满足": flags.count(True) == 3, "至少一项满足": flags.count(True) > 0,
                      "前两项中恰一项满足且第三项满足": sum(flags[:2]) == 1 and third,
                      "第一项满足且后两项不都满足": first and not (second and third)}
            return ("是" if checks[context["条件"]] else "否"), "text"

        case "sequence":
            cycle = context["周期"]
            start, count = context["已出现数"], context["接写数"]
            require(bool(cycle) and 0 <= start <= 36 and 1 <= count <= 8, "Invalid continuation length")
            return (cycle * (1 + (start + count) // len(cycle)))[start:start + count], "json"

        case "reorder":
            entries, steps = context["清单"], context["步数"]
            operation = context["操作"]
            require(bool(entries) and 0 <= steps < len(entries) and operation in {"逆序", "左移"}, "Invalid list rearrangement")
            indices = range(len(entries) - 1, -1, -1) if operation == "逆序" else ((steps + i) % len(entries) for i in range(len(entries)))
            return [entries[i] for i in indices], "json"

        case "reference":
            targets = {}
            for sentence in context["原话"]:
                grammar = re.fullmatch(r"([^：，]+)：我先提到([^，。]+)，随后提到([^，。]+)。", sentence)
                require(grammar is not None and grammar[1] not in targets, "Malformed or ambiguous utterance")
                targets[grammar[1]] = {"前者": grammar[2], "后者": grammar[3]}
            return targets[context["说话人"]][context["指代"]], "text"

        case "constraints":
            ids = [identifier for identifier, group, score, count in context["表"] if group == context["组"]
                   and score >= context["最低分"] and count <= context["最多件数"]]
            return ids, "json"

        case "support":
            records = {}
            for statement in context["原文"]:
                grammar = re.fullmatch(r"([^的：。]+)的([^：。]+)：([^：。]+)。", statement)
                require(grammar is not None and (grammar[1], grammar[2]) not in records, "Malformed or conflicting source passage")
                records[grammar[1], grammar[2]] = grammar[3]
            subject, field, value = context["陈述"]
            if (subject, field) not in records:
                return "未提及", "text"
            return ("支持" if records[subject, field] == value else "矛盾"), "text"
    raise ValueError(f"Unrecognized task family: {family}")


def verify_dialogue(messages, split, expected_family=None, expected_style=None, expected_prompt_id=None):
    require(isinstance(messages, list) and 2 <= len(messages) <= 8 and len(messages) % 2 == 0, "Invalid dialogue length")
    require(all(type(message) is dict and set(message) == {"role", "content"} and type(message["content"]) is str
                and message["content"] for message in messages), "Invalid message shape")
    require(tuple(message["role"] for message in messages) == ROLE_CHAIN * (len(messages) // 2), "Invalid dialogue roles")
    if expected_prompt_id is not None:
        require(prompt_identity(messages) == expected_prompt_id, "Visible prompt identity changed")
    family, style, context = initial_record(messages[0]["content"], split)
    require(expected_family is None or family == expected_family, "Task family differs from the visible instruction")
    require(expected_style is None or style == expected_style, "Task style differs from the visible template")

    previous = None
    for turn in range(0, len(messages), 2):
        if turn:
            context = apply_visible_update(family, context, messages[turn]["content"], split)
        try:
            expected, mode = recompute(family, context)
            answer = messages[turn + 1]["content"]
            if mode == "json":
                valid = equal_value(load_object(answer), expected)
            elif mode == "date":
                valid = re.fullmatch(r"\d{4}-\d{2}-\d{2}", answer) is not None and date.fromisoformat(answer) == expected
            else:
                valid = answer == expected
        except (TypeError, KeyError, IndexError, ArithmeticError) as exc:
            raise ValueError(f"Invalid {family} scenario at assistant turn {turn // 2 + 1}") from exc
        require(valid, f"Incorrect {family} answer at assistant turn {turn // 2 + 1}")
        require(not turn or expected != previous, "Follow-up must change the correct answer")
        previous = expected
    return family, style


def verify_encoded_row(row, tokenizer, max_tokens):
    """Re-encode every raw message and check the roles with an independent token walk."""
    verify_dialogue(row["messages"], row["split"], row["task_family"], row["style"], row["prompt_identity"])
    require(row["id"] == row["prompt_identity"] and row["template_id"] == f"{row['split']}:{row['task_family']}:{row['style']:02d}",
            "Row or template identity is inconsistent")
    expected = encode_conversation(tokenizer, row["messages"])
    require(row["input_ids"] == expected["input_ids"] and row["labels"] == expected["labels"], "Stored encoding differs from messages")
    ids, labels = row["input_ids"], row["labels"]
    require(2 < len(ids) <= max_tokens and len(ids) == len(labels), "Encoded dialogue exceeds context or has mismatched labels")
    require(ids[0] == SPECIAL_IDS["<|bos|>"] and labels[0] == IGNORE_LABEL and
            ids[-1] == labels[-1] == SPECIAL_IDS["<|eos|>"], "Invalid BOS/EOS positions or supervision")
    position = 1
    vocab_size = tokenizer.get_vocab_size()
    raw = [{"role": "system", "content": DEFAULT_SYSTEM}, *row["messages"]]
    for message in raw:
        role, marker = message["role"], SPECIAL_IDS[f"<|{message['role']}|>"]
        require(ids[position] == marker and labels[position] == IGNORE_LABEL, "Misplaced role marker or supervision")
        start = position + 1
        position = ids.index(SPECIAL_IDS["<|end_turn|>"], start, len(ids) - 1)
        require(position > start and all(type(token) is int and len(SPECIAL_IDS) <= token < vocab_size
                                         for token in ids[start:position]), "Empty content or invalid/reserved content token")
        expected_labels = ids[start:position + 1] if role == "assistant" else [IGNORE_LABEL] * (position - start + 1)
        require(labels[start:position + 1] == expected_labels, "Invalid assistant-only supervision or turn end label")
        position += 1
    require(position == len(ids) - 1, "Unexpected tokens after the last turn")
