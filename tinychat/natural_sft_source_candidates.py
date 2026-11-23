"""Build auditable public candidate pools for the natural Chinese SFT."""

import argparse
from collections import Counter
import hashlib
import heapq
from itertools import islice
import json
from pathlib import Path
import re
import sys
import time

from tokenizers import Tokenizer

from .chat_format import check_special_ids, contains_reserved, encode_conversation, IGNORE_LABEL
from .sft_data import clean_turn, quality_reason, read_json_records
from .stage1_data import normalized_content


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "data/public_sources/natural_sft_v3"
DEFAULT_OUTPUT = ROOT / "data/natural_sft_v3_candidates/public"
DEFAULT_REPORT = ROOT / "reports/natural_sft_v3"
TOKENIZER = ROOT / "data/stage1_v1/tokenizer.json"
MAX_TOKENS = 512
SOURCES = {
    "cmrc_train": ("hfl/cmrc2018", "137f2c45a24275fb68f6961c4d357f46288886aa", "cc-by-sa-4.0", "data/train-00000-of-00001.parquet"),
    "cmrc_validation": ("hfl/cmrc2018", "137f2c45a24275fb68f6961c4d357f46288886aa", "cc-by-sa-4.0", "data/validation-00000-of-00001.parquet"),
    "moss_helpfulness": ("OpenMOSS-Team/moss-002-sft-data", "d530c921add4c7551b4e2309e0e7fe9468278311", "cc-by-4.0", "zh_helpfulness.json"),
    "moss_honesty": ("OpenMOSS-Team/moss-002-sft-data", "d530c921add4c7551b4e2309e0e7fe9468278311", "cc-by-4.0", "zh_honesty.json"),
}

MOSS_PAIR = re.compile(r"\s*\[Human\]\s*[:：](.*?)<eoh>\s*\[MOSS\]\s*[:：](.*?)<eoa>", re.S)
ROLE = re.compile(r"(?im)\[\s*(?:human|moss|assistant|system|user)\s*\]|<\s*(?:eo[ah]|\|[^\n>]+\|)\s*>|^\s*(?:Human|Assistant|System|MOSS|User|用户|助手)\s*[:：]")
IDENTITY = re.compile(r"\b(?:MOSS|ChatGPT|OpenAI|GPT-?[34])\b|复旦(?:大学|NLP)|上海人工智能实验室|我是.{0,12}(?:助手|人工智能|语言模型)|我(?:的名字|由.{0,15}(?:训练|开发|创建))", re.I)
RISK = re.compile(
    r"自杀|自残|抑郁症|医疗|医生|处方|服药|用药|药物|患者|诊断|疗法|临床|病症|癌症|牙科|急救|过敏|减肥|疾病|感染|食品中毒|生肉|食用野生"
    r"|律师|法律|税务|报税|税率|贷款|股票|基金|证券|加密货币|虚拟币|利率|投资|保单|理财|诈骗|传销|银行账户|信用卡|护照|签证"
    r"|窃取|走私|违禁|偷盗|杀人|伤害他人|殴打|袭击|谋杀|暴力|枪支|武器|炸药|弹药|毒品|恐怖主义|种族歧视|黑客|漏洞|越狱|渗透|破解|键盘记录|恶意软件|绕过验证|密码泄露"
    r"|病毒|病原[体菌]|转基因|基因编辑|合成生物|细菌培养|核武器|危险品|性侵|色情|赌博|隐私数据|习近平|政府政策|总统|党派|选举|军事|战争"
    r"|加热易燃|一氧化碳|易燃易爆|高压电|电线接头|漂白剂|农药"
)
TECHNICAL = re.compile(
    r"```|https?://|www\.|\b(?:python|javascript|java|SQL|C\+\+|printf|import|function)\b|代码|编程|计算机程序|正则表达式|可执行文件|系统架构|深度学习"
    r"|积分|微分|矩阵|数列|多项式|方程|导数|函数|概率论|几何证明|算法|复杂度|量子|相对论|元素周期|摩尔|化学反应|热力学|电磁|声波频率"
    r"|聊天机器人|自然语言处理|语音识别技术|机器学习|智能机器人|神经网络|人工智能技术|语义(?:分析|理解)|最新(?:新闻|疫情|消息|版本|发展|数据|进展)"
    r"|详细介绍|详细讲解|研究报告|学术论文|写一(?:篇|首)|举出.{0,5}(?:国家|名人)|著名|名胜古迹|历史文化|创始于|出身于|民国|古代|公元|维基百科|据(?:报道|统计|研究)"
    r"|\d{3,}|\d\s*[%％]|\d{1,2}[:：]\d{2}|[a-zA-Z]\s*=[^=]", re.I
)
BAD_FORMAT = re.compile(r"\ufffd|\[\s*(?:插入|待填|姓名|你的名字|日期|收件人)[^\]\n]*\]|(?:\.\.\.|……)\s*$|(?m:^\s*#{2,}\s)|<[/a-z]+>|\|.{0,60}\|")
OVERCLAIM = re.compile(r"(?:我(?:可以|会|已经|刚刚)(?:帮你)?(?:预订|联系|发送|查询实时|打开你|查到了)|我(?:亲自|曾经|昨天|上次).{0,15}(?:体验|吃过|联系|推荐|去过|买了)|(?:百分之百|百分百|一定能|保证可以).{0,8}(?:成功|赚钱|解决))")
CONVERSATIONAL = re.compile(r"^(?:你好[啊呀]?[！!。]?|早[上安][好呀]?[！!。]?|晚上好[！!。]?|你好吗[？?]?|谢谢(?:你|啦|了)?[！!。]?|晚安[！!。]?|最近怎么样[？?]?|在吗[？?]?)$")
PERSONAL = re.compile(r"我(?:最近|今天|明天|平时|刚刚|总是|这几天|已经|有点|很|比较|感觉|觉得|想|准备|打算|买了|需要|家|朋友|同事|室友|没|不会|不知道|不太|的)|能帮我|给我(?:个|点|一点|一些|几条)")
EVERYDAY = re.compile(
    r"厨房|做饭|煮饭|早餐|晚饭|午饭|带饭|炒菜|水果|收纳|家务|打扫|清洁|整理|衣服|衣柜|房间|行李|书桌|书包|礼物|生日|节日|搬家|通勤|下班|周末|出门|下雨|散步|旅行计划|户外活动|骑车|跑步|锻炼|休息|睡觉|早起"
    r"|读书|阅读|做笔记|笔记本|写日记|学习计划|练习|英语口语|学习习惯|复习|作业|上课|同学|自我介绍|面试|会议|沟通|聊天|朋友|同事|队友|室友|聚会|活动安排|邀请|回复|答复|道歉|感谢|道别|说服"
    r"|紧张|焦虑|烦恼|难过|生气|开心|压力|心情|尴尬|孤单|情绪|失落|担心|无聊|习惯|自律|拖延|专注|效率|计划|时间管理|兴趣|爱好|手工|画画|照片|拍照|摄影|园艺|阳台|养花|植物|猫咪|小狗|宠物|电影|音乐"
)
PRACTICAL = re.compile(r"怎么|怎样|如何|能不能|可不可以|有什么|想要|想让|想学|想请|帮我|请给|能帮|给.{0,5}建议|推荐|适合|比较好|好不好|合适吗|怎么办|有什么办法|需要注意|该注意|我该|要不要|你觉得")
EXPLAIN = re.compile(r"^(?:请(?:你)?\s*)?(?:解释|说明|简述|什么是|什么叫|为什么|为何|用.{1,16}(?:解释|说明))|是什么[？?。]?$|(?:有什么|有何)区别[？?。]?$|有什么不同[？?。]?$")
SIMPLE_CONCEPT = re.compile(r"比喻|成语|谚语|俗语|同义词|反义词|标点|逗号|句号|词义|意思|理解|例子|概括|大意|友谊|友情|诚信|诚实|尊重|礼貌|合作|分工|目标|习惯|坚持|自信|自律|同理心|耐心|沟通|倾听|反馈|阅读|日记|写作|分类|整理|反思|专注|效率|计划|时间管理|想象力|创造力|彩虹|影子|露珠|霜|下雨|水循环|发芽|落叶|浮在水面|季节变化|蒸发|融化|凝固")
CORRECTION = re.compile(r"(?:你|我)(?:刚才|刚刚|前面|之前)(?:说|提|问|指)|我说的是|我(?:的意思是|是说|指的是|想说的是)|我(?:说|记)错了|不是.{0,28}(?:而是|是说|是指|才是)|改成|换成|纠正|更正|重新(?:说|写)|不对[，,。]|我是问|我问的是|你误会|我没有说")
REFERENCE = re.compile(r"刚才|上面|上一(?:条|个|段)|之前.{0,10}(?:说|提)|前面|(?:第[一二三四五12345](?:个|点|种|条)|这个|那个|这些|那些|其中|它们)|再(?:多|简单|具体|详细|补充|举|解释|说|给)|(?:具体|详细)一点|能举.{0,3}例|换个说法|短一点|简短些|你说的|你提到|按照你说|你推荐|接着|^\s*(?:那|那么|除此之外|还有|然后|所以|如果|假如)|(?:和|跟).{1,20}相比|呢[？?]?$|还有别的|还有其他")
FILLER_BIGRAMS = frozenset("可以 一个 我们 他们 什么 如何 怎么 这些 那些 这样 需要 比较 可能 一些 如果 相关 不同 方式 方面 其中 进行 使用 方法 具体 时间 帮助 注意 更好 主要 情况 推荐 一般 了解 关于 介绍 解释 这个 那个 还有 你说 你的 任何 当然 例如 建议".split())
SENTENCE_END = re.compile(r"[。！？!?]+[”’」』）】》]*|\n+")
LEADING_REFERENCE = re.compile(r"^(?:他|她|它|他们|这些|其中|这一|该[地剧片校国队文人]|此时|而|因此|其后)")
MOSS_UNGROUNDED_TOPIC = re.compile(
    r"影视|电影|电视剧|网剧|剧集|导演|演员|主演|豆瓣|影评|演职员|明星|票房|奥斯卡|艺人|《[^》]+》"
    r"|人工智能|计算机|机器人|聊天模型|机器情感|机器意识|理解情感|主观体验|感官输入|\bAI\b", re.I
)
MOSS_ABSTRACT_PROMPT = re.compile(r"(?:实现|制定|设定|达成).{0,6}(?:个人)?目标|提高.{0,6}(?:生活质量|工作效率)|处理(?:好)?情绪问题|管理情绪|调节情绪|激励自己|克服惰性|建立自信|发掘潜力|保持积极")
MOSS_UNSUPPORTED = re.compile(r"实践证明|研究表明|据(?:调查|评分|数据)|无论你想|都能为你提供|实施惩罚|\b100%|保证.{0,15}(?:有效|成功|提高)")
MOSS_VAGUE_OVERLAP = re.compile(r"生活质量|工作效率|时间管理|积极心态|自我价值|个人目标|正面思维|身体健康|情绪管理|心理健康|健康生活|保持积极|实践证明|具体措施|这些习惯|这些建议|这些办法|以上方法|以上这些|提升生活|发挥潜力|有帮助|会有用|好处|效果不错|长期坚持|达成目标")
MOSS_SIMPLE_EXPLAIN = re.compile(r"(?:什么是|什么叫|什么意思|是什么意思|含义|怎样理解).{0,18}(?:比喻|谚语|俗语|标点|逗号|句号|同义词|反义词)|(?:这个成语|这个词语|这句话).{0,12}(?:意思|理解)|为什么.{0,18}(?:下雨|有彩虹|有影子|会发芽|会落叶|会融化|会蒸发)")

REASON_QUESTION = re.compile(r"为什么|为何|何故|何以|原因(?:是|在于|有|为|是什么|有哪些|[？?])|何种原因|什么原因")
PURPOSE_QUESTION = re.compile(r"用途|用于|目的|作用|用来|用在|拿来|旨在")
EFFECT_QUESTION = re.compile(r"(?:有|产生|带来|造成).{0,12}(?:影响|效果|意义|好处|结果)|(?:影响|效果|意义|好处)(?:是什么|有哪些)")
PRIMARY = r"(?:主要|首要)(?:是|在于|为)?\s*"
PURPOSE_SUPPORT = r"(?:用(?:于|来|在)|旨在|以便|为了|(?:用途|作用|目的)(?:是|在于|为))"
REASON_SUPPORT = r"(?:因为|由于|缘于|源于|之所以|由此|(?:原因|缘由)(?:是|在于|为)|故而)"
EFFECT_SUPPORT = r"(?:导致|造成|产生|使得|带来|有助于|从而|(?:影响|效果|意义|好处)(?:是|在于|为))"
CMRC_EMPTY_FIELD = re.compile(r"[:：]\s*[,，)）]|[（(]\s*[)）]|[、，]\s*[。！？!?]")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def message(role, content):
    return {"role": role, "content": content}


def length_metrics(tokenizer, messages):
    encoded = encode_conversation(tokenizer, messages)
    assistant_turns = len(messages) // 2
    supervised = sum(label != IGNORE_LABEL for label in encoded["labels"])
    return {"tokens": len(encoded["input_ids"]), "assistant_tokens": supervised,
            "assistant_content_tokens": supervised - assistant_turns - 1}


def source_fields(key, row_id, group_id):
    source, revision, license_name, filename = SOURCES[key]
    return {"source": source, "source_revision": revision, "source_license": license_name,
            "source_file": filename, "source_row_id": row_id, "source_group_id": group_id}


def parse_moss(row):
    if not isinstance(row, dict) or not isinstance(row.get("plain_text"), str):
        return None
    text = row["plain_text"]
    parsed = []
    offset = 0
    while match := MOSS_PAIR.match(text, offset):
        parsed.extend((message("user", clean_turn(match.group(1))), message("assistant", clean_turn(match.group(2)))))
        offset = match.end()
    if not parsed or text[offset:].strip() or row.get("num_turns") != len(parsed) // 2:
        return None
    return parsed


def moss_quality(pair, *, quality_v3=False):
    for item in pair:
        text = item["content"]
        if not text or contains_reserved(text) or ROLE.search(text):
            return "format_or_role_token"
        if IDENTITY.search(text):
            return "source_identity"
        if RISK.search(text):
            return "high_risk_topic"
        if TECHNICAL.search(text):
            return "complex_or_external_facts"
        if BAD_FORMAT.search(text):
            return "bad_format"
        if quality_v3 and MOSS_UNGROUNDED_TOPIC.search(text):
            return "ungrounded_facts_or_recommendation"
    answer = pair[1]["content"]
    if OVERCLAIM.search(answer):
        return "unsupported_claim"
    if quality_v3:
        question = pair[0]["content"]
        if MOSS_UNSUPPORTED.search(answer):
            return "unsupported_or_extreme_advice"
        if MOSS_ABSTRACT_PROMPT.search(question) and not PERSONAL.search(question):
            return "abstract_advice_template"
    if re.search(r"(?:可以帮你|请提供|请告诉我).{0,30}(?:更多信息|更多资料|具体内容)", answer) and not answer.endswith(("?", "？")):
        return "answer_missing"
    return quality_reason(pair)


def single_category(question, *, quality_v3=False):
    if CONVERSATIONAL.fullmatch(question):
        return "natural"
    if EXPLAIN.search(question):
        if quality_v3:
            return "grounded" if MOSS_SIMPLE_EXPLAIN.search(question) else None
        return "grounded" if SIMPLE_CONCEPT.search(question) else None
    if EVERYDAY.search(question) and (PERSONAL.search(question) or PRACTICAL.search(question)):
        return "natural"
    return None


def shared_terms(left, right):
    def bigrams(text):
        return {part[index:index + 2] for part in re.findall(r"[\u3400-\u9fff]+", text)
                for index in range(len(part) - 1)}

    return (bigrams(left) & bigrams(right)) - FILLER_BIGRAMS


def dependency(previous, question, answer, *, quality_v3=False):
    earlier = "\n".join(item["content"] for item in previous)
    direct = shared_terms(question, earlier)
    back_reference = shared_terms(answer, earlier)
    if quality_v3:
        # A reference word alone establishes the prompt's dependency, but the
        # answer itself must carry recognizable detail from the prior content.
        old = MOSS_VAGUE_OVERLAP.sub(" ", previous[-1]["content"])
        new = MOSS_VAGUE_OVERLAP.sub(" ", answer)
        answer_terms = shared_terms(old, new) - {"以上", "所有", "达到", "努力", "能够", "获得", "提高", "大家", "这些", "一些", "很好", "应该", "知道", "真正", "自身", "自己", "保持", "坚持", "尝试", "提升", "选择", "改善", "有用", "有趣", "想要", "想法", "个人", "有益"}
        concrete = len(answer_terms) >= 2 and len(normalized_content(answer)) >= 18
        if not concrete and not (CORRECTION.search(question) and direct and len(shared_terms(question, new)) >= 2):
            return None
    if CORRECTION.search(question) and (direct or len(back_reference) >= 2):
        return "correction"
    if REFERENCE.search(question) and (direct or (len(back_reference) >= 2 and len(question) <= 32)):
        return "follow_up"
    preceding_answer = previous[-1]["content"]
    if preceding_answer.endswith(("？", "?")) and not question.endswith(("？", "?")):
        if len(question) <= 32 and (direct or len(back_reference) >= 2):
            return "clarification"
    return None


def moss_candidate(tokenizer, row, key, index, *, quality_v3=False):
    messages = parse_moss(row)
    if messages is None:
        return None, "invalid_conversation"
    first_reason = moss_quality(messages[:2], quality_v3=quality_v3)
    if first_reason:
        return None, first_reason
    category = single_category(messages[0]["content"], quality_v3=quality_v3)
    if category is None:
        return None, "outside_intended_tasks"
    metrics = length_metrics(tokenizer, messages[:2])
    if metrics["tokens"] > MAX_TOKENS:
        return None, "over_context_length"

    selected = messages[:2]
    scanned_dependency = []
    stop_reason = "source_end"
    for end in range(4, len(messages) + 1, 2):
        prefix = messages[:end]
        question, answer = (item["content"] for item in prefix[-2:])
        stop_reason = moss_quality(prefix[-2:], quality_v3=quality_v3) or quality_reason(prefix)
        if stop_reason:
            break
        relation = dependency(prefix[:-2], question, answer, quality_v3=quality_v3)
        if relation is None:
            stop_reason = "independent_or_unproven_turn"
            break
        next_metrics = length_metrics(tokenizer, prefix)
        if next_metrics["tokens"] > MAX_TOKENS:
            stop_reason = "over_context_length"
            break
        selected, metrics = prefix, next_metrics
        scanned_dependency.append(relation)
        stop_reason = "source_end"

    source_id = row.get("id")
    group_id = "moss-002-sft-data:first-user:" + digest(normalized_content(messages[0]["content"]))
    result = source_fields(key, str(source_id) if source_id is not None else str(index), group_id)
    result.update({"messages": selected, "category": "multi" if scanned_dependency else category,
                   "subtype": scanned_dependency[-1] if scanned_dependency else ("explanation" if category == "grounded" else "everyday"),
                   **metrics,
                   "transforms": ["remove_source_system_prefix", "parse_source_role_delimiters",
                                  "normalize_turn_whitespace", "select_complete_prefix"],
                   "source_evidence": {"source_row_index": index, "original_turns": len(messages) // 2,
                                       "retained_turns": len(selected) // 2, "dependency": scanned_dependency,
                                       "prefix_stop_reason": stop_reason}})
    if quality_v3:
        result["transforms"].append("screen_topic_and_concrete_dependency_v3")
    return result, None


def sentence_spans(context):
    start = 0
    for match in SENTENCE_END.finditer(context):
        end = match.end()
        if context[start:end].strip():
            yield start, end
        start = end
    if context[start:].strip():
        yield start, len(context)


def cmrc_prompt(question, passage, with_evidence, variant):
    if with_evidence:
        instructions = ("按原文回答，并引用支持答案的一句原文。", "请回答问题，再摘出能支持答案的原句。",
                        "这道题的答案是什么？请一并写出依据原句。")
    else:
        instructions = ("请根据下面的材料回答问题。", "读下面这段文字，直接回答问题。",
                        "只根据这段材料，回答最后的问题。")
    return f"{instructions[variant]}\n\n{passage}\n\n{question}"


def semantic_support(question, context, sentence_start, sentence_end, offset, gold):
    """Require explicit local evidence for questions that distinguish relation types."""
    for name, pattern in (("reason", REASON_QUESTION), ("purpose", PURPOSE_QUESTION), ("effect", EFFECT_QUESTION)):
        if pattern.search(question):
            break
    else:
        return "span"
    sentence = context[sentence_start:sentence_end]
    before_gold = sentence[:offset - sentence_start]
    clause = re.split(r"[；;。？！?!]", before_gold)[-1]
    # Predicates must precede or start the answer, not follow it. Preserve
    # '主要用于' as a unit so '主要贸易伙伴' cannot validate a use question.
    lead = (clause[-75:] + gold[:12]).replace(" ", "")
    primary = bool(re.search(r"(?:主要|首要).{0,4}(?:" + pattern.pattern + ")", question))
    if name == "purpose":
        predicate = PRIMARY + PURPOSE_SUPPORT if primary else PURPOSE_SUPPORT
        if primary:
            predicate += r"|(?:主要|首要)的?(?:用途|作用|目的)(?:是|在于|为)"
        action = re.search(r"(?:用于|用来|旨在)([\u4e00-\u9fff]{1,8}?)(?:什么|哪些|何种|哪种)", question)
        if action and action.group(1) not in lead:
            return None
    elif name == "reason":
        predicate = REASON_SUPPORT
        if primary:
            predicate = PRIMARY + REASON_SUPPORT + r"|(?:主要|首要)的?(?:原因|缘由)(?:是|在于|为)"
    else:
        predicate = EFFECT_SUPPORT
        if primary:
            predicate = PRIMARY + EFFECT_SUPPORT + r"|(?:主要|首要)的?(?:影响|效果|意义|好处)(?:是|在于|为)"
    return name if re.search(predicate, lead) and shared_terms(question, sentence) else None


def cmrc_candidate(tokenizer, row, key, index, *, evidence_v2=False, quality_v3=False):
    if not isinstance(row, dict):
        return None, "invalid_row"
    row_id, context, question, answers = (row.get(field) for field in ("id", "context", "question", "answers"))
    if not all(isinstance(field, str) and field.strip() for field in (row_id, context, question)):
        return None, "invalid_question_or_context"
    if not isinstance(answers, dict) or not isinstance(answers.get("text"), list) or not isinstance(answers.get("answer_start"), list):
        return None, "invalid_answer_fields"
    if not answers["text"] or len(answers["text"]) != len(answers["answer_start"]):
        return None, "invalid_answer_fields"
    valid = [(text, start) for text, start in zip(answers["text"], answers["answer_start"])
             if isinstance(text, str) and text.strip() and type(start) is int and start >= 0
             and context[start:start + len(text)] == text]
    if not valid:
        return None, "no_verified_answer_offset"
    if any(contains_reserved(text) or ROLE.search(text) or RISK.search(text) for text in (context, question)):
        return None, "unsafe_or_invalid_source_content"

    spans = list(sentence_spans(context))
    # Prefer the most commonly annotated offset. Source order breaks a tie.
    ordered_answers = [annotation for annotation, _ in Counter(valid).most_common()]
    variant = int(digest(row_id)[:4], 16)
    failures = Counter()
    for gold, offset in ordered_answers:
        anchor = next((i for i, (start, end) in enumerate(spans) if start <= offset < end), None)
        last = next((i for i, (start, end) in enumerate(spans) if start < offset + len(gold) <= end), None)
        if anchor is None or last is None:
            continue
        sentence_start, sentence_end = spans[anchor][0], spans[last][1]
        sentence = context[sentence_start:sentence_end].strip()
        support = "offset_only"
        if quality_v3:
            support = semantic_support(question, context, sentence_start, sentence_end, offset, gold)
            if support is None:
                failures["question_evidence_relation_unproven"] += 1
                continue
            if CMRC_EMPTY_FIELD.search(sentence):
                failures["broken_evidence_sentence"] += 1
                continue
        request_evidence = variant % 20 != 0 if evidence_v2 else variant % 3 == 0
        with_evidence = (request_evidence and anchor == last and 10 <= len(sentence) <= 108
                         and len(sentence) >= len(gold) + 6 and "“" not in sentence and "”" not in sentence
                         and not LEADING_REFERENCE.search(sentence) and bool(shared_terms(question, sentence)))
        answer = gold.strip()
        if with_evidence:
            answer = (f"{answer}\n原文写道：“{sentence}”", f"{answer}\n依据是：“{sentence}”",
                      f"{answer}\n材料里对应的一句是：“{sentence}”")[(variant // 3) % 3]

        def encode_window(left, right):
            start, end = spans[left][0], spans[right][1]
            prompt = cmrc_prompt(question, context[start:end].strip(), with_evidence, (variant // 9) % 3)
            messages = [message("user", prompt), message("assistant", answer)]
            return start, end, messages, length_metrics(tokenizer, messages)

        # Keep the whole passage when it fits. Otherwise grow a complete-sentence
        # window around the gold span. Adjacent text supplies pronoun antecedents.
        best = encode_window(0, len(spans) - 1)
        if best[3]["tokens"] > MAX_TOKENS:
            best = encode_window(anchor, last)
            if best[3]["tokens"] > MAX_TOKENS:
                continue
            left, right = anchor, last
            while left > 0 or right < len(spans) - 1:
                if left > 0 and (right == len(spans) - 1 or anchor - left <= right - last):
                    options = ((left - 1, right), (left, right + 1))
                else:
                    options = ((left, right + 1), (left - 1, right))
                for new_left, new_right in options:
                    if new_left < 0 or new_right >= len(spans):
                        continue
                    window = encode_window(new_left, new_right)
                    if window[3]["tokens"] <= MAX_TOKENS:
                        best, left, right = window, new_left, new_right
                        break
                else:
                    break

        start, end, messages, metrics = best
        if quality_v3 and CMRC_EMPTY_FIELD.search(context[start:end]):
            failures["broken_retained_passage"] += 1
            continue
        match = re.fullmatch(r"(.+)_QUERY_\d+", row_id)
        source_doc = match.group(1) if match else None
        group_id = "cmrc2018:document:" + (source_doc or digest(normalized_content(context)))
        result = source_fields(key, row_id, group_id)
        result.update({"messages": messages, "category": "grounded", **metrics,
                       "subtype": "quote" if with_evidence else "extractive",
                       "transforms": ["verify_source_answer_offset", "intact_sentence_window", "request_grounded_answer",
                                      "quote_verified_evidence_sentence" if with_evidence else "return_verified_answer_span"],
                       "source_evidence": {"source_row_index": index, "source_doc_id": source_doc,
                                           "context": context, "question": question, "gold_answer": gold,
                                           "gold_start": offset, "answer_start": offset, "valid_annotations": len(valid),
                                           "invalid_annotations": len(answers["text"]) - len(valid),
                                           "passage_start": start, "passage_end": end,
                                           "evidence_sentence_start": sentence_start,
                                           "evidence_sentence_end": sentence_end,
                                           "question_evidence_relation": support,
                                           "evidence_sentence": sentence if with_evidence else None}})
        if quality_v3:
            result["transforms"].append("screen_explicit_question_evidence_relation_v3")
        return result, None
    return None, failures.most_common(1)[0][0] if failures else "over_context_or_unusable_sentence_window"


def input_path(root, key):
    source, _, _, filename = SOURCES[key]
    return root / source.split("/", 1)[1] / filename


def source_rows(path, key):
    if key.startswith("cmrc_"):
        from pyarrow import parquet

        for batch in parquet.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
    else:
        with path.open("rb") as stream:
            yield from read_json_records(stream)


def emit(event, **payload):
    print(json_line({"event": event, **payload}), flush=True)


def build_source(tokenizer, key, path, out, max_rows=None, *, evidence_v2=False, source_range=None, quality_v3=False):
    counts, stops, totals = Counter(), Counter(), Counter()
    seen_prompts, groups = set(), set()
    started = time.monotonic()
    last_progress = started
    rows = source_rows(path, key)
    if max_rows is not None:
        rows = islice(rows, max_rows)
    for index, row in enumerate(rows):
        counts["scanned"] += 1
        if key.startswith("cmrc_"):
            result, reason = cmrc_candidate(tokenizer, row, key, index, evidence_v2=evidence_v2, quality_v3=quality_v3)
        else:
            result, reason = moss_candidate(tokenizer, row, key, index, quality_v3=quality_v3)
        if result is None:
            counts[f"rejected:{reason}"] += 1
        else:
            prompt_key = tuple(normalized_content(item["content"]) for item in result["messages"][::2])
            if prompt_key in seen_prompts:
                counts["rejected:duplicate_prompt"] += 1
                continue
            seen_prompts.add(prompt_key)
            groups.add(result["source_group_id"])
            counts[f"accepted:{result['category']}"] += 1
            counts[f"subtype:{result['subtype']}"] += 1
            totals["tokens"] += result["tokens"]
            totals["assistant_tokens"] += result["assistant_tokens"]
            totals["assistant_content_tokens"] += result["assistant_content_tokens"]
            totals["max_tokens"] = max(totals["max_tokens"], result["tokens"])
            if "prefix_stop_reason" in result["source_evidence"]:
                stops[result["source_evidence"]["prefix_stop_reason"]] += 1
            if source_range is not None:
                result["source_evidence"]["source_range"] = source_range
            if out:
                out.write(json_line(result) + "\n")
        now = time.monotonic()
        if now - last_progress >= 8:
            accepted = sum(counts[f"accepted:{cat}"] for cat in ("grounded", "natural", "multi"))
            emit("progress", source=key, scanned=index + 1, accepted=accepted)
            last_progress = now
    return {"source": key, "source_range": source_range, "counts": counts,
            "prefix_stop_reasons": stops, "token_totals": totals,
            "unique_source_groups": len(groups), "seconds": round(time.monotonic() - started, 2),
            "assistant_tokens_semantics": "assistant content, one end turn per assistant, and final EOS",
            "assistant_content_tokens_semantics": "assistant content only"}


def verify_candidate(tokenizer, candidate):
    messages = candidate["messages"]
    metrics = length_metrics(tokenizer, messages)
    if any(candidate[key] != value for key, value in metrics.items()) or metrics["tokens"] > MAX_TOKENS:
        raise ValueError("Candidate token counts differ from the frozen formatter")
    if (len(messages) > 2) != (candidate["category"] == "multi"):
        raise ValueError("Candidate turn count disagrees with category")
    evidence = candidate["source_evidence"]
    metadata = (candidate["source"], candidate["source_revision"], candidate["source_license"], candidate["source_file"])
    if metadata not in SOURCES.values() or not candidate["source_row_id"]:
        raise ValueError("Candidate source metadata differs from the pinned source")
    if candidate["source"] == "hfl/cmrc2018":
        gold, offset = evidence["gold_answer"], evidence["gold_start"]
        context = evidence["context"]
        start, end = evidence["passage_start"], evidence["passage_end"]
        original_boundaries = {boundary for span in sentence_spans(context) for boundary in span}
        if (start not in original_boundaries or end not in original_boundaries
                or not start <= offset < offset + len(gold) <= end
                or context[offset:offset + len(gold)] != gold
                or evidence.get("answer_start", offset) != offset
                or evidence["question"] not in messages[0]["content"]
                or context[start:end].strip() not in messages[0]["content"]
                or gold.strip() not in messages[1]["content"]):
            raise ValueError("Candidate answer, offset, or retained passage disagrees with source")
        group = evidence["source_doc_id"] or digest(normalized_content(context))
        if candidate["source_group_id"] != "cmrc2018:document:" + group:
            raise ValueError("Candidate group differs from the source document")
        if candidate["subtype"] == "quote":
            sentence_start, sentence_end = evidence["evidence_sentence_start"], evidence["evidence_sentence_end"]
            if not start <= sentence_start <= offset < offset + len(gold) <= sentence_end <= end:
                raise ValueError("Candidate evidence does not contain the gold span")
            sentence = context[sentence_start:sentence_end].strip()
            if sentence not in messages[1]["content"] or evidence.get("evidence_sentence", sentence) != sentence:
                raise ValueError("Candidate evidence differs from the source sentence")
    else:
        if len(evidence["dependency"]) != len(messages) // 2 - 1:
            raise ValueError("Multi-turn candidate is missing dependency records")
        if not 1 <= evidence["retained_turns"] == len(messages) // 2 <= evidence["original_turns"]:
            raise ValueError("Candidate turns disagree with source")
        group = digest(normalized_content(messages[0]["content"]))
        if candidate["source_group_id"] != "moss-002-sft-data:first-user:" + group:
            raise ValueError("Candidate group differs from the source conversation")
        if source_range := evidence.get("source_range"):
            if (candidate["source_file"] != "zh_helpfulness.json" or source_range["byte_start"] != 0
                    or source_range["byte_end_inclusive"] + 1 != source_range["bytes"]
                    or not source_range["complete_original_records"]):
                raise ValueError("Candidate source range metadata is invalid")


def candidate_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_preview(tokenizer, paths, report_dir):
    verified, strata = Counter(), Counter()
    lengths = (128, 256, 512)

    def records():
        for path in paths:
            for candidate in candidate_rows(path):
                verify_candidate(tokenizer, candidate)
                category = candidate["category"]
                verified[category] += 1
                token_band = next(upper for upper in lengths if candidate["tokens"] <= upper)
                stratum = (path.stem, category, token_band)
                strata[stratum] += 1
                rank = digest(json_line((stratum, candidate["source_row_id"], candidate["source_evidence"]["source_row_index"])))
                yield stratum, rank, candidate

    # Keep up to 20 ranked rows per stratum in memory, then cycle across strata.
    by_stratum = {}
    for stratum, rank, candidate in records():
        heap = by_stratum.setdefault(stratum, [])
        entry = (-int(rank, 16), candidate)
        if len(heap) < 20:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    ordered_strata = [sorted(heap, key=lambda entry: entry[0], reverse=True) for _, heap in sorted(by_stratum.items())]
    chosen = []
    for index in range(20):
        for heap in ordered_strata:
            if index < len(heap) and len(chosen) < 20:
                chosen.append(heap[index][1])
    with (report_dir / "source_candidates_preview.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for candidate in chosen:
            stream.write(json_line(candidate) + "\n")
    report = {"verified": verified, "strata": [{"source": source, "category": category, "max_tokens": band,
                                                "count": count} for (source, category, band), count in sorted(strata.items())],
              "preview_rows": len(chosen), "tokenizer": str(TOKENIZER)}
    with (report_dir / "source_candidates_verified.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", choices=SOURCES, help="Process named source(s). Defaults to every pinned file.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--probe-rows", type=int, help="Scan at most this many rows per source and write no files.")
    parser.add_argument("--verify-preview", action="store_true", help="Validate completed candidate files and create the 20-row preview.")
    parser.add_argument("--cmrc-evidence-v2", action="store_true", help="Use revised CMRC evidence mix and filenames.")
    parser.add_argument("--quality-v3", action="store_true", help="Use stricter CMRC relation and MOSS conversational quality screens.")
    parser.add_argument("--moss-helpfulness-prefix", type=Path,
                        help="Ordered JSONL of complete source records from the original helpfulness byte prefix.")
    parser.add_argument("--moss-helpfulness-range-bytes", type=int,
                        help="Number of original bytes received starting at byte zero, before converting to JSONL.")
    args = parser.parse_args()
    if args.probe_rows is not None and (args.probe_rows < 1 or args.verify_preview):
        parser.error("--probe-rows must be positive and cannot be combined with --verify-preview")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_special_ids(tokenizer)
    keys = args.source or list(SOURCES)
    prefix_bytes = args.moss_helpfulness_range_bytes
    if args.moss_helpfulness_prefix is not None:
        if "moss_helpfulness" not in keys or prefix_bytes is None or prefix_bytes <= 0 or args.verify_preview:
            parser.error("A helpfulness prefix requires the helpfulness source and a positive original byte count")
    elif prefix_bytes is not None:
        parser.error("A helpfulness byte count requires --moss-helpfulness-prefix")

    def output_path(key):
        suffix = ".quality_v3" if args.quality_v3 else ""
        if not suffix and args.cmrc_evidence_v2 and key.startswith("cmrc_"):
            suffix = ".evidence_v2"
        return args.output / f"{key}{suffix}.jsonl"

    if args.verify_preview:
        paths = [output_path(key) for key in keys]
        for path in paths:
            if not path.is_file():
                parser.error(f"Candidate file is missing: {path}")
        args.report_dir.mkdir(parents=True, exist_ok=True)
        emit("verified", **write_preview(tokenizer, paths, args.report_dir))
        return

    for key in keys:
        path = input_path(args.source_root, key)
        source_range = None
        if key == "moss_helpfulness" and args.moss_helpfulness_prefix is not None:
            path = args.moss_helpfulness_prefix
            source_range = {"byte_start": 0, "byte_end_inclusive": prefix_bytes - 1, "bytes": prefix_bytes,
                            "complete_original_records": True, "local_file": path.name}
        if not path.is_file() or not path.stat().st_size:
            parser.error(f"Pinned source is missing or empty: {path}")
        emit("source_start", source=key, bytes=path.stat().st_size, probe_rows=args.probe_rows)
        if args.probe_rows is not None:
            emit("probe_complete", **build_source(tokenizer, key, path, None, args.probe_rows,
                                                evidence_v2=args.cmrc_evidence_v2 or args.quality_v3,
                                                source_range=source_range, quality_v3=args.quality_v3))
            continue
        args.output.mkdir(parents=True, exist_ok=True)
        args.report_dir.mkdir(parents=True, exist_ok=True)
        final = output_path(key)
        pending = final.with_suffix(".jsonl.incomplete")
        report_path = args.report_dir / f"source_candidates_{final.stem}.json"
        if final.exists() or report_path.exists():
            raise FileExistsError(f"Candidate output already exists for {key}")
        with pending.open("x", encoding="utf-8", newline="\n") as stream:
            report = build_source(tokenizer, key, path, stream, evidence_v2=args.cmrc_evidence_v2 or args.quality_v3,
                                  source_range=source_range, quality_v3=args.quality_v3)
        pending.rename(final)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        emit("source_complete", path=str(final), **report)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
