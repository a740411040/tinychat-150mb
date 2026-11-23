"""Route local questions and freeze a new, source-stratified 96-question probe."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re

from tokenizers import Tokenizer

from .chat_format import DEFAULT_SYSTEM, SPECIAL_IDS, check_special_ids, encode_conversation
from .comparison_sft_data import grams
from .natural_sft_source_candidates import SOURCES
from .natural_sft_v4_rewrite_seeds import DEST as EXISTING_PILOT, locator_key, prior_review_locations
from .natural_sft_v4_semantic_pool import (
    CMRC_PREFIX, CMRC_SEPARATOR, MAX_TOKENS, MOSS, OUTPUT as POOL, REPORT, ROOT, TOKENIZER,
    json_rows, parse_moss_exact, relative, structural_reject, write_json,
)
from .sft_data import read_json_records
from .stage1_data import normalized_content


OUTPUT = ROOT / "data/natural_sft_v4_scale20k"
SCALE_REPORT = REPORT / "scale20k"
DEFAULT_PROTECTED = OUTPUT / "protected_inputs.jsonl"
HONESTY = MOSS / "zh_honesty.json"
PROBE_QUOTAS = {"natural": 32, "grounded": 24, "multi": 24, "instruction": 16}
SEED = 20261123
MAX_FIXED_SINGLE = 400
MAX_FIXED_MULTI = 374

PRIVATE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|\d{17}[0-9xX])(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
BROKEN = re.compile(r"\ufffd|[\u0e00-\u0e7f\u0590-\u06ff\uac00-\ud7af]|\{[%#]|<\|[^>]{1,45}\|>")
ROLE_TEXT = re.compile(r"(?mi)^\s*(?:\[(?:human|moss|assistant|system|user)\]|(?:A|答|答案|MOSS|Assistant|System|User|Human)\s*[:：])")
MISSING_LEAD = re.compile(r"^\s*(?:你说得对|你说的没错|你说得没错|你刚才|你刚说|如你所说|如上所述|继续刚才|刚才讨论|好的[，,]|确实[，,]|那[么我][，,]|(?:这|那)(?:段|篇|个|些|种)\S{0,10}(?:是什么意思|怎么理解|有什么不同|怎么看))")
UNSAFE = re.compile(
    r"处方|剂量|药物副作用|怎么治疗|如何治疗|病因|如何诊断|诊断为|服用.{0,8}(?:药|片)|自杀|自残|抑郁症"
    r"|诉讼|起诉|法律责任|法律规定|如何报税|刑法|合同[纠违]|律师"
    r"|股票|基金投资|买基金|加密货币|收益率|投资组合|期货|如何理财|理财产品|贷款利率"
)
TIME_BOUND = re.compile(
    r"(?:最新|当前|实时|今天|现在|最近[一几][天周月年]).{0,14}(?:新闻|股价|价格|汇率|票价|比分|航班|政策|名单|型号)"
    r"|(?:新闻|报价|股价|汇率|比分|时刻表|预报|票价).{0,8}(?:最新|当前|实时|价格|今天|多少)"
)
PHYSICAL = re.compile(r"^(?:你能|能不能|可以|请你|请帮我|帮我|你可以|我希望你).{0,12}(?:来我家|去我家|接我|来接我|帮我搬|为我下单|替我支付|打电话给|代我订|替我发给)")
LONG_FORM = re.compile(r"(?:不少于|至少|约|大约|写|写出|撰写|生成)\s*(\d{3,5})\s*字|[一二三四五六七八九两几]?千字|(?:长篇|整本|万字)(?:小说|报告|论文|故事|详细文本)")
ASK = re.compile(r"[？?]|回答|指出|选出|求出|(?:哪[一个些里]|什么|多少|为什么|是否|如何|怎样|是谁)")
USER_MATERIAL_HINT = re.compile(
    r"根据(?:以下|下面|上[面述]|所给|给定|这段)|阅读(?:以下|下面|上[面述])|"
    r"(?:以下|下面|下列|如下|上文|给定|所给).{0,16}(?:文本|材料|段落|文章|对话|短文|故事|数据|表格)|"
    r"文中|材料中|对话中|段落中|表中|据文|故事里|上面这段|从这段"
)
MATERIAL_LABEL = r"(?:给一段文本内容|文本内容|对话内容|文章内容|材料|短文|文本|段落|故事|描述|资料|数据(?:表)?|案例|情景)"
LABELED_MATERIAL = re.compile(
    r"(?:\A|\n)\s*" + MATERIAL_LABEL
    + r"[：:]\s*(?P<material>[\s\S]{12,3600}?)(?:\n+|[。！]\s*)(?P<question>(?:问题[一二三1-9]?[：:]|根据[^\n]{0,19}回答[：:问题]*|请问).+)",
    re.S,
)
QUESTION_THEN_MATERIAL = re.compile(
    r"(?:\A|\n)\s*(?P<question>(?:问题|题目)[：:][^\n]{3,350})\n+\s*"
    + MATERIAL_LABEL + r"[：:]\s*(?P<material>[\s\S]{12,3400})\Z"
)
DIALOGUE_THEN_QUESTION = re.compile(
    r"\A\s*(?P<material>(?:[^\n：:]{1,16}[：:][^\n]+\n){3,18})\s*"
    r"(?P<question>(?:根据(?:以上|上面|这段)[^\n]{0,16}回答|问题)[：:][\s\S]{3,})"
)
TAIL_QUESTION = re.compile(r"(?:\n\s*|(?<=[。！]))(?P<question>(?:请问|读完[^\n]{0,12}|这个[^\n]{0,8}|他|她|他们|故事里)[^\n。？?]{2,130}[？?])\s*\Z")
QUOTED = re.compile(r'[“「『"](?P<material>[^\n”」』"]{12,1600})[”」』"]')
TABLE = re.compile(r"(?m)^(?:\s*\|[^\n]+\|\s*\n){2,30}")
STRONG_TASK = re.compile(r"改写|换一种说法|重新表达|润色|翻译|压缩|缩写|总结|概括|提取|排序|合并.{0,8}句|转换成|格式化|列成.{0,8}(?:清单|列表)|重排|拟[一一个标题]|写.{0,8}(?:邮件|短信|通知|留言|标题)|按(?:字母|时间|顺序)")
INSTRUCTION_PAYLOAD = re.compile(r'[：:]\s*\S{4,}|[“「『"][^\n]{4,}[”」』"]|\n\s*\S{6,}')
EVERYDAY = re.compile(
    r"家里|朋友|同事|同学|家人|邻居|室友|周末|下班|学习|开学|上班|考试|面试|工作|收纳|整理|做饭|买菜|烘焙|宠物|"
    r"钥匙|衣服|家具|礼物|邀请|早起|回家|出门|练习|旅行|计划|习惯|工作效率|沟通|聊天|表达|桌面|房间|作息|时间管理|"
    r"移动支付|独处|分享|紧张|失落|烦恼|焦虑|开心|疲惫|尴尬|难过|我想"
)
ASSIGNMENT = re.compile(r"^\s*(?:写一篇|编写|生成|给定|把下|介绍.{0,8}(?:发展|现状)|请[列指]出|阐述|分析|列举)")
DOCUMENT_ACTION = re.compile(r"(?:提取|摘录|挑出|找出|总结|摘要|概括|改写|缩写|标注|分段)")
DOCUMENT_REFERENCE = re.compile(r"这(?:篇|段|则|个)(?:文章|小说|故事|文本|童话|段落|寓言|报告)|(?:上[面述的]{1,3}|刚才写的)(?:文章|故事|内容|材料)")
DOCUMENT_CREATE = re.compile(r"写|讲|创作|描述|编|生成")
DIRECT_REFERENCE = re.compile(r"(?:你(?:刚才|刚)?(?:说|提到|推荐|写出|建议)(?:的)?|关于你说的)\s*[“「『\"]?(?P<target>[^，。？！；、！？:：；”“」』\"?]{2,30})")
GENERIC_TARGET = re.compile(r"^(?:这些|那些|这[个种篇份条]|第[一二三四五0-9]个|建议|方法|东西|内容|话|事)(?:的|们|就是|是|.{0,1})?$")
PRIOR_POINTER = re.compile(r"你(?:刚才|刚)?(?:说|提到|推荐|建议)|(?:这|那)(?:个|种|些|一)(?:方法|计划|办法|方式|做法|安排|建议|思路|措施|设备|项目)|上述|上面|它们|其余|(?:这样|这么)(?:做|办)")
CONSTRAINT = re.compile(
    r"还想(?:加|改|换|去掉)|再加(?:上|入)|换成|改成|去掉|改为|"
    r"(?:只(?:有|能|想)|没有|不方便|没法|无法|不想|必须|要求|限于)(?:[^，。？！：:；]{2,25})|"
    r"预算|设备有限|不能出门|缺少|如果.{2,20}(?:不行|失败|找不到|买不到|没有|无法|时间不够)|"
    r"除了.{2,25}(?:之外|以外)|还想(?:加|要|换)|把.{2,20}(?:换成|改成)"
)
SCOPE_CHANGE = re.compile(r"(?:对了|顺便问|换个话题|另外想问).{0,6}")
NEEDS_MATERIAL = re.compile(
    r"(?:提供|发给我|发一下|给我|贴出|没有看到).{0,10}(?:原文|文章|段落|材料|文本|内容|对话)"
    r"|(?:原文|文章|段落|材料|文本|内容|对话).{0,12}(?:发给我|给我|发一下|贴出来)"
)
SUPPLIED_LATER = re.compile(r"^(?:好|当然|在这里|如下|这就是|材料|文本|给你|可以)[，。:：！\s]")
OPTION_WORD = r"(?:办法|方法|建议|选项|选择|措施|做法|步骤|安排|技巧|方案|思路|策略|练习|资源|活动|计划|途径|技术|条件)"
IMPLICIT_POINTER = re.compile(
    r"(?:上述|上面|前面|刚才提到的|刚才说的|你说的|你提到的|前面那些)(?:的|这些|那几种)?"
    + OPTION_WORD
    + r"|(?:这些|那些|这种|那种|这个|那个|这两种|这几个|这几种)(?:的)?"
    + OPTION_WORD
    + r"|同(?:样的|一)?条件"
)
OTHER_OPTION = re.compile(r"还有(?:没有|没有什么|什么|哪些|别的|其他|其它){0,3}\s*(?:别的|其他|其它)?" + OPTION_WORD)
IMPLICIT_OPERATIONS = (
    ("compare_or_choose", re.compile(r"哪(?:一)?[个种项].{0,14}(?:适合|合适|好|有效|实用|优先)|哪(?:一)?[个种项]|不同.{0,6}(?:效果|作用|特点)|有什么不同|优缺点|有什么区别|相比|比较")),
    ("concretize", re.compile(r"具体|怎么(?:做|用|开始)|如何(?:做|实施|落实|操作|使用)|例子|示范|操作步骤|展开讲|详细解释")),
    ("apply_under_context", re.compile(r"(?:是否|能否|能不|还能|也|仍然).{0,9}(?:适用|有用|有效|适合)|对.{1,12}(?:有用|适用|有效)|有什么(?:好处|作用|效果)")),
    ("avoid_repeating_options", OTHER_OPTION),
)
IN_TURN_CHOICES = re.compile(
    r"(?:我(?:目前|手边|自己)?(?:有|在考虑|准备|在用|想选)|我的选择是|分别是|备选(?:是|有)|选项(?:是|有)|比如说)"
    r"[^。？！?]{2,90}(?:和|、|还是|或者|与)[^。？！?]{1,50}"
)
INLINE_TOPIC_SWITCH = re.compile(r"话说回来|话题一转|问个别的|顺便|对了|换个话题")
IMPLICIT_HEALTH_OR_RISK = re.compile(
    r"(?i)\bburnout\b|职业倦怠|职业过劳|症状|心理咨询|精神科|兽医|胸痛|昏厥|疾病防治|服药|急救|急性|症候群|"
    r"身体不适|确诊|过敏|喝醉|投资|理财|贷款|暴力|自残|自杀|治疗|诊断|用药|临床|处方|律师|诉讼|报税"
)
IMPLICIT_PRACTICAL_CONTEXT = re.compile(r"学(?:做|画|拍|摄影|英语|外语|编织|弹琴|做饭)|练琴|读书|办公|备考|上课|烹饪|创作")
PERSONAL_CLAIM = re.compile(r"我(?:自己|本人|亲自|亲手|亲历|曾经|以前|在工作中|在实际工作中|在生活中|当时|一直在尝试).{0,26}|我(?:使用|构建|训练)了.{0,14}|我的模型")
EXPERIENCE_LOCK = re.compile(
    r"你(?:亲自|本人|当时|曾经|实际在|什么时候|在哪里|工作时|使用了|训练了|构建了|拿到了|经历的)|"
    r"你的(?:亲身|亲历|真实|当时|工作经历|个人经历|训练过程|训练数据|客户案例)|你自己.{0,9}(?:怎么做的|做过|经历过)"
)
LIST_MARKER = re.compile(r"(?m)(?:^|[\n；;])\s*(?:[-*•]|\(?[（]?[0-9一二三四五六七八九十]{1,3}[.、．：:）)]|[一二三四五六七八九]是)\s*")
ACTION_WORD = re.compile(
    r"学|选|练|先|写|试|做|设|寻找|分解|拆|记录|回顾|准备|使用|利用|加入|交流|讨论|投入|调整|改变|清理|了解|检查|"
    r"留出|预留|休息|判断|排列|安排|联系|核对|遵守|对比|收起|定位|定期|熟悉|提高|控制|坚持|看|读|听|换|租|买|挂|放|带|夹|贴"
)
EMPTY_ACTION = re.compile(r"^(?:好的|当然|以下是|你可以|我可以|希望|如果你|很高兴|这很重要|有几种方法).{0,12}$")
V2_PSEUDO_CONTENT = re.compile(
    r"(?mi)^\s*(?:Answer|Reference Answer|Assistant Reply)\s*[:：]|</?(?:tool_call|tool_response|function_call)\b"
)
V2_LONG_MINIMUM = re.compile(r"(?<!\d)([1-9]\d{2,4})\s*(?:个)?字\s*(?:以上|起步|起|至少)")
V2_PLATFORM = r"(?:微信(?:公众平台)?|抖音|小红书|支付宝|淘宝|B站|公众号)"
V2_SERVICE_LIMIT = r"(?:最多|限额|上限|绑定.{0,6}(?:几|多少)|数量限制)"
V2_PLATFORM_LIMIT = re.compile(V2_PLATFORM + r".{0,40}" + V2_SERVICE_LIMIT + "|" + V2_SERVICE_LIMIT + r".{0,45}" + V2_PLATFORM)
V2_UNRESOLVED_WORK = re.compile(r"《[a-z]{5,}》(?:的剧|的电影|的书)?[^。！？?]{0,30}(?:好看|历史背景|讲的什么|剧情)")
V2_EXPLICIT_TRANSFORMATION = re.compile(
    r"^\s*(?:请|帮我)?(?:用.{1,20}(?:语气|口吻|文风)(?:来)?(?:重写|改写)|重写(?:下面|以下|上面)|"
    r"(?:根据|为|给).{1,26}(?:生成|拟|起|写).{0,9}(?:标题|题目)|"
    r"(?:从|请从).{1,24}(?:找出|提取).{0,32}(?:用.{0,6}分隔|按.{0,6}格式))"
)
V2_DIRECT_RECIPE = re.compile(r"(?:有没有|你能|请|能不能)?.{0,18}(?:什么.{0,11}食谱推荐|推荐.{0,12}(?:食谱|菜谱))")


def reject_user(content, *, first=False, substantive=None):
    subject = content if substantive is None else substantive
    if PRIVATE.search(content):
        return "private_identifier_shape"
    if BROKEN.search(content) or ROLE_TEXT.search(content):
        return "broken_text_or_source_role"
    spans = list(LONG_FORM.finditer(subject))
    if any(not match[1] or int(match[1]) >= 600 for match in spans):
        return "oversized_requested_output"
    if UNSAFE.search(subject):
        return "out_of_scope_professional_or_safety_topic"
    if TIME_BOUND.search(subject):
        return "time_sensitive_fact_or_price"
    if PHYSICAL.search(subject):
        return "requires_physical_or_external_action"
    if first and MISSING_LEAD.search(subject):
        return "missing_origin_history"
    if not re.search(r"[\u3400-\u9fff]", subject):
        return "no_chinese_question"
    return None


@dataclass(frozen=True)
class Material:
    span: tuple[int, int]
    question: str
    mode: str


def material_from_user(text):
    for pattern, mode in (
        (LABELED_MATERIAL, "labeled_material_then_question"),
        (QUESTION_THEN_MATERIAL, "labeled_question_then_material"),
        (DIALOGUE_THEN_QUESTION, "visible_dialogue_then_question"),
    ):
        if (found := pattern.search(text)) and ASK.search(found["question"]):
            return Material(found.span("material"), found["question"], mode)
    if table := TABLE.search(text):
        outside = text[:table.start()] + text[table.end():]
        if ASK.search(outside):
            return Material(table.span(), outside, "visible_table_and_question")
    for quote in QUOTED.finditer(text):
        outside = text[:quote.start()] + text[quote.end():]
        payload = quote["material"]
        if ASK.search(outside) and (USER_MATERIAL_HINT.search(outside) or re.search(r"[，。；]", payload)):
            return Material(quote.span("material"), outside, "visible_quote_and_question")
    if found := TAIL_QUESTION.search(text):
        before = text[:found.start("question")]
        if len(before.strip()) >= 28 and re.search(r"[。！？]", before):
            return Material((0, len(before)), found["question"], "untitled_paragraph_then_question")
    return None


def anchor_range(text, start, end, index):
    return {"message_index": index, "chars_zero_based_half_open": [start, end], "text": text[start:end]}


def substantive_anchor(messages, index):
    text = messages[index]["content"]
    start = len(text) - len(text.lstrip())
    limit = min(len(text), start + 96)
    return anchor_range(text, start, limit, index)


def dependency(first, second):
    user, old_reply, followup = (first[0]["content"], first[1]["content"], second["content"])
    if SCOPE_CHANGE.search(followup):
        return None
    if DOCUMENT_REFERENCE.search(followup) and DOCUMENT_ACTION.search(followup) and DOCUMENT_CREATE.search(user):
        if len(normalized_content(old_reply)) >= 35:
            return "document", "previous_document_operation", [substantive_anchor(first, 0), substantive_anchor(first, 1)]
    reference = DIRECT_REFERENCE.search(followup)
    if reference:
        target = reference["target"].strip()
        words = re.split(r"(?:听起来|是指|能不能|能否|具体|什么意思|要怎么|如何|怎么|怎么办|太|不太|有什么|能不能|是否|有用|这点)", target)
        goal = words[0].strip()
        for i, text in ((1, old_reply), (0, user)):
            if len(goal) >= 2 and not GENERIC_TARGET.fullmatch(goal) and (location := text.find(goal)) >= 0:
                return "daily", "referent_verbatim_in_earlier_turn", [anchor_range(text, location, location + len(goal), i)]
    if constraint := CONSTRAINT.search(followup):
        if PRIOR_POINTER.search(followup) or re.search(r"^(?:我)?(?:还想|再加|改成|换成|改为|去掉|除了)", followup):
            return "daily", "named_constraint_inherits_earlier_task", [
                substantive_anchor(first, 0), anchor_range(followup, *constraint.span(), 2),
            ]
    return None


def option_spans(text):
    """Locate distinct earlier actions using list delimiters before scanning clauses."""
    markers = list(LIST_MARKER.finditer(text))
    if len(markers) >= 2:
        ranges = [(marker.end(), markers[i + 1].start() if i + 1 < len(markers) else len(text))
                  for i, marker in enumerate(markers)]
    else:
        ranges = [match.span() for match in re.finditer(r"[^。！？；;\n]{8,}", text)]
    result = {}
    for start, end in ranges:
        phrase = text[start:end]
        start += len(phrase) - len(phrase.lstrip())
        end = min(end, start + 112)
        span = text[start:end].rstrip()
        key = normalized_content(span)
        if len(key) >= 8 and ACTION_WORD.search(span) and not EMPTY_ACTION.fullmatch(span):
            result.setdefault(key, anchor_range(text, start, start + len(span), 1))
    return list(result.values())


def v2_user_shape(messages):
    for item in messages:
        if item["role"] != "user":
            continue
        text = item["content"]
        if V2_PSEUDO_CONTENT.search(text) or re.search(r"[\u0400-\u04ff]", text) and re.search(r"[\u3040-\u30ff]", text):
            return "pseudo_answer_or_mixed_script_corruption"
        if any(int(match[1]) >= 600 for match in V2_LONG_MINIMUM.finditer(text)):
            return "explicit_minimum_length_exceeds_response_budget"
        if V2_PLATFORM_LIMIT.search(text):
            return "unverified_current_platform_limit"
        if V2_UNRESOLVED_WORK.search(text) and not USER_MATERIAL_HINT.search(text):
            return "unresolved_work_reference_with_no_visible_context"
    return None


def implicit_dependency(messages):
    """Return auditable routing anchors and a narrow operation, not semantic approval."""
    if len(messages) < 4:
        return None, "implicit_no_second_exchange"
    first, reply, follow = (messages[index]["content"] for index in (0, 1, 2))
    pointer = IMPLICIT_POINTER.search(follow)
    other = OTHER_OPTION.search(follow)
    if not pointer and not other:
        return None, "implicit_no_lookup_expression"
    if (not (EVERYDAY.search(first) or IMPLICIT_PRACTICAL_CONTEXT.search(first))
            or ASSIGNMENT.search(first) or DOCUMENT_REFERENCE.search(follow)):
        return None, "implicit_outside_daily_learning_work"
    if IMPLICIT_HEALTH_OR_RISK.search(first + "\n" + follow):
        return None, "implicit_health_or_professional_risk"
    if reason := v2_user_shape(messages[:4]):
        return None, reason
    if reject_user(first, first=True) or reject_user(follow):
        return None, "implicit_user_shape_gate"
    if any(PRIVATE.search(item["content"]) or ROLE_TEXT.search(item["content"]) for item in messages[1:4:2]):
        return None, "implicit_unusable_earlier_message"
    if INLINE_TOPIC_SWITCH.search(follow) or IN_TURN_CHOICES.search(follow):
        return None, "implicit_topic_switch_or_self_contained_reference"
    if PERSONAL_CLAIM.search(reply) and EXPERIENCE_LOCK.search(follow):
        return None, "implicit_followup_locks_unsupported_personal_experience"
    applicable = [name for name, pattern in IMPLICIT_OPERATIONS if pattern.search(follow)]
    if not applicable:
        return None, "implicit_no_comparison_refinement_application_or_alternative"
    action = "avoid_repeating_options" if other else applicable[0]
    options = option_spans(reply)
    plural = pointer and re.search(r"这些|那些|两种|几种|几个", pointer[0])
    needs_set = action in {"compare_or_choose", "avoid_repeating_options"} or plural
    if len(options) < (2 if needs_set else 1):
        return None, "implicit_no_locatable_prior_action_set"
    return {
        "operation": action,
        "source_spans": [substantive_anchor(messages, 0), *options[:4]],
    }, None


def fixed_tokens(tokenizer, messages):
    """Measure user/system text and all conversation framing with empty reply payloads."""
    prefix = 3 + len(tokenizer.encode(DEFAULT_SYSTEM, add_special_tokens=False).ids)
    for item in messages:
        if item["role"] == "user":
            prefix += 4 + len(tokenizer.encode(item["content"], add_special_tokens=False).ids)
    return prefix


def ground_evidence(material, message_index):
    return {
        "user_message_index": message_index, "source_type": material.mode,
        "material_chars_zero_based_half_open": list(material.span),
        "adjudication": "Only use the visible material; assess question and evidence sufficiency before answering.",
    }


def cmrc_route(row, text):
    source = row["sources"][0]
    transform = source["transform"]
    start = len(CMRC_PREFIX)
    end = start + transform["passage_chars_zero_based_half_open"][1]
    if not text.startswith(CMRC_PREFIX) or not text[end:].startswith(CMRC_SEPARATOR):
        return None
    question = text[end + len(CMRC_SEPARATOR):]
    answer_left, answer_right = transform["answer_chars_zero_based_half_open"]
    if reject_user(text, first=True, substantive=question):
        return None
    if re.search(r"-\{|(?:^|\W)zh-(?:hans|hant|cn)\s*:|\[\[|\{\{|[（(][，、；\s]*[）)]", text[start:end]):
        return None
    route = ("grounded", 1, "pinned_cmrc_full_document_and_verified_offset", None, [])
    evidence = ground_evidence(Material((start, end), question, "cmrc_complete_context"), 0)
    evidence["annotation_chars_zero_based_half_open"] = [start + answer_left, start + answer_right]
    return route, evidence, question


def route_row(row, tokenizer, *, implicit=None):
    messages = row["messages"]
    user = messages[0]["content"]
    source = row["sources"][0]
    cmrc = source["input"].startswith("cmrc_")
    evidence, question = None, None
    if implicit is not None:
        route = ("multi", 2, "implicit_reference_candidate", "daily", implicit["source_spans"])
    elif cmrc:
        match = cmrc_route(row, user)
        if match is None:
            return None, "cmrc_question_or_full_context_out_of_scope"
        route, evidence, question = match
    else:
        first_material = material_from_user(user)
        reject = reject_user(user, first=True, substantive=first_material.question if first_material else None)
        if reject:
            return None, reject
        route = None
        if first_material:
            route = ("grounded", 1, first_material.mode, None, [])
            evidence, question = ground_evidence(first_material, 0), first_material.question
        if route is None and len(messages) >= 4:
            followup = messages[2]["content"]
            late_material = material_from_user(followup)
            late_subject = late_material.question if late_material else None
            if not reject_user(followup, substantive=late_subject):
                material_request = NEEDS_MATERIAL.search(messages[1]["content"])
                if late_material and material_request and (
                    SUPPLIED_LATER.search(followup) or USER_MATERIAL_HINT.search(followup)
                ):
                    anchors = [
                        substantive_anchor(messages, 0),
                        anchor_range(messages[1]["content"], *material_request.span(), 1),
                    ]
                    route = ("grounded", 2, "visible_material_supplied_after_request", None, anchors)
                    evidence, question = ground_evidence(late_material, 2), late_material.question
                elif multi := dependency(messages[:2], messages[2]):
                    subtype, reason, anchors = multi
                    route = ("multi", 2, reason, subtype, anchors)
        if route is None and STRONG_TASK.search(user) and INSTRUCTION_PAYLOAD.search(user):
            route = ("instruction", 1, "bounded_action_with_visible_payload", None, [])
        if route is None and EVERYDAY.search(user) and not ASSIGNMENT.search(user) and len(user) <= 220:
            route = ("natural", 1, "daily_life_learning_work_or_emotion_cue", None, [])
        if route is None:
            return None, "no_supported_seed_route"

    category, turns, reason, subtype, anchors = route
    selected = [dict(item) for item in messages[:turns * 2]]
    options = row.get("prefix_options")
    if options:
        fixed = options[turns - 1]["tokens"] - options[turns - 1]["supervised_tokens"] + turns
        original_tokens = options[turns - 1]["tokens"]
    else:
        fixed = fixed_tokens(tokenizer, selected)
        original_tokens = len(encode_conversation(tokenizer, selected, final_eos=False)["input_ids"])
    if original_tokens > MAX_TOKENS:
        return None, "original_complete_prefix_exceeds_context"
    if fixed > (MAX_FIXED_MULTI if turns > 1 else MAX_FIXED_SINGLE):
        return None, "insufficient_response_budget"
    sources = row["sources"]
    assistants = list(range(1, len(selected), 2))
    constraints = [{"assistant_message_index": index, "review_focus":
                    "Correct only this assistant turn; keep user text, visible material, and any later referents valid."}
                   for index in assistants]
    seed = {
        "id": f"scale-{row['id']}", "category": category, "source_group": row["group_id"],
        "sources": sources, "pool_prefix_id": row["id"] if row["id"].startswith("prefix-") else None,
        "source_license": source.get("license", source.get("usage")),
        "original_messages": selected, "user_messages": selected[::2],
        "selected_original_message_range_zero_based_half_open": [0, len(selected)],
        "rewrite_assistant_turn_indices": assistants, "constraints": constraints,
        "dependency_anchors": [
            {"user_message_index": 2, "history_through_assistant_message_index": 1, "anchor_type": reason,
             "anchor": "Preserve earlier task and referenced details when rewriting turn 1; old assistant text requires correction.",
             "source_spans": anchors}
        ] if anchors else [],
        "max_context": MAX_TOKENS, "final_eos": False,
        "tokens_without_assistant_content": fixed, "assistant_content_token_budget": MAX_TOKENS - fixed,
        "original_prefix_tokens": original_tokens, "route_reason": reason, "regression_seed": False,
        "old_assistant_text_status": "unverified_reference_only",
    }
    if evidence is not None:
        seed["grounded_evidence"] = evidence
    if subtype is not None:
        seed["multi_subtype"] = subtype
    if implicit is not None:
        seed["implicit_operation"] = implicit["operation"]
        seed["constraints"] = [
            {"assistant_message_index": assistants[0], "review_focus":
             "给出能供第二问实际比较、细化、迁移或排除重复的具体做法。旧答的事实和个人经历须重新判断。"},
            {"assistant_message_index": assistants[1], "review_focus":
             "实际承接第一答的对象和约束；若问其他办法，避开已给动作。无法保真或不需前文时拒收。"},
        ]
    return (seed, question), None


class LexicalIndex:
    """Find a traceable conflict with the established lexical thresholds."""

    def __init__(self):
        self.exact = {}
        self.records = []
        self.postings = defaultdict(list)
        self.anchors = defaultdict(list)

    def add(self, text, identifier, group):
        if not text or text in self.exact:
            return
        index = len(self.records)
        self.exact[text] = index
        self.records.append((text, identifier, group))
        if len(text) >= 12:
            for gram in grams(text):
                self.postings[gram].append(index)
        if len(text) >= 24:
            self.anchors[text[:12]].append(index)

    def match(self, text, *, allow_group=None, ignore_groups=()):
        def result(index, kind):
            _, identifier, group = self.records[index]
            if group not in ignore_groups and (allow_group is None or group != allow_group):
                return kind, identifier, group
            return None

        if text in self.exact and (hit := result(self.exact[text], "exact")):
            return hit
        if len(text) < 12:
            return None
        features = grams(text)
        rarest = sorted(features, key=lambda g: (len(self.postings.get(g, ())), g))
        if len(text) >= 24:
            for offset in range(len(text) - 11):
                for index in self.anchors.get(text[offset:offset + 12], ()):
                    if self.records[index][0] in text and (hit := result(index, "containment")):
                        return hit
            for index in self.postings.get(rarest[0], ()):
                if text in self.records[index][0] and (hit := result(index, "containment")):
                    return hit
        neighbors = set()
        for gram in rarest[:len(features) // 5 + 1]:
            neighbors.update(self.postings.get(gram, ()))
        for index in sorted(neighbors):
            if not (hit := result(index, "five_gram_jaccard")):
                continue
            other = grams(self.records[index][0])
            if 5 * min(len(features), len(other)) < 4 * max(len(features), len(other)):
                continue
            if 9 * len(features & other) >= 4 * (len(features) + len(other)):
                return hit
        return None


def user_keys(messages):
    users = [normalized_content(item["content"]) for item in messages if item["role"] == "user"]
    return users, "|".join(users)


def load_protected(path):
    index, trajectories = LexicalIndex(), {}
    count = 0
    for number, row in enumerate(json_rows(path), 1):
        messages = row["messages"]
        if not messages or any(item["role"] != "user" for item in messages):
            raise ValueError(f"Protected index must be a user-only projection: {path}:{number}")
        users, trajectory = user_keys(messages)
        identity = row.get("id", f"protected-line-{number}")
        index.add(users[0], identity, "protected")
        if len(users) > 1:
            trajectories[trajectory] = identity
        count += 1
    if not count:
        raise ValueError(f"Protected input projection is empty: {path}")
    return index, trajectories, count


def honesty_rows(counts):
    repo, revision, license_name, file_name = SOURCES["moss_honesty"]
    with HONESTY.open("rb") as source:
        for ordinal, row in enumerate(read_json_records(source), 1):
            counts["moss_honesty"] += 1
            messages, spans, reason = parse_moss_exact(row.get("plain_text"), row.get("num_turns"))
            reason = reason or structural_reject(messages)
            if reason:
                yield None, reason
                continue
            raw_id = row.get("id", ordinal)
            group = f"honesty:{raw_id}"
            provenance = {
                "input": "moss_honesty", "repo": repo, "revision": revision, "license": license_name,
                "license_record": relative(MOSS / "README.md"), "source_file": file_name, "local_file": relative(HONESTY),
                "source_row_id": raw_id, "json_array_ordinal_1_based": ordinal, "source_group_id": group,
                "original_message_count": len(messages),
                "transform": {"type": "moss_plain_text_role_framing", "source_num_turns": row.get("num_turns"),
                              "content_chars_zero_based_half_open": spans,
                              "role_map": {"Human": "user", "MOSS": "assistant"}},
            }
            yield {"id": f"honesty-{ordinal:06}", "group_id": group, "sources": [provenance],
                   "messages": messages[:4]}, None


def prior_indexes():
    prior_groups, prior_questions = {}, LexicalIndex()
    for old in json_rows(EXISTING_PILOT):
        identifier = old["id"]
        group = old["source_group"]
        prior_groups[group] = identifier
        users, _ = user_keys(old["user_messages"])
        prior_questions.add(users[0], identifier, group)
    prior_locators, _ = prior_review_locations()
    return prior_groups, prior_questions, prior_locators


def cmrc_duplicate(question, span, peers):
    """Compare question text within one document and avoid repeating its annotation span."""
    for old, old_span, identifier in peers:
        if span == old_span:
            return "same_document_same_annotation_span", identifier
        if question == old:
            return "same_document_exact_question", identifier
        if min(len(question), len(old)) >= 24 and (question in old or old in question):
            return "same_document_contained_question", identifier
        first, second = grams(question), grams(old)
        if min(len(question), len(old)) >= 12 and 9 * len(first & second) >= 4 * (len(first) + len(second)):
            return "same_document_near_question", identifier
    return None


class Intake:
    def __init__(self, protected):
        self.protected, self.protected_paths, self.protected_count = load_protected(protected)
        self.prior_groups, self.prior_questions, self.prior_locators = prior_indexes()
        self.firsts = LexicalIndex()
        self.trajectories = {}
        self.groups = {}
        self.cmrc_questions = defaultdict(list)
        self.materials = LexicalIndex()
        self.group_aliases = {}
        self.edges = []

    def group_root(self, group):
        parent = self.group_aliases.get(group, group)
        while parent != group:
            group, parent = parent, self.group_aliases.get(parent, parent)
        return group

    def register_material(self, seed):
        if not (evidence := seed.get("grounded_evidence")):
            return
        start, end = evidence["material_chars_zero_based_half_open"]
        raw = seed["original_messages"][evidence["user_message_index"]]["content"]
        key = normalized_content(raw[start:end])
        if len(key) < 24:
            return
        group = seed["source_group"]
        if hit := self.materials.match(key, allow_group=group):
            root, peer = self.group_root(group), self.group_root(hit[2])
            if root != peer:
                self.group_aliases[root] = peer
                self.conflict(seed, f"retained_visible_material_group_{hit[0]}", hit[1], hit[2])
        self.materials.add(key, seed["id"], group)

    def freeze_groups(self, records):
        old_ids = {}
        for group, identifier in self.prior_groups.items():
            old_ids[self.group_root(group)] = identifier
        for seed in records:
            group = seed["source_group"]
            seed["original_source_group"] = group
            seed["source_group"] = root = self.group_root(group)
            if not seed["existing_rewrite_pilot_id"] and root in old_ids:
                seed["existing_rewrite_pilot_id"] = old_ids[root]
                seed["seen_in_previous_pilot"] = True

    def conflict(self, seed, reason, other=None, group=None):
        if other is not None:
            self.edges.append({
                "seed_id": seed["id"], "source_group": seed["source_group"], "reason": reason,
                "conflicts_with": other, "conflict_source_group": group,
            })
        return reason

    def add(self, seed, question):
        identifier, group = seed["id"], seed["source_group"]
        user_parts, user_path = user_keys(seed["original_messages"])
        if not all(user_parts):
            return self.conflict(seed, "empty_user_comparison_key")
        first = user_parts[0]
        is_cmrc = seed.get("grounded_evidence", {}).get("source_type") == "cmrc_complete_context"
        protected_parts = [first]
        if question is not None:
            bare = normalized_content(question)
            if bare:
                protected_parts.append(bare)
        if len(user_parts) > 1 and len(user_parts[1]) >= 24:
            protected_parts.append(user_parts[1])
        for prompt in protected_parts:
            if hit := self.protected.match(prompt):
                return self.conflict(seed, f"protected_input_{hit[0]}", *hit[1:])
        if user_path in self.protected_paths:
            return self.conflict(seed, "protected_exact_user_trajectory", self.protected_paths[user_path], "protected")
        if not is_cmrc and group in self.groups:
            return self.conflict(seed, "original_or_first_user_group_already_selected", self.groups[group], group)
        if len(user_parts) > 1 and user_path in self.trajectories:
            other, old_group = self.trajectories[user_path]
            return self.conflict(seed, "exact_user_trajectory", other, old_group)
        document_peers = None
        if is_cmrc:
            bare = normalized_content(question)
            if not bare:
                return self.conflict(seed, "empty_cmrc_question_comparison_key")
            span = seed["grounded_evidence"]["annotation_chars_zero_based_half_open"]
            document_peers = self.cmrc_questions[group]
            if hit := cmrc_duplicate(bare, span, document_peers):
                return self.conflict(seed, hit[0], hit[1], group)
            if len(document_peers) >= 3:
                return self.conflict(seed, "cmrc_document_three_question_limit", document_peers[0][2], group)
        if hit := self.firsts.match(first, allow_group=group if is_cmrc else None):
            return self.conflict(seed, f"first_user_{hit[0]}", *hit[1:])

        if is_cmrc:
            document_peers.append((bare, span, identifier))
        self.firsts.add(first, identifier, group)
        self.register_material(seed)
        if len(user_parts) > 1:
            self.trajectories[user_path] = (identifier, group)
        self.groups.setdefault(group, identifier)
        old_pilot = self.prior_groups.get(group)
        if not old_pilot and (old_match := self.prior_questions.match(first)):
            old_pilot = old_match[1]
        seen_in_previous = any(
            locator_key(source) in self.prior_locators
            for source in seed["sources"] if source["input"] != "moss_honesty"
        )
        seed["existing_rewrite_pilot_id"] = old_pilot
        seed["seen_in_previous_pilot"] = bool(old_pilot or seen_in_previous)
        seed["deduplication"] = {
            "primary_user_normalized": first,
            "trajectory_normalized": user_path,
            "cmrc_document_group": group if is_cmrc else None,
            "material_question_normalized": normalized_content(question) if question is not None else None,
        }
        return None


def source_samples(records, requested, rng, used_groups):
    strata = defaultdict(lambda: defaultdict(list))
    for row in records:
        primary = row["sources"][0]["input"]
        width = row["tokens_without_assistant_content"]
        stratum = (row.get("implicit_operation", row["route_reason"]), width // 100)
        strata[primary][stratum].append(row)
    sources = sorted(strata)
    rng.shuffle(sources)
    groups = {}
    for source, buckets in strata.items():
        keys = sorted(buckets)
        rng.shuffle(keys)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        groups[source] = keys
    selected = []
    while requested and sources:
        active = []
        for source in sources:
            keys = groups[source]
            while keys:
                key = keys.pop(0)
                bucket = strata[source][key]
                row = None
                while bucket and row is None:
                    candidate = bucket.pop()
                    if candidate["source_group"] not in used_groups:
                        row = candidate
                if bucket:
                    keys.append(key)
                if row is not None:
                    selected.append(row)
                    used_groups.add(row["source_group"])
                    requested -= 1
                    break
            if keys:
                active.append(source)
            if not requested:
                break
        sources = active
    return selected


def make_probe(records):
    rng = random.Random(SEED)
    by_category = defaultdict(list)
    for row in records:
        if not row["seen_in_previous_pilot"]:
            by_category[row["category"]].append(row)
    selected, used = {}, set()
    for category, target in PROBE_QUOTAS.items():
        rows = by_category[category]
        if category == "grounded":
            cmrc = [row for row in rows if row["grounded_evidence"]["source_type"] == "cmrc_complete_context"]
            local = [row for row in rows if row["grounded_evidence"]["source_type"] != "cmrc_complete_context"]
            picks = source_samples(cmrc, 10, rng, used)
            selected[category] = picks + source_samples(local, target - len(picks), rng, used)
        elif category == "multi":
            daily = [row for row in rows if row["multi_subtype"] == "daily"]
            document = [row for row in rows if row["multi_subtype"] == "document"]
            picks = source_samples(document, 8, rng, used)
            selected[category] = picks + source_samples(daily, target - len(picks), rng, used)
        else:
            selected[category] = source_samples(rows, target, rng, used)
        if len(selected[category]) < target:
            raise ValueError(f"Structural probe quota {category}: {len(selected[category])} < {target}")
        rng.shuffle(selected[category])
    source_repos = {row["sources"][0]["repo"] for picks in selected.values() for row in picks}
    if len(source_repos) < 4:
        raise ValueError(f"Probe has fewer than four source repositories: {sorted(source_repos)}")
    output, offsets = [], Counter()
    for batch in range(1, 7):
        widths = (6, 4, 4, 2) if batch <= 4 else (4, 4, 4, 4)
        for category, width in zip(PROBE_QUOTAS, widths):
            start = offsets[category]
            offsets[category] += width
            output.extend({**row, "pilot_batch": batch} for row in selected[category][start:start + width])
    return output


def write_rows(path, records):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(tokenizer, protected, *, limit=None):
    intake = Intake(protected)
    records = []
    processed, rejected = Counter(), defaultdict(Counter)
    accepted = defaultdict(Counter)

    def consider(row, name):
        outcome, reason = route_row(row, tokenizer)
        if outcome is not None:
            seed, question = outcome
            reason = intake.add(seed, question)
            if not reason:
                accepted[name][seed["category"]] += 1
                records.append(seed)
        if reason:
            rejected[name][reason] += 1

    for row in json_rows(POOL / "pool.jsonl"):
        name = row["sources"][0]["input"]
        if limit and processed[name] >= limit:
            continue
        processed[name] += 1
        consider(row, name)
    print(f"Processed {sum(processed.values()):,} existing pool rows; routed {len(records):,}.", flush=True)
    for row, reject in honesty_rows(processed):
        if reject:
            rejected["moss_honesty"][reject] += 1
        else:
            consider(row, "moss_honesty")
        if processed["moss_honesty"] % 40_000 == 0:
            print(f"Processed {processed['moss_honesty']:,} honesty rows; routed {len(records):,} total.", flush=True)
        if limit and processed["moss_honesty"] >= limit:
            break
    intake.freeze_groups(records)
    return records, intake, processed, accepted, rejected


def summarize(records, probe, intake, protected, processed, accepted, rejected):
    categories = Counter(row["category"] for row in records)
    probe_counts = Counter(row["category"] for row in probe)
    return {
        "status": "routing_and_structural_probe_only",
        "author_submission_cap_including_previous_and_probe": 25_200,
        "author_submissions_for_this_build": 0,
        "protected_user_only_projection": relative(protected),
        "protected_projection_records": intake.protected_count,
        "processed": dict(processed), "routes_by_primary_source": accepted, "rejections_by_primary_source": rejected,
        "retained_candidate_counts": categories,
        "all_split_goal": {"natural": 7350, "grounded": 5250, "multi": 5250, "instruction": 3150},
        "retained_unique_groups": len({row["source_group"] for row in records}),
        "retained_cross_material_group_unions": len(intake.group_aliases),
        "route_reason_counts": Counter(row["route_reason"] for row in records),
        "multi_subtypes": Counter(row["multi_subtype"] for row in records if row["category"] == "multi"),
        "candidate_cmrc_questions": sum(len(value) for value in intake.cmrc_questions.values()),
        "candidate_cmrc_documents": sum(bool(value) for value in intake.cmrc_questions.values()),
        "maximum_cmrc_questions_per_document": max(map(len, intake.cmrc_questions.values()), default=0),
        "candidate_deficits_before_quality_under_cmrc_caps": {
            "natural": max(0, 7350 - categories["natural"]),
            "grounded": max(0, 5250 - categories["grounded"] + max(0, sum(map(len, intake.cmrc_questions.values())) - 3150)),
            "multi": max(0, 5250 - categories["multi"]),
            "instruction": max(0, 3150 - categories["instruction"]),
        },
        "observed_conflict_edges": len(intake.edges),
        "prior_pilot_flagged_rows": sum(row["seen_in_previous_pilot"] for row in records),
        "probe_seed": SEED, "probe_categories": probe_counts,
        "probe_source_repos": Counter(row["sources"][0]["repo"] for row in probe),
        "probe_primary_inputs": Counter(row["sources"][0]["input"] for row in probe),
        "probe_routes": Counter(row["route_reason"] for row in probe),
        "probe_cmrc_questions": sum(row.get("grounded_evidence", {}).get("source_type") == "cmrc_complete_context" for row in probe),
        "probe_multi_subtypes": Counter(row["multi_subtype"] for row in probe if row["category"] == "multi"),
        "probe_max_original_prefix_tokens": max((row["original_prefix_tokens"] for row in probe), default=None),
        "probe_min_new_assistant_budget": min((row["assistant_content_token_budget"] for row in probe), default=None),
        "known_gaps": [
            "Route reasons and concrete history spans require independent review of the selected prompts and rewritten responses.",
            "Lexical string similarity cannot establish that semantically paraphrased or similarly templated prompts are disjoint.",
            "Only the first two complete exchanges are examined for scale seeds; an otherwise eligible later dependency can be missed.",
            "Protected prompts compare the main user text and complete trajectory; short common follow-ups are not isolated as standalone fuzzy keys.",
            "Raw source aliases remain in each accepted pooled seed. Rejected variants have observed conflict edges, not exhaustive near-neighbor components.",
            "A retained candidate count or a clean 96-seed pilot cannot guarantee the 21,000 final quota or license eligibility.",
        ],
        "format": {"tokenizer": relative(TOKENIZER), "max_context": MAX_TOKENS, "final_eos": False},
    }


class VariantLedger:
    """Keep cross-category options together without multiplying multi-turn capacity."""

    def __init__(self, rows, protected):
        self.protected, self.protected_paths, self.protected_count = load_protected(protected)
        self.parents, self.group_categories, self.original_groups = {}, defaultdict(set), {}
        self.first_users = LexicalIndex()
        self.links = []
        for row in rows:
            group, source = row["source_group"], row["original_source_group"]
            self.original_groups[source] = group
            self.group_categories[group].add(row["category"])
            self.first_users.add(row["deduplication"]["primary_user_normalized"], row["id"], group)

    def root(self, group):
        while group in self.parents:
            parent = self.parents[group]
            if parent in self.parents:
                self.parents[group] = self.parents[parent]
            group = parent
        return group

    def add(self, candidate):
        users, trajectory = user_keys(candidate["original_messages"])
        if not all(users):
            return "empty_normalized_user_key"
        for user in (users[0], *(text for text in users[1:] if len(text) >= 24)):
            if hit := self.protected.match(user):
                return "protected_input_" + hit[0]
        if trajectory in self.protected_paths:
            return "protected_exact_user_trajectory"

        original = candidate["source_group"]
        inherited = self.original_groups.get(original, original)
        linked = {self.root(inherited)}
        visited = set()
        while hit := self.first_users.match(users[0], ignore_groups=visited):
            visited.add(hit[2])
            linked.add(self.root(hit[2]))
        if any("multi" in self.group_categories[group] for group in linked):
            return "existing_multi_variant_or_near_duplicate"

        target = min(linked)
        competitor = set()
        for peer in sorted(linked - {target}):
            competitor.update(self.group_categories[peer])
            self.group_categories[target].update(self.group_categories.pop(peer))
            self.parents[peer] = target
            self.links.append({"candidate_id": candidate["id"], "group": peer, "connected_to": target})
        competitor.update(self.group_categories[target])
        self.group_categories[target].add("multi")
        candidate["original_source_group"] = original
        candidate["source_group"] = target
        candidate["competing_route_categories"] = sorted(competitor - {"multi"})
        candidate["deduplication"] = {
            "primary_user_normalized": users[0], "trajectory_normalized": trajectory,
            "cmrc_document_group": None, "material_question_normalized": None,
        }
        self.first_users.add(users[0], candidate["id"], target)
        return None


def earlier_probe_index():
    prior_groups, prior_queries = set(), LexicalIndex()
    for path in (EXISTING_PILOT, OUTPUT / "probe_seeds_v1.jsonl"):
        for row in json_rows(path):
            group = row["source_group"]
            prior_groups.update((group, row.get("original_source_group", group)))
            users, _ = user_keys(row["original_messages"])
            prior_queries.add(users[0], row["id"], group)
    return prior_groups, prior_queries


def make_implicit_probe(rows, prior_groups, prior_queries, ledger):
    inputs = ("moss_helpfulness", "smoltalk_everyday", "moss_honesty")
    rng = random.Random(SEED + 1)
    protected_groups = {ledger.root(group) for group in prior_groups}
    eligible = defaultdict(list)
    blocked = Counter()
    for row in rows:
        if row["route_reason"] != "implicit_reference_candidate":
            continue
        source = row["sources"][0]["input"]
        first = row["deduplication"]["primary_user_normalized"]
        if row["source_group"] in protected_groups or prior_queries.match(first):
            blocked[source] += 1
        else:
            eligible[source].append(row)
    picked, used_groups = [], set()
    for source in inputs:
        picked.extend(source_samples(eligible[source], 8, rng, used_groups))
    if not all(any(row["sources"][0]["input"] == source for row in picked) for source in inputs):
        raise ValueError("No new implicit route in at least one probe source")
    if len(picked) < 24:
        picked.extend(source_samples([row for source in inputs for row in eligible[source]], 24 - len(picked), rng, used_groups))
    if len(picked) != 24:
        raise ValueError(f"Implicit probe quota 24 has only {len(picked)} new groups")
    rng.shuffle(picked)
    return [{**row, "pilot_batch": index // 8 + 1} for index, row in enumerate(picked)], blocked


def extend_routes(tokenizer, protected, *, limit=None):
    base_rows = []
    base_exclusions, base_reroutes = Counter(), Counter()
    for row in json_rows(OUTPUT / "seed_pool_v1.jsonl"):
        if reason := v2_user_shape(row["original_messages"]):
            base_exclusions[reason] += 1
            continue
        first = row["user_messages"][0]["content"]
        evidence = row.get("grounded_evidence", {})
        if (row["category"] in ("natural", "grounded") and len(row["original_messages"]) == 2
                and V2_EXPLICIT_TRANSFORMATION.search(first) and INSTRUCTION_PAYLOAD.search(first)):
            base_reroutes[row["category"] + "_to_instruction"] += 1
            row["category"] = "instruction"
            row["route_reason"] = "explicit_transformation_with_source_text_v2"
            row["route_variant"] = "v1_base_route_corrected_v2"
            row.pop("grounded_evidence", None)
        elif (row["category"] == "grounded" and len(row["original_messages"]) == 2
                and evidence.get("source_type") in ("untitled_paragraph_then_question", "visible_quote_and_question")
                and V2_DIRECT_RECIPE.search(first.splitlines()[0]) and not USER_MATERIAL_HINT.search(first)):
            base_reroutes["grounded_to_natural_recipe_request"] += 1
            row["category"] = "natural"
            row["route_reason"] = "direct_recipe_suggestion_without_source_material_v2"
            row["route_variant"] = "v1_base_route_corrected_v2"
            row.pop("grounded_evidence", None)
        base_rows.append(row)
    categories_by_id = {row["id"]: row["category"] for row in base_rows}
    tracked = json.loads((SCALE_REPORT / "seed_conflict_edges_v1.json").read_text(encoding="utf-8"))["edges"]
    old_competitors = {
        edge["seed_id"] for edge in tracked if edge["reason"].startswith(("first_user_", "original_or_first_user_"))
        and edge["conflicts_with"] in categories_by_id and categories_by_id[edge["conflicts_with"]] != "multi"
    }
    ledger = VariantLedger(base_rows, protected)
    prior_groups, prior_queries = earlier_probe_index()
    added, processed = [], Counter()
    rejected, selected, operation_counts = defaultdict(Counter), defaultdict(Counter), Counter()

    def consider(row):
        source = row["sources"][0]["input"]
        candidate, reason = None, None
        if reason := v2_user_shape(row["messages"][:4]):
            rejected[source][reason] += 1
            return
        if f"scale-{row['id']}" in old_competitors:
            baseline, _ = route_row(row, tokenizer)
            if baseline and baseline[0]["category"] == "multi":
                candidate = baseline[0]
                candidate["route_variant"] = "recovered_original_multi_route"
                candidate["id"] += "-recovered-v2"
        if candidate is None:
            if dependency(row["messages"][:2], row["messages"][2]):
                rejected[source]["existing_explicit_dependency_route"] += 1
                return
            implicit, reason = implicit_dependency(row["messages"])
            if implicit is not None:
                routed, reason = route_row(row, tokenizer, implicit=implicit)
                if routed is not None:
                    candidate = routed[0]
                    candidate["route_variant"] = "added_implicit_multi_route"
                    candidate["id"] += "-implicit-v2"
        if candidate is None:
            rejected[source][reason] += 1
        elif reason := ledger.add(candidate):
            rejected[source][reason] += 1
        else:
            candidate["alternative_to_v1_id"] = f"scale-{row['id']}" if f"scale-{row['id']}" in categories_by_id else None
            candidate["existing_rewrite_pilot_id"] = None
            candidate["seen_in_previous_pilot"] = False
            selected[source][candidate["route_variant"]] += 1
            if "implicit_operation" in candidate:
                operation_counts[candidate["implicit_operation"]] += 1
            added.append(candidate)

    for row in json_rows(POOL / "pool.jsonl"):
        if len(row["messages"]) >= 4:
            source = row["sources"][0]["input"]
            if limit and processed[source] >= limit:
                continue
            processed[source] += 1
            consider(row)
    print(f"Scanned {sum(processed.values()):,} complete two-exchange rows from v1 pool.", flush=True)
    for row, reject in honesty_rows(processed):
        if not reject and len(row["messages"]) >= 4:
            consider(row)
        if processed["moss_honesty"] % 40_000 == 0:
            print(f"Scanned {processed['moss_honesty']:,} honesty rows; {len(added):,} new route options.", flush=True)
        if limit and processed["moss_honesty"] >= limit:
            break

    new_choice_groups = {ledger.root(row["source_group"]) for row in added if row["competing_route_categories"]}
    prior_group_roots = {ledger.root(group) for group in prior_groups}
    added_ids = {row["id"] for row in added}
    for row in base_rows + added:
        group = ledger.root(row["source_group"])
        row["source_group"] = group
        if row["id"] in added_ids and (group in prior_group_roots
                                      or prior_queries.match(row["deduplication"]["primary_user_normalized"])):
            row["seen_in_previous_pilot"] = True
        row.setdefault("route_variant", "original_v1_route")
        if group in new_choice_groups:
            row["exclusive_route_group"] = group
    stats = {
        "status": "structural_route_candidates_only",
        "v1_pool": "data/natural_sft_v4_scale20k/seed_pool_v1.jsonl",
        "unchanged_v1_probe": "data/natural_sft_v4_scale20k/probe_seeds_v1.jsonl",
        "protected_user_only_projection": relative(protected),
        "protected_projection_records": ledger.protected_count,
        "scanned": dict(processed), "added_by_source": selected, "route_rejections": rejected,
        "implicit_operation_counts": operation_counts,
        "v2_base_exclusions": base_exclusions, "v2_base_reroutes": base_reroutes,
        "v1_rows_transferred": len(base_rows), "new_multi_options": len(added),
        "new_exclusive_route_groups": len(new_choice_groups),
        "observed_connected_group_links": ledger.links,
        "available_by_category_nonadditive": Counter(row["category"] for row in base_rows + added),
        "available_groups_by_category_nonadditive": {
            kind: len({row["source_group"] for row in base_rows + added if row["category"] == kind})
            for kind in PROBE_QUOTAS
        },
        "multi_daily": sum(row.get("multi_subtype") == "daily" for row in base_rows + added),
        "multi_document": sum(row.get("multi_subtype") == "document" for row in base_rows + added),
        "author_submissions": 0,
        "notes": [
            "One final choice from a shared original or conflicting route group. Category and route counts are not additive.",
            "Every added implicit route preserves a complete two-exchange source prefix and source spans. The writer and independent review must reject inapplicable or unrepairable seeds.",
        ],
    }
    return base_rows, added, stats, ledger, prior_groups, prior_queries


def write_v2(tokenizer, protected, *, limit=None):
    pool_path, probe_path = OUTPUT / "seed_pool_v2.jsonl", OUTPUT / "probe_implicit_multi_v2.jsonl"
    report_path = SCALE_REPORT / "seed_inventory_v2.json"
    if not limit:
        for path in (pool_path, probe_path, report_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
    base, added, stats, ledger, prior_groups, prior_queries = extend_routes(tokenizer, protected, limit=limit)
    if limit:
        print(json.dumps({k: stats[k] for k in ("scanned", "added_by_source", "route_rejections")}, ensure_ascii=False))
        return
    probe, blocked = make_implicit_probe(added, prior_groups, prior_queries, ledger)
    for row in probe:
        old = row["original_messages"]
        if old[::2] != row["user_messages"] or fixed_tokens(tokenizer, old) != row["tokens_without_assistant_content"]:
            raise ValueError(f"Probe user or fixed token accounting differs for {row['id']}")
        encoded = encode_conversation(tokenizer, old, final_eos=False)["input_ids"]
        if (len(encoded) != row["original_prefix_tokens"] or len(encoded) > MAX_TOKENS
                or encoded[-1] != SPECIAL_IDS["<|end_turn|>"] or SPECIAL_IDS["<|eos|>"] in encoded):
            raise ValueError(f"Probe prefix lacks the expected no-EXTRA-EOS encoding for {row['id']}")
        for span in row["dependency_anchors"][0]["source_spans"]:
            start, end = span["chars_zero_based_half_open"]
            if old[span["message_index"]]["content"][start:end] != span["text"]:
                raise ValueError(f"Probe anchor does not match the source span for {row['id']}")
    stats.update({
        "probe_seed": SEED + 1, "probe_rows": len(probe),
        "probe_unique_groups": len({row["source_group"] for row in probe}),
        "probe_by_source": Counter(row["sources"][0]["input"] for row in probe),
        "probe_operations": Counter(row["implicit_operation"] for row in probe),
        "probe_previous_64_or_96_overlap": 0, "previous_probe_overlap_exclusions": blocked,
        "probe_minimum_assistant_budget": min(row["assistant_content_token_budget"] for row in probe),
    })
    write_rows(pool_path, base + added)
    write_rows(probe_path, probe)
    write_json(report_path, stats)
    print(f"Wrote {len(base) + len(added):,} route options and {len(probe)} implicit probe seeds.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-inputs", type=Path, default=DEFAULT_PROTECTED)
    parser.add_argument("--quick-limit", type=int, help="Read a bounded sample per raw input and report routes without writing")
    parser.add_argument("--implicit-v2", action="store_true", help="Extend v1 without overwriting the base pool or its probe")
    args = parser.parse_args()
    protected = args.protected_inputs.resolve()
    pool_path, probe_path = OUTPUT / "seed_pool_v1.jsonl", OUTPUT / "probe_seeds_v1.jsonl"
    report_path = SCALE_REPORT / "seed_inventory_v1.json"
    edges_path = SCALE_REPORT / "seed_conflict_edges_v1.json"
    if not args.quick_limit and not args.implicit_v2:
        for path in (pool_path, probe_path, report_path, edges_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_special_ids(tokenizer)
    if args.implicit_v2:
        write_v2(tokenizer, protected, limit=args.quick_limit)
        return
    records, intake, processed, accepted, rejected = build(tokenizer, protected, limit=args.quick_limit)
    if args.quick_limit:
        print(json.dumps({"processed": processed, "routed": accepted, "rejected": rejected}, ensure_ascii=False))
        return
    probe = make_probe(records)
    for row in probe:
        old = row["original_messages"]
        if fixed_tokens(tokenizer, old) != row["tokens_without_assistant_content"]:
            raise ValueError(f"Probe fixed token accounting differs for {row['id']}")
        if len(encode_conversation(tokenizer, old, final_eos=False)["input_ids"]) != row["original_prefix_tokens"]:
            raise ValueError(f"Probe original prefix accounting differs for {row['id']}")
    report = summarize(records, probe, intake, protected, processed, accepted, rejected)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SCALE_REPORT.mkdir(parents=True, exist_ok=True)
    write_rows(pool_path, records)
    write_rows(probe_path, probe)
    write_json(report_path, report)
    write_json(edges_path, {"scope": "Observed conflicts with accepted representatives or protected prompts", "edges": intake.edges})
    print(f"Wrote {len(records):,} routed candidates and {len(probe)} probe seeds.", flush=True)


if __name__ == "__main__":
    main()
