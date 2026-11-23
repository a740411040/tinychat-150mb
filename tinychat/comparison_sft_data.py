"""Build the paired-comparison SFT corpus without changing any Stage 1 input."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
import json
from pathlib import Path
import random
import re
import sqlite3

from tokenizers import Tokenizer

from .chat_format import IGNORE_LABEL, SPECIAL_IDS, check_special_ids, encode_conversation
from .sft_data import MAX_JSON_RECORD, SOURCE_KINDS, conversation_key, parse_conversation, quality_reason, read_json_records
from .stage1_data import (
    HAN, SPLITS, build_lock, eval_prompt_keys, file_identity, json_line, normalized_content,
    sha256, source_file, write_json,
)


SEED = 20261122
MAX_TOKENS = 512
FAMILY_CAP = 1_000
SOURCE = "programmatic/comparison_sft_v2"
CORPUS_ID = "sft_clean_v2"
QUOTAS = {
    "train": {"belle_single": 14_400, "belle_multi": 2_000, "program_single": 2_600, "program_multi": 1_000},
    "validation": {"belle_single": 360, "belle_multi": 60, "program_single": 65, "program_multi": 15},
    "test": {"belle_single": 360, "belle_multi": 60, "program_single": 65, "program_multi": 15},
}

SENSITIVE = re.compile(
    r"诊断|病情|病例|药物|药品|剂量|疫苗|治疗|患者|医疗|癌症|保健品|症状|偏方|健康建议|"
    r"心理疾病|心理健康|急救|医药|怀孕|疾病|疗法|处方|食物中毒|过敏|自杀|自残|"
    r"金融|股票|炒股|股市|基金|投资|理财|贷款|利率|透支|交易所|加密货币|房价|汇率|市场行情|税率|"
    r"法律建议|判刑|诉讼|电击|枪械|炸弹|炸药|武器|黑客|入侵|恶意软件|盗窃|赌博|色情"
)
TIME_SENSITIVE = re.compile(
    r"(?:最新|热门)的?(?:新闻|产品|手机|股票|政策|排名|数据|信息|事件|消息|趋势|版本|研究|公司)|"
    r"(?:最近|今天|明天|明年|目前|当前|当下).{0,9}(?:天气|行情|首发|价格|新闻|政策|排名|汇率)|"
    r"(?:实时|截至).{0,12}(?:数据|指数|新闻|排名)|"
    r"(?:推荐|购买|选购|预算).{0,36}(?:价格|美元|人民币|元以内|元左右)"
)
SYNTHETIC_PROVENANCE = re.compile(r"(?:https?://|www\.|数据来源[:：]|根据.{0,16}(?:调查|研究|数据显示))", re.I)
BOILERPLATE = re.compile(
    r"^(?:当然可以|好的|很高兴).{0,18}(?:为您|帮您|为你|帮你).{0,16}(?:完成|撰写|提供|处理)[。！!]?$|"
    r"作为(?:一个|一名)?(?:AI|人工智能|语言模型)|我(?:无法|不能)(?:访问|联网|查询|生成)|"
    r"请(?:您|你)?(?:稍等|耐心等待)|稍后(?:为您|给您|给你)|"
    r"\[(?:你的|您的|收件人|发件人|公司名称|姓名|日期|此处|插入|省略)[^]\n]{0,22}\]|"
    r"(?:以下略|此处略|略去|此处省略)|(?:Action|Assistant|Human)\s*[:：]|"
    r"(?:不便|不方便|无法|不能).{0,6}(?:全部展示|完整展示|全文展示|完整列出)|"
    r"(?:我已经|我帮你)(?:发送|预订|订购|支付|上传|下载|打电话|打开网页)",
    re.I,
)
UNSUPPORTED = re.compile(
    r"请(?:你|您)?(?:查找|搜索|查询|上网|浏览|访问|发送|上传|下载|打电话|购买|预订|显示图片)|"
    r"(?:上传的|上传一张|附件中|图片中|从图片|如下图|打开网页|网页链接|点击链接)|"
    r"(?:创建|生成|制作).{0,12}(?:头像|图片|音频|主题曲|视频)|"
    r"(?:不少于|至少|大约|至少要|超过|不低于|约)\s*(?:[5-9]\d\d|\d{4,}|[五六七八九]百|[一二两三四五六七八九十]千)\s*[字词]|"
    r"(?:完整代码|写.{0,8}[Cc]ode|编写.{0,20}(?:程序|代码|脚本))"
)
EXACT_FORMAT = re.compile(r"(?:仅|只|不要).{0,12}(?:输出|回答|返回|提供|解释)|\b(?:JSON|XML|YAML|CSV)\b", re.I)
CLOSED_ARITHMETIC = re.compile(
    r"(?:计算|求解|求出|算出).{0,22}(?:[0-9０-９]|算式|面积|概率|平均值|总和)|"
    r"(?:是多少|等于多少|多少[元个米只张斤辆套])|"
    r"质数|奇数|偶数|最大值|最小值|平均值|方程|算式|数学题|求和|数列|"
    r"(?:之间的距离|相距多少|有多远|核算|几公里|多少公里)"
)
COUNT_REQUEST = re.compile(
    r"(?:至少|给出|提供|列出|包括|生成|写出|拟定|提出|想出|包含|讲述|写|用)\s*"
    r"([2-9]|[二两三四五六七八九十])\s*(?:个|条|项|句|种|点|首|字|本)"
)
OUT_OF_CONTEXT = re.compile(
    r"^(?:继续|接着|再来|再说|然后呢|还需要|接下来|能否再|能不能再|"
    r"就这个|上面(?:的|说的)|上述(?:内容|文本)|前面(?:的|说的)|刚才(?:的|说的)|根据之前|"
    r"(?:这个|那个|这些|那些|它|他|她)(?:是|的|们|怎么|可以|如何|有)?[吗呢？?。])"
)
THIN_REPLY = re.compile(r"^(?:好的[，,。!！]?|没问题[，,。!！]?|谢谢[，,。!！]?|不客气[，,。!！]?|是的[，,。!！]?)$")
FOLLOW_UP = re.compile(
    r"上述|前面|上一(?:段|句|条|个)|文章中|文中|刚才|你(?:写|说|提|提供)|"
    r"这(?:段|篇|个|些|样|种|条|一|份)|上面|那(?:段|篇|个|些|样|种|条)|"
    r"再(?:写|举|提供|给|解释|详细)|把.{0,20}(?:改|缩|变|换)|"
    r"(?:修改|增加|删掉|重复|深入|继续|请补充|更具体|更简洁|换成|了解更多)|"
    r"请(?:问|帮)?.{0,10}(?:详细说|举例|展开)|^.{0,6}(?:谢谢|好的|没问题)[，,。]"
)
DEICTIC_LINK = re.compile(
    r"前面|前者|上面|之前|上述|刚才|上一|你(?:写|说|提到|提供)|从(?:文章|文中)|文章中|"
    r"文中|这(?:段|篇|个|些|样|种|条|一|份|两)|那(?:段|篇|个|些|样|种|条)|^.{0,8}继续"
)
FOLLOWUP_INTENTS = (
    ("summarize", r"概括|总结|摘要|小结"),
    ("extract", r"提取|找出|列出.{0,10}(?:要点|信息|句子)|摘出"),
    ("revise", r"修改|重写|改写|缩短|扩写|更简洁|更详细|更正式|换成|改成|调整|润色"),
    ("illustrate", r"举例|例如|例子|举.{0,4}例|例证"),
    ("continue", r"续写|接着写|继续写|下一段|下一章|接下来.?"),
    ("compare", r"对比|有何区别|有何不同|异同|与.{0,7}相比|比较.{0,6}(?:一下|两者|这两|二者|差异|利弊|优缺点)"),
    ("specify", r"详细|具体|进一步|更多|解释|说明|如何|怎样"),
)
SINGLE_INTENTS = (
    ("combine_sentences", r"(?:两|多|以下|下列).{0,10}(?:句|句子).{0,12}(?:组合|合并|连成|组成)|"
                          r"将.{0,12}(?:段落|句子).{0,8}(?:合并|组合)"),
    ("rewrite", r"改写|重写|润色|简化|编辑|更简洁|换一种说法|缩写|缩短"),
    ("proofread", r"错别字|更正|纠正|改正|校对|语法错误|拼写错误"),
    ("summarize", r"摘要|总结|概括|简述|小结"),
    ("extract_keywords", r"提取.{0,12}(?:关键|核心词)|关键词.{0,9}(?:识别|提取)|找.{0,10}关键词"),
    ("extract_entities", r"提取|抽取|找出|识别|筛选|匹配"),
    ("text_emotion", r"情感|情绪|语气|态度|感情色彩|正面.{0,8}负面"),
    ("classify", r"分类|归类|标记|标签|属于哪|分到"),
    ("translation", r"翻译|译成|译为|译文"),
    ("grounded_question", r"根据|基于|从.{0,5}(?:文|段落|材料)[中内]?|已知"),
    ("opposing_views", r"对比|比较|区别|优缺点|利弊|正反两|异同"),
    ("story", r"故事|童话|寓言|传说|情节|剧本|续写"),
    ("poem", r"诗歌|诗句|首诗|歌词|押韵|诗词|诗意|短诗"),
    ("fiction_character", r"角色|人物|主角|主人公|人设"),
    ("dialogue", r"对话|访谈|聊天|讨论"),
    ("correspondence", r"邮件|书信|一封信|感谢信|邀请函|短信|贺卡|通知"),
    ("promotion", r"广告|宣传|文案|商品描述|产品描述|推销|海报"),
    ("slogan", r"口号|标语|宣传语|一句话.{0,10}(?:倡议|号召)|格言"),
    ("naming", r"标题|命名|起.{0,4}名字|取.{0,4}名字|名称"),
    ("brainstorming", r"创意|头脑风暴|点子|设想|假设|构思|想象|构想|虚构"),
    ("outline", r"大纲|提纲|框架|议程|结构|目录"),
    ("exercise_design", r"设计.{0,10}(?:问题|题目|练习|问卷|调查)|拟.{0,8}(?:问题|题目|问卷)"),
    ("event_planning", r"活动|制定.{0,8}计划|时间表|日程|行程|安排|方案|策划"),
    ("short_essay", r"(?:写|创作|生成|撰写|写出|起草).{0,20}(?:一篇|一段|作文|短文|文章|段落|描述|范文)"),
    ("language_example", r"造句|例句|使用.{0,12}(?:单词|成语|短语|词语|词汇)|同义词|反义词"),
    ("practical_steps", r"步骤|制作|准备|布置|整理|技巧|如何.{0,10}(?:清理|写|整理|练习|改进|创建)"),
    ("bounded_advice", r"建议|方法|怎么办|怎样.{0,12}(?:学习|管理|练习|交流)|如何.{0,12}(?:学习|管理|交流)"),
    ("everyday_explanation", r"解释|说明|含义|什么意思|为什么|是什么|原理"),
)
SINGLE_INTENTS = tuple((name, re.compile(pattern, re.I)) for name, pattern in SINGLE_INTENTS)
FOLLOWUP_INTENTS = tuple((name, re.compile(pattern, re.I)) for name, pattern in FOLLOWUP_INTENTS)
GROUNDED_FAMILIES = {"rewrite", "proofread", "summarize", "extract_keywords", "extract_entities",
                     "classify", "translation", "text_emotion", "grounded_question", "combine_sentences"}
GROUND = re.compile(r"(?:[:：]\s*[^\n]{10,}|[\n\r]\s*[^ \n].{8,}|[“「《][^”」》]{9,}[”」》])")
PLACEHOLDER_CONTEXT = re.compile(
    r"^(?:[《“「]?\s*)?(?:一篇|一段)?(?:关于|有关).{3,40}(?:的)?(?:文章|材料|文档|语句|段落)[。！？》”」]?$|"
    r"^(?:包含|包括).{0,28}(?:数据集|列表|文档|材料)[，,].{0,8}(?:可能包括|可能包含)"
)
SPECIFIC = re.compile(r"(?:历史|名人|排名|科学家|研究表明|统计|著名|人口|GDP|\d{4}年|哪一年|是谁)", re.I)
QUOTE = re.compile(r"[《“「]([^》”」]{3,32})[》”」]")
TOPIC_CUE = re.compile(r"(?:关于|有关|围绕|主题为)\s*[“「《]?\s*([^，,。！；\n”」》]{5,34}?)"
                       r"(?:的(?:短文|故事|文章|小说|诗歌|对话|文案|寓言|散文|演讲|段落)|[”」》])")
LEADERS = re.compile(r"^(?:(?:你能|能否|能不能|请问|麻烦|请|帮我|给我|你好)[，,：: ]*)+")
ORDINAL = re.compile(r"(?m)^\s*(?:\d{1,2}[.、)）]|[一二三四五六七八九十][、.])\s*")
DANGLING_LIST = re.compile(r"(?:^|\n)[ \t]*(?:[-*]|\d{1,2}[.、)）]|[一二三四五六七八九十][、.])[ \t]*$")
REQUESTED_LENGTH = re.compile(r"(?:(至少|不低于|不少于|不超过|最多)\s*)?([1-9]\d{1,3})\s*字")
NUMBER = re.compile(r"(?<![A-Za-z\d])\d{2,}(?![A-Za-z\d])")
POEM_LINES = re.compile(r"([2-9二两三四五六七八九十])\s*行(?:诗|的诗)")
POEM_WIDTH = re.compile(r"每行(?:只有|只写|仅有|应有|包含|有|为)?\s*([2-9二两三四五六七八九十])\s*个?字")
EXTRACT_ACTION = re.compile(r"提取|抽取|摘取|摘录|摘出|抽出|引用原文")
SUMMARY_ACTION = re.compile(r"概括|摘要|小结|总结")
EXHAUSTIVE_ACTION = re.compile(r"所有|全部|逐一|每(?:个|项|条)")
SUMMARY_INTENT_CLAIM = re.compile(r"打算|决心|决定|承诺|计划|表示愿意|愿意|打定主意|准备")
SUMMARY_SCOPE_CLAIM = re.compile(r"(?:详细|全面|系统|逐一)(?:地)?(?:介绍|分析|论述|比较|列举)")
ARM_ROLES = re.compile(r"(?m)^(?:用户|助手|机器人|人类)\s*[:：]")
ENDINGS = frozenset("。？！?!.…；;”’」』）)]】》~～：:\n0123456789")
CHINESE_DIGITS = dict(zip("二两三四五六七八九十", (2, 2, 3, 4, 5, 6, 7, 8, 9, 10)))


@dataclass
class Candidate:
    row: dict
    family: str
    group: tuple[str, ...]
    kind: str


def rank(value, salt):
    return sha256(f"{SEED}:{salt}:{value}".encode("utf-8"))


def prompts(messages):
    return tuple(normalized_content(item["content"]) for item in messages if item["role"] == "user")


@lru_cache(maxsize=16_384)
def grams(text):
    return frozenset(text[i:i + 5] for i in range(len(text) - 4))


class ConflictIndex:
    """Exact matches, symmetric containment, and exact five-gram Jaccard threshold queries."""

    def __init__(self, containment=False, minimum=12, substring_minimum=24):
        self.exact = set()
        self.postings = defaultdict(list)
        self.anchor = defaultdict(list)
        self.containment = containment
        self.minimum = minimum
        self.substring_minimum = substring_minimum
        self.anchor_length = min(12, substring_minimum)
        self.posting_minimum = min(minimum, substring_minimum) if containment else minimum

    def add(self, key):
        if not key or key in self.exact:
            return
        self.exact.add(key)
        features = grams(key) if len(key) >= self.posting_minimum else frozenset()
        for feature in features:
            self.postings[feature].append(key)
        if self.containment and len(key) >= self.substring_minimum:
            self.anchor[key[:self.anchor_length]].append(key)

    def match(self, key):
        if key in self.exact:
            return "exact"
        if self.containment and len(key) >= self.substring_minimum:
            for offset in range(len(key) - self.anchor_length + 1):
                if any(other in key for other in self.anchor.get(key[offset:offset + self.anchor_length], ())):
                    return "substring"
            rarest = min(grams(key), key=lambda feature: len(self.postings.get(feature, ())))
            if any(key in other for other in self.postings.get(rarest, ())):
                return "substring"
        if len(key) < self.minimum:
            return None
        features = grams(key)
        if not features:
            return None
        # At J >= 4/5, at least one of the rarest floor(|query|/5)+1 features must intersect.
        required = len(features) // 5 + 1
        candidates = set()
        rarest = sorted(features, key=lambda feature: (len(self.postings.get(feature, ())), feature))[:required]
        for feature in rarest:
            candidates.update(self.postings.get(feature, ()))
        for other in candidates:
            if len(other) < self.minimum:
                continue
            other_features = grams(other)
            if 5 * min(len(features), len(other_features)) < 4 * max(len(features), len(other_features)):
                continue
            intersection = len(features.intersection(other_features))
            if 9 * intersection >= 4 * (len(features) + len(other_features)):
                return "jaccard"
        return None


def identify(patterns, text):
    return next((name for name, pattern in patterns if pattern.search(text)), None)


def final_punctuation_ok(text):
    if DANGLING_LIST.search(text):
        return False
    if len(text) <= 105:
        return not text.endswith(("如下", "例如", "首先", "包含", "包括", "如下所示", "：", ":"))
    return text[-1] in ENDINGS and not text.endswith(("：", ":", "…", "..."))


def check_count(user, answer):
    if lines := POEM_LINES.search(user):
        written = lines.group(1)
        wanted = int(written) if written.isdecimal() else CHINESE_DIGITS[written]
        observed = [part for part in re.split("[。？！?!，,；;\n]+", answer) if HAN.search(part)]
        if len(observed) != wanted:
            return False
        if width := POEM_WIDTH.search(user):
            written = width.group(1)
            wanted = int(written) if written.isdecimal() else CHINESE_DIGITS[written]
            if any(len(HAN.findall(part)) != wanted for part in observed):
                return False
    if match := REQUESTED_LENGTH.search(user[:100]):
        qualifier, amount = match.groups()
        amount = int(amount)
        length = len(HAN.findall(answer))
        if qualifier in {"至少", "不少于", "不低于"}:
            if length < amount:
                return False
        elif qualifier in {"最多", "不超过"}:
            if length > amount:
                return False
        elif not amount * 0.75 <= length <= amount * 1.25:
            return False
    match = COUNT_REQUEST.search(user)
    if not match:
        return True
    written = match.group(1)
    target = int(written) if written.isdecimal() else CHINESE_DIGITS[written]
    unit = user[match.end() - 1]
    if unit == "字":
        return len(HAN.findall(answer)) >= target
    if unit == "句":
        return len([part for part in re.split("[。？！?!\n]+", answer) if len(normalized_content(part)) >= 3]) >= target
    return len(ORDINAL.findall(answer)) >= target


def grounded_answer_reason(query, answer, source):
    extract = EXTRACT_ACTION.search(query)
    summary = SUMMARY_ACTION.search(query)
    if not extract and not summary:
        return None
    if extract and EXHAUSTIVE_ACTION.search(query):
        return "unverified_exhaustive_extraction"
    source_text = normalized_content(source)
    source_features = {source_text[i:i + 3] for i in range(len(source_text) - 2)}
    minimum_overlap = 0.28 if extract else 0.14
    for piece in re.split(r"[。？！!?\n；;]+", answer):
        text = normalized_content(piece)
        if len(text) < 18:
            continue
        features = {text[i:i + 3] for i in range(len(text) - 2)}
        if len(features & source_features) < minimum_overlap * len(features):
            return "weakly_grounded_extraction" if extract else "weakly_grounded_summary"
    if summary and not extract:
        if SUMMARY_INTENT_CLAIM.search(answer) and not SUMMARY_INTENT_CLAIM.search(source):
            return "unverified_summary_motive"
        if SUMMARY_SCOPE_CLAIM.search(answer) and not SUMMARY_SCOPE_CLAIM.search(source):
            return "unverified_summary_scope"
    return None


def belle_quality(messages):
    reason = quality_reason(messages)
    if reason:
        return reason, None
    if not 2 <= len(messages) <= 8:
        return "full_history_length", None
    joined = "\n".join(item["content"] for item in messages)
    if len(joined) > 1450:
        return "pretokenization_length", None
    if SENSITIVE.search(joined) or TIME_SENSITIVE.search(joined):
        return "sensitive_or_time_sensitive", None
    if ARM_ROLES.search(joined):
        return "embedded_role", None
    initial = messages[0]["content"]
    instruction, _, context = initial.partition("\n\n")
    if not context and "\n" in instruction:
        first_line, _, rest = instruction.partition("\n")
        if re.search(r"^(?:根据|基于|给定|给出|输入|以下|将|对|从|把|请(?:根据|给定|给出|提取|总结|将|对))", first_line):
            instruction, context = first_line, rest
    primary = identify(SINGLE_INTENTS, instruction)
    if OUT_OF_CONTEXT.search(LEADERS.sub("", initial)) or (primary in GROUNDED_FAMILIES and not GROUND.search(initial)):
        return "missing_initial_context", None
    context = context.strip()
    context_body = context
    if context:
        context_body = re.sub(r"^(?:文本|内容|原文|材料|文章|段落)\s*[:：]\s*", "", context)
        if primary in GROUNDED_FAMILIES and PLACEHOLDER_CONTEXT.search(context_body):
            return "placeholder_context", None
        if (primary == "text_emotion" and re.search("导演|主演|演员表", context_body)
                and not re.search("评论|评价|好评|差评|观后感", context_body)):
            return "missing_sentiment_evidence", None
        if (primary == "fiction_character" and re.search(r"(?:和|与)人物有关.{0,16}(?:问题|陈述)", instruction)
                and re.search(r"行星|卫星|小行星|太阳系", context_body)
                and not re.search(r"人物|姓名|先生|女士|出生|逝世|[他她](?:曾|的|在)", context_body)):
            return "incompatible_context_subject", None
        if primary in {"rewrite", "proofread", "summarize", "extract_entities", "extract_keywords", "classify"}:
            initial_numbers = set(NUMBER.findall(context_body))
            if set(NUMBER.findall(messages[1]["content"])) - initial_numbers:
                return "ungrounded_numeric_claim", None
    history = []
    for user, assistant in zip(messages[::2], messages[1::2]):
        query, answer = user["content"], assistant["content"]
        query_key = normalized_content(query)
        if len(query_key) < 5 or THIN_REPLY.fullmatch(answer):
            return "thin_turn", None
        if len(query_key) >= 12 and normalized_content(answer).startswith(query_key):
            return "echoed_request", None
        if (re.search(r"推荐|选购|购买建议", query)
                and re.search(r"(?:预算|价格).{0,16}(?:美元|人民币|元)", query)):
            return "unverified_current_price", None
        if re.search(r"术语|词汇表|术语表", query) and re.search(r"转换|替换|改成|改为", query):
            return "unverified_glossary_transformation", None
        if re.search(r"同义词|近义词|反义词", query) and len(ORDINAL.findall(answer)) > 5:
            return "unverified_lexical_list", None
        if UNSUPPORTED.search(query) or EXACT_FORMAT.search(query) or CLOSED_ARITHMETIC.search(query):
            return "unverified_constraint_or_external_action", None
        if BOILERPLATE.search(answer) or SYNTHETIC_PROVENANCE.search(answer):
            return "boilerplate_or_unverified_provenance", None
        if not final_punctuation_ok(answer) or answer.count("\x60\x60\x60") % 2:
            return "incomplete_assistant", None
        if not check_count(query, answer):
            return "unmet_requested_count", None
        if re.search(r"(?:全英文|用英[语文]|英文作答|英语作答|写.{0,12}英语)", query) and len(HAN.findall(answer)) > 8:
            return "output_language_mismatch", None
        if (context and user is messages[0] and primary == "extract_entities"
                and re.search(r"提取原文|原样摘录|摘出原句|找出.{0,12}相关的句子", instruction)):
            extracted = normalized_content(answer)
            if extracted and extracted not in normalized_content(context):
                return "failed_literal_extraction", None
        if history and re.search(r"(?:提取|抽取|摘录|摘出).{0,24}(?:原文|原话|原句|句子|段落)", query):
            source = grams(normalized_content("\n".join(history)))
            extracted = grams(normalized_content(answer))
            if extracted and len(source & extracted) * 2 < len(extracted):
                return "failed_literal_extraction", None
        if history and not DEICTIC_LINK.search(query):
            terms = grams(normalized_content(query))
            source = grams(normalized_content("\n".join(history)))
            if len(terms & source) < 2:
                return "unlinked_followup", None
        source_text = "\n".join(history) if history else context_body or initial
        relevant_query = query if history else instruction
        if reason := grounded_answer_reason(relevant_query, answer, source_text):
            return reason, None
        history.extend((query, answer))
    if primary is None:
        return "unclassified_intent", None
    if primary in {"everyday_explanation", "bounded_advice", "practical_steps", "short_essay"} and SPECIFIC.search(initial):
        return "unverified_specific_facts", None
    if len(messages) == 2:
        return None, f"belle:{primary}"
    followups = [item["content"] for item in messages[2::2]]
    if not all(FOLLOW_UP.search(text) for text in followups):
        return "unlinked_followup", None
    follow_family = identify(FOLLOWUP_INTENTS, followups[-1])
    if follow_family is None:
        return "unclassified_followup", None
    return None, f"belle:multi_{follow_family}"


def narrow_groups(messages):
    first = messages[0]["content"].split("\n", 1)[0]
    first = LEADERS.sub("", first)
    topic = normalized_content(first)
    shape = re.sub(r"[《“「][^》”」]{2,44}[》”」]|\d+(?:[.\-]\d+)*", "{值}", first)
    shape = normalized_content(shape)
    group = [f"topic:{topic}"] if len(topic) >= 8 else []
    if shape != topic and len(shape) >= 8:
        group.append(f"template:{shape}")
    if cue := TOPIC_CUE.search(first):
        group.append(f"explicit_topic:{normalized_content(cue.group(1))}")
    for entity in sorted(set(QUOTE.findall(first))):
        clean = normalized_content(entity)
        if len(clean) >= 4:
            group.append(f"entity:{clean}")
    if not group:
        group.append(f"exchange:{conversation_key(messages[:2])}")
    return tuple(group)


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        yield from (json.loads(line) for line in source if line.strip())


def acquire(output, old_data, limits, local_root=None):
    sources = json.loads((old_data / "manifest.json").read_text(encoding="utf-8"))["sources"]
    output.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(output / "candidates.sqlite3")) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                repo TEXT, revision TEXT, ordinal INTEGER, family TEXT, payload TEXT,
                PRIMARY KEY (repo, revision, ordinal)
            );
            CREATE TABLE IF NOT EXISTS progress (
                repo TEXT, revision TEXT, rows INTEGER, counts TEXT, PRIMARY KEY (repo, revision)
            );
        """)
        for source in sources:
            repo, revision = source["repo"], source["revision"]
            kind = SOURCE_KINDS[repo]
            entry, = source["files"]
            limit = limits[kind]
            done, raw_counts = db.execute("SELECT rows, counts FROM progress WHERE repo=? AND revision=?",
                                          (repo, revision)).fetchone() or (0, "{}")
            if done >= limit:
                continue
            counts = Counter(json.loads(raw_counts))
            print(f"[comparison SFT] {kind}: source rows {done:,} through {limit:,}", flush=True)
            with source_file(source, entry, local_root=local_root) as raw, closing(read_json_records(raw)) as records:
                end = done
                for ordinal, value in enumerate(islice(records, done, limit), start=done + 1):
                    end = ordinal
                    messages = parse_conversation(value, kind)
                    reason, family = belle_quality(messages) if messages else ("invalid_conversation", None)
                    if reason:
                        counts[reason] += 1
                    else:
                        db.execute("INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                                   (repo, revision, ordinal, family, json_line(messages)))
                        counts["accepted"] += 1
                    if ordinal % 256 == 0:
                        db.execute("INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?)",
                                   (repo, revision, ordinal, json.dumps(counts)))
                        db.commit()
                    if ordinal % 8192 == 0:
                        print(f"[comparison SFT] {kind}: {ordinal:,} scanned; {counts['accepted']:,} retained", flush=True)
                db.execute("INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?)",
                           (repo, revision, end, json.dumps(counts)))
                db.commit()
            print(f"[comparison SFT] {kind}: {end:,} scanned; {counts['accepted']:,} retained", flush=True)


def acquire_multi_spans(output, old_data, offsets, limit, local_root=None):
    """Sample pinned JSONL at bounded byte offsets to avoid an early-template block."""
    sources = json.loads((old_data / "manifest.json").read_text(encoding="utf-8"))["sources"]
    source, = [entry for entry in sources if SOURCE_KINDS[entry["repo"]] == "multi"]
    repo, revision = source["repo"], source["revision"]
    entry, = source["files"]
    output.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(output / "candidates_spans.sqlite3")) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                repo TEXT, revision TEXT, ordinal INTEGER, family TEXT, payload TEXT,
                PRIMARY KEY (repo, revision, ordinal)
            );
            CREATE TABLE IF NOT EXISTS spans (
                repo TEXT, revision TEXT, start_byte INTEGER, next_byte INTEGER, rows INTEGER, counts TEXT,
                PRIMARY KEY (repo, revision, start_byte)
            );
        """)
        with source_file(source, entry, local_root=local_root) as stream:
            for offset in offsets:
                if not 0 < offset < entry["bytes"]:
                    raise ValueError(f"Invalid multiturn source byte offset: {offset}")
                curs = db.execute("SELECT next_byte, rows, counts FROM spans WHERE repo=? AND revision=? AND start_byte=?",
                                  (repo, revision, offset)).fetchone()
                start, done, reasons = curs or (offset, 0, "{}")
                if done >= limit:
                    continue
                print(f"[comparison SFT] multi span {offset:,}: {done:,} to {limit:,} rows", flush=True)
                stream.seek(start)
                if curs is None:
                    if len(stream.readline()) > MAX_JSON_RECORD:
                        raise ValueError(f"Oversized partial row at byte {offset}")
                counts = Counter(json.loads(reasons))
                for count in range(done + 1, limit + 1):
                    position = stream.tell()
                    raw = stream.readline()
                    if not raw or len(raw) > MAX_JSON_RECORD:
                        raise ValueError(f"Unexpected EOF or oversized row at byte {position}")
                    messages = parse_conversation(json.loads(raw), "multi")
                    reason, family = belle_quality(messages) if messages else ("invalid_conversation", None)
                    if reason:
                        counts[reason] += 1
                    else:
                        db.execute("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?)",
                                   (repo, revision, position, family, json_line(messages)))
                        counts["accepted"] += 1
                    if count % 256 == 0 or count == limit:
                        db.execute("INSERT OR REPLACE INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                                   (repo, revision, offset, stream.tell(), count, json.dumps(counts)))
                        db.commit()
                print(f"[comparison SFT] multi span {offset:,}: {limit:,} scanned; {counts['accepted']:,} retained",
                      flush=True)


PROGRAM_QUOTAS = {
    "train": {
        "difference": 400, "equal_share": 400, "record_lookup": 450, "key_value_pairs": 450,
        "ordered_filter": 450, "boolean_field": 450, "slot_revision": 500, "clarify_item": 500,
    },
    "validation": {"minimum_record": 25, "swap_fields": 20, "inclusive_count": 20, "record_correction": 15},
    "test": {"two_item_bill": 25, "join_rooms": 20, "strict_comparison": 20, "record_confirmation": 15},
}
PROGRAM_MULTI = {"slot_revision", "clarify_item", "filter_then_select", "record_correction", "record_confirmation"}
PROGRAM_SPLIT = {family: split for split, families in PROGRAM_QUOTAS.items() for family in families}
PROGRAM_SPLIT["filter_then_select"] = "train"
PROGRAM_ENTITIES = {
    "train": (
        ("挂钟", "竹篓", "画框", "手账", "花盆", "拼图", "围巾", "木夹", "纸扇", "杯垫", "笔袋", "布包",
         "棋盘", "收纳盒", "长尾夹", "陶罐", "粉笔", "软尺", "水壶", "蒲扇", "文件盒", "样品瓶", "便签本"),
        ("东馆", "西馆", "北馆", "南馆", "石桥库", "白墙库", "山脚仓", "斜坡仓",
         "银杏柜", "青砖柜", "玻璃柜", "推拉柜", "备品间", "工具间", "阅览室", "木工房"),
    ),
    "validation": (
        ("磁贴", "量角器", "纸盒", "鲤鱼灯", "风铃", "水杯", "棉线", "量杯"),
        ("风筝室", "松木棚", "雨花廊", "百叶窗台", "彩绘门厅", "曲廊边", "水榭"),
    ),
    "test": (
        ("卡纸", "橡皮", "键盘", "毛线球", "橘灯", "棋钟", "卷纸", "陶笛"),
        ("香草园", "封样间", "日晒台", "楼梯口", "竹影厅", "花架前", "后庭"),
    ),
}
QUALITIES = ("有提手", "有隔板", "带盖", "带标签", "可折叠", "已贴签", "已清点", "可叠放")
ARITHMETIC_STRUCTURES = {
    "difference": "a-b",
    "equal_share": "a/b (exact division)",
    "minimum_record": "argmin(a,b,c)",
    "inclusive_count": "count(low<=x<=high)",
    "two_item_bill": "a*b+c*d",
    "strict_comparison": "a<b",
}


def comparison_quotas(belle_multi):
    quotas = {split: dict(values) for split, values in QUOTAS.items()}
    program = {split: dict(values) for split, values in PROGRAM_QUOTAS.items()}
    change = QUOTAS["train"]["belle_multi"] - belle_multi
    if not 0 <= change <= 1_000:
        raise ValueError("BELLE multi-turn quota must be between 1,000 and 2,000")
    quotas["train"]["belle_multi"] -= change
    quotas["train"]["belle_single"] += change
    quotas["train"]["program_multi"] += change
    quotas["train"]["program_single"] -= change
    share, rem = divmod(change, 4)
    for index, family in enumerate(("boolean_field", "ordered_filter", "key_value_pairs", "record_lookup")):
        program["train"][family] -= share + (index < rem)
    extra_family_count = min(change, 500)
    if extra_family_count:
        program["train"]["filter_then_select"] = extra_family_count
    remaining = change - extra_family_count
    program["train"]["slot_revision"] += remaining // 2
    program["train"]["clarify_item"] += remaining - remaining // 2
    return quotas, program


def exchange(user, assistant):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": str(assistant)}]


def render_program(family, index):
    """Reconstruct a complete deterministic task, with the answer computed from its input fields."""
    objects, locations = PROGRAM_ENTITIES[PROGRAM_SPLIT[family]]
    rng = random.Random(f"{SEED}:{family}:{index}")
    x, y, z, w = rng.sample(range(6, 98), 4)
    item, other, third = rng.sample(objects, 3)
    room, dest, spare = rng.sample(locations, 3)
    tag = rng.choice(QUALITIES)
    if family == "difference":
        large, small = sorted((x, y), reverse=True)
        request = rng.choice([
            f"清点时有{large}件{item}，借走{small}件，还剩几件？",
            f"{room}最初有{large}件{item}。拿出{small}件后，库里还剩多少件？",
            f"盘点记录为{large}件{item}，实际只有{small}件。少了多少件？",
            f"{room}有{large}件{item}，{dest}有{small}件。前者比后者多多少件？",
        ])
        return exchange(request + "只写数字。", large - small)
    if family == "equal_share":
        groups = 2 + z % 11
        quotient = 2 + x % 35
        amount = groups * quotient
        request = rng.choice([
            f"{amount}件{item}平均装进{groups}个盒子，每盒几件？",
            f"把{amount}件{item}等量分给{groups}个小组，每组分到几件？",
            f"{room}将{amount}件{item}均分为{groups}份，一份有几件？",
            f"{groups}个展位平均分配{amount}件{item}，每个展位放几件？",
        ])
        return exchange(request + "只答数字。", amount // groups)
    if family == "record_lookup":
        table = dict(zip((item, other, third), (room, dest, spare)))
        target = rng.choice(list(table))
        order = list(table)
        rng.shuffle(order)
        mapping = rng.choice(["；".join(f"{name}放在{table[name]}" for name in order),
                              "、".join(f"{name}→{table[name]}" for name in order),
                              "\n".join(f"{name}：{table[name]}" for name in order)])
        ask = rng.choice([f"{target}放在哪里？", f"找{target}应去哪里？", f"查一下{target}对应的位置。"])
        return exchange(f"{mapping}。\n{ask}只回答地点。", table[target])
    if family == "key_value_pairs":
        table = dict(zip((item, other, third), (x, y, z)))
        order = list(table)
        rng.shuffle(order)
        instruction = rng.choice(["转成「物品=件数」的写法", "每项写作「物品=件数」", "用「物品=件数」逐项重写"])
        context = "，".join(f"{name}有{table[name]}件" for name in order)
        answer = "；".join(f"{name}={table[name]}" for name in order)
        return exchange(f"{context}。{instruction}，保持原顺序，以全角分号分隔。只给转换结果。", answer)
    if family == "ordered_filter":
        table = dict(zip((item, other, third), (x, y, z)))
        limit = sorted(table.values())[rng.choice((0, 1))] + 1
        data = "，".join(f"{name}{count}件" for name, count in table.items())
        instruction = rng.choice(["按记录顺序写出达到门槛的物品名", "按原顺序列出符合条件的物品",
                                  "筛出满足件数要求的物品，并保留相对顺序"])
        answer = "、".join(name for name, count in table.items() if count >= limit)
        return exchange(f"本次记录：{data}。门槛是至少{limit}件。{instruction}，只用顿号分隔。", answer)
    if family == "boolean_field":
        active = {item, third}
        data = "、".join(sorted(active))
        target = rng.choice((item, other, third))
        question = rng.choice([f"{target}是否{tag}？", f"按记录判断：{target}{tag}吗？",
                               f"检查{target}是不是{tag}。"])
        return exchange(f"{room}登记了{item}、{other}、{third}，其中只有{data}{tag}。{question}只答是或否。",
                        "是" if target in active else "否")
    if family == "slot_revision":
        context = rng.choice([f"登记{item}在{room}，有{x}件。", f"{room}存放{x}件{item}。", f"这批{item}共{x}件，放在{room}。"])
        pattern = rng.choice(["地点|数量", "数量|地点"])
        answer = f"{room}|{x}" if pattern == "地点|数量" else f"{x}|{room}"
        if z % 3 == 0:
            change = f"更正：改放到{dest}，数量不变。"
            new_place, new_count = dest, x
        elif z % 3 == 1:
            change = f"刚复点是{y}件，仍放在原处。"
            new_place, new_count = room, y
        else:
            change = f"将地点改为{dest}、数量改为{y}。"
            new_place, new_count = dest, y
        updated = f"{new_place}|{new_count}" if pattern == "地点|数量" else f"{new_count}|{new_place}"
        return (exchange(f"{context}按{pattern}输出，数量只写数字。", answer)
                + exchange(f"{change}仍按刚才的格式输出。", updated))
    if family == "clarify_item":
        table = {item: x, other: y}
        target = rng.choice(list(table))
        setup = rng.choice([
            f"登记册上有{item}{x}件、{other}{y}件。帮我看一下数量。没说哪件时，先问清楚。",
            f"{room}清点过{item}（{x}件）和{other}（{y}件）。我想查件数，如果对象不明确请先确认。",
            f"已知{item}{x}件，{other}{y}件。请告诉我件数。先问我是查哪一种。",
        ])
        clarify = rng.choice([f"你想查{item}还是{other}的件数？", f"想查哪一种，{item}还是{other}？"])
        reply = rng.choice([f"查{target}的数量。只答数字。", f"我问的是{target}，请只给件数。"])
        return exchange(setup, clarify) + exchange(reply, table[target])
    if family == "filter_then_select":
        counts = dict(zip((item, other, third), (x, y, z)))
        confirmed = set(rng.sample(tuple(counts), 2))
        eligible = {name: count for name, count in counts.items() if name in confirmed}
        record = "；".join(f"{name}={count}件（{'已复核' if name in confirmed else '待复核'}）"
                          for name, count in counts.items())
        direction = rng.choice(("最少", "最多"))
        target = min(eligible, key=eligible.get) if direction == "最少" else max(eligible, key=eligible.get)
        initial = f"清单：{record}。只选出已复核的物品，保留记录顺序，名称用顿号隔开。"
        followup = f"从刚才选出的名单里，找件数{direction}的一项。按「名称=件数」输出，件数只写数字。"
        return exchange(initial, "、".join(eligible)) + exchange(followup, f"{target}={eligible[target]}")
    if family == "minimum_record":
        table = dict(zip((room, dest, spare), (x, y, z)))
        data = "；".join(f"{name}{minutes}分钟" for name, minutes in table.items())
        return exchange(f"到各处的步行时长为{data}。只写用时最短的地点。", min(table, key=table.get))
    if family == "swap_fields":
        return exchange(f"现有位置标签写成「{room}／{item}」。调换斜杠两侧的字段，保留全角斜杠，只写更新结果。",
                        f"{item}／{room}")
    if family == "inclusive_count":
        seq = (x, y, z, w)
        low, high = sorted(rng.sample(range(4, 100), 2))
        return exchange(f"记录为{'、'.join(map(str, seq))}。有几项落在{low}到{high}（含边界）之间？只答数量。",
                        sum(low <= value <= high for value in seq))
    if family == "record_correction":
        return (exchange(f"标签记录为物品「{item}」、位置「{room}」。请给我物品名称。", item)
                + exchange(f"名称改成「{other}」，地点沿用。用「地点/物品」输出最后记录。", f"{room}/{other}"))
    if family == "two_item_bill":
        subtotal = x * (z % 7 + 1) + y * (w % 7 + 1)
        return exchange(f"{item}每件{x}元买{z % 7 + 1}件，{other}每件{y}元买{w % 7 + 1}件。应付多少元？只答数字。",
                        subtotal)
    if family == "join_rooms":
        record = {item: room, other: dest, third: spare}
        order = (third, item, other)
        return exchange(f"货架记录：[ {room}：{item}；{dest}：{other}；{spare}：{third} ]。依次查{third}、{item}、{other}。只返回地点，用箭头 → 连接。",
                        " → ".join(record[name] for name in order))
    if family == "strict_comparison":
        rhs = x if w % 4 == 0 else y
        return exchange(f"样本甲的数据为{x}，样本乙为{rhs}。若甲的数值严格小于乙，输出「左」；否则输出「右」。",
                        "左" if x < rhs else "右")
    if family == "record_confirmation":
        return (exchange(f"看到暂存卡片：编号{x}在{room}，编号{y}在{dest}。编号{y}在哪里？只写地点。", dest)
                + exchange(f"我想核对一下，是编号{y}在{room}吗？若错，请给正确信息，采用「编号在地点」格式。",
                           f"{y}在{dest}"))
    raise ValueError(f"Unsupported program family: {family}")


def program_candidates(family):
    seen = set()
    for index in range(100_000):
        messages = render_program(family, index)
        key = conversation_key(messages)
        if key in seen:
            continue
        seen.add(key)
        origin = {"family": family, "index": index}
        kind = "program_multi" if family in PROGRAM_MULTI else "program_single"
        row = {"id": rank(f"{family}:{index}", "program"), "messages": messages, "source": SOURCE,
               "program_case": origin}
        yield Candidate(row, f"program:{family}", (f"program_family:{family}",), kind)


def candidate_keys(messages):
    parts = prompts(messages)
    if len(parts) == 1:
        return parts
    standalone = (key for key, turn in zip(parts[1:], messages[2::2], strict=True)
                  if len(key) >= 24 and not DEICTIC_LINK.search(turn["content"]))
    return (parts[0], "|".join(parts), *standalone)


def external_index(old_data, eval_paths):
    evaluation = ConflictIndex(containment=True, substring_minimum=6)
    old_sft = ConflictIndex(containment=True)
    identities = []
    for path in eval_paths:
        keys, identity = eval_prompt_keys(path)
        identity["substring_min_chars"] = evaluation.substring_minimum
        identities.append(identity)
        for key in sorted(keys):
            evaluation.add(key)
    for split in ("validation", "test"):
        path = old_data / f"{split}.jsonl"
        rows = 0
        for row in read_jsonl(path):
            for key in prompts(row["messages"]):
                old_sft.add(key)
            rows += 1
        if not rows:
            raise ValueError(f"Empty historic heldout split: {path}")
        identities.append({"path": str(path.resolve()), **file_identity(path), "rows": rows,
                           "substring_min_chars": old_sft.substring_minimum})
    return evaluation, old_sft, identities


def pt_chunks(text):
    """Index meaningful document pieces without treating a short common sentence as leakage."""
    complete = normalized_content(text)
    if 40 <= len(complete) <= 384:
        yield complete
    units = [part for x in re.split(r"[\n。！？!?]+", text) if (part := normalized_content(x))]
    for i, part in enumerate(units):
        if 40 <= len(part) <= 384:
            yield part
        if i + 1 < len(units):
            pair = part + units[i + 1]
            if 40 <= len(pair) <= 384:
                yield pair
    if len(complete) > 384:
        for start in range(0, len(complete), 96):
            piece = complete[start:start + 192]
            if len(piece) >= 40:
                yield piece


def pt_index(base):
    index = ConflictIndex(containment=True, minimum=40, substring_minimum=40)
    identities = []
    for split in ("validation", "test"):
        path = base / f"{split}.jsonl"
        rows = 0
        initial = len(index.exact)
        for row in read_jsonl(path):
            for key in pt_chunks(row["text"]):
                index.add(key)
            rows += 1
        if not rows:
            raise ValueError(f"Empty historic pretraining heldout split: {path}")
        identities.append({"path": str(path.resolve()), **file_identity(path), "rows": rows,
                           "new_normalized_chunks": len(index.exact) - initial, "min_chars": 40})
    return index, identities


def belle_candidates(old_data, output, acquisition):
    for row in read_jsonl(old_data / "train.jsonl"):
        messages = row["messages"]
        # Stage 1 could drop earlier turns to form a window. Use raw source rows for every multi-turn example.
        if len(messages) != 2:
            continue
        reason, family = belle_quality(messages)
        if reason is None:
            record = {"id": row["id"], "messages": messages, "source": row["source"],
                      "source_record": {"old_train_id": row["id"]}}
            yield Candidate(record, family, narrow_groups(messages), "belle_single")
    for name, location in (("candidates.sqlite3", "ordinal"), ("candidates_spans.sqlite3", "byte_offset")):
        path = (output / name).resolve()
        if not path.exists():
            continue
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            records = db.execute("SELECT repo, revision, ordinal, payload FROM candidates ORDER BY repo, ordinal").fetchall()
            statement = ("SELECT repo, revision, NULL, rows, counts FROM progress ORDER BY repo"
                         if location == "ordinal"
                         else "SELECT repo, revision, start_byte, rows, counts FROM spans ORDER BY start_byte")
            acquisition.extend(
                {"repo": repo, "revision": revision, "start_byte": start, "scanned_records": count,
                 "counts": json.loads(reasons)}
                for repo, revision, start, count, reasons in db.execute(statement)
            )
        for repo, revision, cursor, content in records:
            messages = json.loads(content)
            # Re-run the current gate so an older acquisition cache cannot publish stale decisions.
            reason, family = belle_quality(messages)
            if reason is not None:
                continue
            record = {"id": rank(f"{repo}:{revision}:{location}:{cursor}", "belle"), "messages": messages, "source": repo,
                      "source_record": {"revision": revision, location: cursor}}
            kind = "belle_multi" if len(messages) > 2 else "belle_single"
            yield Candidate(record, family, narrow_groups(messages), kind)


def validate_labels(row):
    ids, labels = row["input_ids"], row["labels"]
    if not 0 < len(ids) == len(labels) <= MAX_TOKENS:
        raise ValueError("SFT row has invalid context length")
    role_ids = {SPECIAL_IDS[f"<|{role}|>"]: role for role in ("system", "user", "assistant")}
    if ids[0] != SPECIAL_IDS["<|bos|>"] or labels[0] != IGNORE_LABEL:
        raise ValueError("SFT row has invalid BOS or BOS supervision")
    if ids[-1] != SPECIAL_IDS["<|eos|>"] or labels[-1] != ids[-1]:
        raise ValueError("SFT row has invalid EOS or EOS supervision")
    roles = []
    active_role = None
    content_tokens = 0
    for token, label in zip(ids[1:-1], labels[1:-1]):
        if token in role_ids:
            if active_role is not None or label != IGNORE_LABEL:
                raise ValueError("Misplaced or supervised role marker")
            active_role = role_ids[token]
            roles.append(active_role)
            content_tokens = 0
        else:
            expected = token if active_role == "assistant" else IGNORE_LABEL
            if active_role is None or label != expected:
                raise ValueError("SFT target is not assistant-only")
            if token == SPECIAL_IDS["<|end_turn|>"]:
                if not content_tokens:
                    raise ValueError("Empty tokenized turn")
                active_role = None
            elif token < len(SPECIAL_IDS):
                raise ValueError("Reserved structural token in message content")
            else:
                content_tokens += 1
    expected_roles = ["system", *[item["role"] for item in row["messages"]]]
    if roles != expected_roles or active_role is not None or roles[1:] != ["user", "assistant"] * (len(row["messages"]) // 2):
        raise ValueError("SFT row roles do not match complete source turns")
    if case := row.get("program_case"):
        if render_program(case["family"], case["index"]) != row["messages"]:
            raise ValueError("Program messages or answers differ from their regenerated inputs")


class Selection:
    def __init__(self, evaluation, old_sft, old_pt, tokenizer):
        self.evaluation = evaluation
        self.old_sft = old_sft
        self.old_pt = old_pt
        self.tokenizer = tokenizer
        self.indexes = {split: ConflictIndex() for split in SPLITS}
        self.groups = {}
        self.train_groups = Counter()
        self.ids = set()
        self.selected = {split: [] for split in SPLITS}
        self.families = {split: Counter() for split in SPLITS}
        self.excluded = Counter()

    def keep(self, candidate, split):
        row = candidate.row
        messages = row["messages"]
        if row["id"] in self.ids:
            return self.reject("selected_id")
        if any(self.groups.get(group, split) != split for group in candidate.group):
            return self.reject("heldout_group")
        if split == "train" and self.families[split][candidate.family] >= FAMILY_CAP:
            return self.reject("family_cap")
        if split == "train" and any(self.train_groups[group] >= FAMILY_CAP for group in candidate.group):
            return self.reject("narrow_group_cap")
        for key in prompts(messages):
            if reason := self.evaluation.match(key):
                return self.reject(f"evaluation_{reason}")
            if reason := self.old_sft.match(key):
                return self.reject(f"old_sft_{reason}")
        for message in messages:
            for key in pt_chunks(message["content"]):
                if reason := self.old_pt.match(key):
                    return self.reject(f"old_pt_{reason}")
        keys = candidate_keys(messages)
        for peer, index in self.indexes.items():
            if peer == split and row["source"] == SOURCE:
                if any(key in index.exact for key in keys):
                    return self.reject("same_split_exact")
                continue
            for key in keys:
                if reason := index.match(key):
                    return self.reject(f"{'same' if peer == split else 'cross'}_split_{reason}")
        encoded = encode_conversation(self.tokenizer, messages)
        if len(encoded["input_ids"]) > MAX_TOKENS:
            return self.reject("over_context_length")
        data = {**row, **encoded, "template_family": candidate.family,
                "group_id": rank(candidate.group[0], "group")}
        validate_labels(data)
        self.selected[split].append(data)
        self.ids.add(row["id"])
        self.families[split][candidate.family] += 1
        for group in candidate.group:
            self.groups[group] = split
            if split == "train":
                self.train_groups[group] += 1
        for key in keys:
            self.indexes[split].add(key)
        return True

    def reject(self, reason):
        self.excluded[reason] += 1
        return False


def collect_real(candidates, selection, split, kind, target, group_sizes):
    kept = 0
    family_counts = Counter()
    ranked = sorted(candidates, key=lambda item: rank(item.row["id"], split))
    if split != "train":
        # A heldout sample owns its groups. Prefer small groups to reserve common training patterns.
        ranked = sorted(ranked, key=lambda item: any(group_sizes[group] > 24 for group in item.group))
    for candidate in ranked:
        if kept == target:
            break
        if selection.keep(candidate, split):
            family_counts[candidate.family] += 1
            kept += 1
    print(f"[comparison SFT] {split} {kind}: {kept:,}/{target:,}, {len(family_counts)} families", flush=True)
    return kept, dict(family_counts)


def collect(args, tokenizer):
    evaluation, old_sft, identities = external_index(args.old_sft, [args.old_eval, *args.eval])
    pt, pt_identities = pt_index(args.base)
    identities.extend(pt_identities)
    print(f"[comparison SFT] indexed {len(evaluation.exact):,} eval prompts, "
          f"{len(old_sft.exact):,} old SFT prompt surfaces and "
          f"{len(pt.exact):,} old pretraining holdout chunks", flush=True)
    selection = Selection(evaluation, old_sft, pt, tokenizer)
    candidates = defaultdict(list)
    old_quality_counts = Counter()
    acquisition = []
    for candidate in belle_candidates(args.old_sft, args.out, acquisition):
        candidates[candidate.kind].append(candidate)
        if "old_train_id" in candidate.row["source_record"]:
            old_quality_counts[candidate.kind] += 1
    inventory = {kind: dict(Counter(item.family for item in rows)) for kind, rows in candidates.items()}
    group_sizes = Counter(group for items in candidates.values() for item in items for group in item.group)
    print(f"[comparison SFT] prefiltered BELLE before overlap checks: "
          f"{[(kind, len(items)) for kind, items in candidates.items()]}", flush=True)
    shortages = {}
    by_selection = {}
    quotas, program_quotas = comparison_quotas(QUOTAS["train"]["belle_multi"])
    for split, kinds in (("validation", ("belle_single", "belle_multi")),
                         ("test", ("belle_single", "belle_multi")),
                         ("train", ("belle_multi", "belle_single"))):
        for kind in kinds:
            target = QUOTAS[split][kind]
            if split == "train" and kind == "belle_single":
                target = quotas[split][kind]
            kept, by_selection[f"{split}_{kind}"] = collect_real(candidates[kind], selection, split, kind, target, group_sizes)
            if kept != target:
                if split == "train" and kind == "belle_multi" and kept >= args.min_belle_multi:
                    quotas, program_quotas = comparison_quotas(kept)
                else:
                    shortages[f"{split}_{kind}"] = target - kept
        for family, quota in program_quotas[split].items():
            for candidate in program_candidates(family):
                if selection.keep(candidate, split):
                    quota -= 1
                if quota == 0:
                    break
            if quota:
                shortages[f"{split}_program_{family}"] = quota
    report = {"prefiltered_belle_families": inventory, "old_train_prefiltered_belle": dict(old_quality_counts),
              "selected_belle_families": by_selection, "selection_rejections": dict(selection.excluded),
              "shortages": shortages, "screened_inputs": identities,
              "exclusive_selected_groups": len(selection.groups),
              "largest_train_narrow_group": max(selection.train_groups.values(), default=0),
              "acquisition": acquisition, "quotas": quotas}
    return selection.selected, report


def row_kind(row):
    source = "program" if row["source"] == SOURCE else "belle"
    turns = "single" if len(row["messages"]) == 2 else "multi"
    return f"{source}_{turns}"


def split_stats(rows):
    kinds = Counter(map(row_kind, rows))
    locations = Counter()
    for row in rows:
        if origin := row.get("source_record"):
            location = ("legacy_train" if "old_train_id" in origin else "pinned_prefix"
                        if "ordinal" in origin else "pinned_byte_span")
            locations[location] += 1
    return {"documents": len(rows), "tokens": sum(len(row["input_ids"]) for row in rows),
            "assistant_tokens": sum(sum(label != IGNORE_LABEL for label in row["labels"]) for row in rows),
            "single": kinds["belle_single"] + kinds["program_single"],
            "multi": kinds["belle_multi"] + kinds["program_multi"],
            "by_kind": dict(kinds), "by_source": dict(Counter(row["source"] for row in rows)),
            "belle_provenance": dict(locations),
            "by_family": dict(Counter(row["template_family"] for row in rows)),
            "belle_family_count": len({row["template_family"] for row in rows if row["source"] != SOURCE}),
            "program_family_count": len({row["template_family"] for row in rows if row["source"] == SOURCE})}


def program_coverage(selected):
    coverage = {}
    for split, (objects, locations) in PROGRAM_ENTITIES.items():
        entities = set()
        combinations = set()
        structures = Counter()
        for row in selected[split]:
            if row["source"] != SOURCE:
                continue
            user_text = "\n".join(turn["content"] for turn in row["messages"] if turn["role"] == "user")
            present = tuple(entity for entity in (*objects, *locations) if entity in user_text)
            if present:
                entities.update(present)
                combinations.add(present)
            if structure := ARITHMETIC_STRUCTURES.get(row["program_case"]["family"]):
                structures[structure] += 1
        coverage[split] = {"entity_vocabulary": sorted(entities), "entity_combinations": len(combinations),
                           "arithmetic_structures": dict(structures)}
    return coverage


def stratified_samples(selected, prior_ids=frozenset()):
    targets = {"train": {"belle_single": 21, "belle_multi": 11, "program_single": 5, "program_multi": 3},
               "validation": {"belle_single": 2, "belle_multi": 1, "program_single": 1, "program_multi": 1},
               "test": {"belle_single": 2, "belle_multi": 1, "program_single": 1, "program_multi": 1}}
    samples = []
    for split, counts in targets.items():
        for kind, count in counts.items():
            ordered = sorted((row for row in selected[split] if row_kind(row) == kind),
                             key=lambda row: rank(row["id"], f"audit:reviewed:{split}"))
            fresh = [row for row in ordered if row["id"] not in prior_ids]
            if len(fresh) >= count:
                ordered = fresh
            used = set()
            distinct = []
            for row in ordered:
                if row["template_family"] not in used:
                    distinct.append(row)
                    used.add(row["template_family"])
            first = {row["id"] for row in distinct}
            ordered = distinct + [row for row in ordered if row["id"] not in first]
            for row in ordered[:count]:
                samples.append({"split": split, "kind": kind, **{key: row[key] for key in
                                ("id", "source", "template_family", "messages")}})
    return samples


def build(args):
    base = json.loads((args.base / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_path = Path(base["tokenizer_path"]).resolve()
    identity = file_identity(tokenizer_path)
    if (base["schema_version"] != 1 or base["frozen"] is not True or base["corpus_id"] != "stage1_v1"
            or base["special_token_ids"] != SPECIAL_IDS or identity != base["files"]["tokenizer.json"]
            or sha256(tokenizer_path.read_bytes()) != base["tokenizer_sha256"]):
        raise ValueError("Frozen Stage 1 tokenizer identity does not match")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    check_special_ids(tokenizer)
    if tokenizer.get_vocab_size() != base["vocab_size"] or base["vocab_size"] != 16_384:
        raise ValueError("Unexpected frozen tokenizer vocabulary")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with build_lock(args.out):
        selected, report = collect(args, tokenizer)
        stats = {split: split_stats(selected[split]) for split in SPLITS}
        report.update({"splits": stats, "family_cap": FAMILY_CAP,
                       "program_coverage": program_coverage(selected)})
        report_prefix = "sftdata_preview" if args.check_only else "sftdata"
        write_json(args.report_dir / f"{report_prefix}_quality.json", report)
        prior_snapshot = args.report_dir / "sftdata_round1_samples.json"
        prior_ids = frozenset()
        if prior_snapshot.exists():
            prior_ids = frozenset(row["id"] for row in json.loads(prior_snapshot.read_text(encoding="utf-8")))
        write_json(args.report_dir / f"{report_prefix}_samples.json", stratified_samples(selected, prior_ids))
        if report["shortages"]:
            raise RuntimeError(f"SFT quotas are incomplete: {report['shortages']}")
        if any(stats[split]["by_kind"] != report["quotas"][split] for split in SPLITS):
            raise RuntimeError("SFT split composition differs from quotas")
        if args.check_only:
            print("[comparison SFT] quotas and labels verified; no corpus frozen", flush=True)
            return
        sources = json.loads((args.old_sft / "manifest.json").read_text(encoding="utf-8"))["sources"]
        sources.append({"repo": SOURCE, "generator": "tinychat/comparison_sft_data.py",
                        "description": "Deterministic local fictional records and arithmetic with recomputable answers.",
                        "usage": "Packaged with the BELLE research-only corpus."})
        files = {"tokenizer.json": identity}
        for split, rows in selected.items():
            name = f"{split}.jsonl"
            path = args.out / name
            pending = path.with_suffix(".jsonl.partial")
            with pending.open("w", encoding="utf-8", newline="\n") as handle:
                for row in sorted(rows, key=lambda item: rank(item["id"], f"output:{split}")):
                    handle.write(json_line(row))
            pending.replace(path)
            files[name] = file_identity(path)
        manifest = {
            "schema_version": 1, "corpus_id": CORPUS_ID, "frozen": True, "seed": SEED, "max_tokens": MAX_TOKENS,
            "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": base["tokenizer_sha256"],
            "vocab_size": base["vocab_size"], "special_token_ids": SPECIAL_IDS, "files": files,
            "counts": {split: stats[split]["documents"] for split in SPLITS}, "splits": stats,
            "quotas": report["quotas"], "sources": sources, "screened_inputs": report["screened_inputs"],
            "belle_acquisition": report["acquisition"], "program_coverage": report["program_coverage"],
            "usage": "Research only; no commercial derivatives under the source publisher's usage restrictions.",
            "supervision": "Assistant content and its end-of-turn markers, plus final EOS, per Stage 1 encoding.",
            "split": "Holdouts chosen first with seeded ranks, preferring groups with at most 24 candidates. "
                     "Selected rows cannot share normalized initial topic, "
                     "recognized lexical template, quoted entity group, or program family across splits. Exact and "
                     "five-character set Jaccard >= 0.8 block supported cross-split prompt surfaces. Broad BELLE "
                     "intent classes span splits. Unquoted entities, paraphrases, later-only contextual overlap "
                     "and semantic overlap remain possible. Program parameters vary within same-split families. "
                     "Complete old and new evaluation prompts use exact matching and symmetric containment from "
                     "six normalized characters; old SFT heldout prompt containment starts at 24. "
                     "Old pretrain heldouts are screened against both roles using normalized whole texts or "
                     "sentence/adjacent-sentence chunks 40–384 characters and rolling 192-character windows "
                     "at stride 96 for longer texts. Filtering checks exact, containment, and Jaccard >= 0.8.",
        }
        name = "manifest.prepared.json" if args.prepare_only else "manifest.json"
        write_json(args.out / name, manifest)
    if args.prepare_only:
        print(f"[comparison SFT] prepared {manifest['counts']} for independent review", flush=True)
        return
    print(f"[comparison SFT] frozen {manifest['counts']}", flush=True)


def publish_prepared(out):
    with build_lock(out):
        prepared = out / "manifest.prepared.json"
        manifest = json.loads(prepared.read_text(encoding="utf-8"))
        expected, _ = comparison_quotas(manifest["quotas"]["train"]["belle_multi"])
        if (manifest["corpus_id"] != CORPUS_ID or manifest["quotas"] != expected
                or manifest["frozen"] is not True):
            raise ValueError("Prepared comparison SFT manifest does not match the frozen protocol")
        for name, identity in manifest["files"].items():
            path = Path(manifest["tokenizer_path"]) if name == "tokenizer.json" else out / name
            if file_identity(path) != identity:
                raise ValueError(f"Prepared file changed before publication: {path}")
        prepared.replace(out / "manifest.json")
    print(f"[comparison SFT] frozen {manifest['counts']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sft_clean_v2"))
    parser.add_argument("--old-sft", type=Path, default=Path("data/sft_stage1_v1"))
    parser.add_argument("--base", type=Path, default=Path("data/stage1_v1"))
    parser.add_argument("--old-eval", type=Path, default=Path("evaluation/stage1_v1/prompts.jsonl"))
    parser.add_argument("--eval", type=Path, nargs="*")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/comparison_v2"))
    parser.add_argument("--local-source", type=Path)
    parser.add_argument("--single-records", type=int, default=100_000)
    parser.add_argument("--multi-records", type=int, default=50_000)
    parser.add_argument("--min-belle-multi", type=int, default=QUOTAS["train"]["belle_multi"])
    parser.add_argument("--multi-offsets", type=int, nargs="+")
    parser.add_argument("--span-records", type=int, default=2_500)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--acquire-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--publish-prepared", action="store_true")
    mode.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    if args.publish_prepared:
        publish_prepared(args.out)
        return
    if args.inventory:
        with closing(sqlite3.connect((args.out / "candidates.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
            for row in db.execute("SELECT repo, rows, counts FROM progress"):
                print(*row)
            for row in db.execute("SELECT family, count(*) FROM candidates GROUP BY family ORDER BY count(*) DESC"):
                print(*row)
        return
    if args.acquire_only:
        if args.multi_offsets:
            acquire_multi_spans(args.out, args.old_sft, args.multi_offsets, args.span_records, args.local_source)
            return
        acquire(args.out, args.old_sft, {"single": args.single_records, "multi": args.multi_records}, args.local_source)
        return
    if not args.eval and not args.check_only:
        parser.error("--eval requires the frozen comparison evaluation file(s)")
    if not 1_000 <= args.min_belle_multi <= QUOTAS["train"]["belle_multi"]:
        parser.error("--min-belle-multi must be between 1,000 and 2,000")
    args.eval = args.eval or []
    build(args)


if __name__ == "__main__":
    main()
