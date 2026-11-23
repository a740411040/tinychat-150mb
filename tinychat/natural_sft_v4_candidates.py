"""Local v4 candidate extraction. Admission still requires semantic review."""

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .chat_format import IGNORE_LABEL, contains_reserved, encode_conversation
from .natural_sft_source_candidates import (
    BAD_FORMAT,
    IDENTITY,
    MOSS_UNSUPPORTED,
    OVERCLAIM,
    RISK,
    ROLE,
    parse_moss,
    sentence_spans,
)
from .sft_data import conversation_key
from .stage1_data import normalized_content


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/stage1_v1/tokenizer.json"
PUBLIC = ROOT / "data/public_sources/natural_sft_v3"
MOSS = PUBLIC / "moss-002-sft-data"
CMRC = PUBLIC / "cmrc2018"
BELLE = ROOT / "data/sft_clean_v2"
BELLE_STAGE1 = ROOT / "data/sft_stage1_v1"
MAX_TOKENS = 512
CATEGORIES = ("natural", "grounded", "multi", "instruction")
BELLE_USAGE = "Research only; no commercial derivatives per publisher"
BELLE_REVISIONS = {
    "BelleGroup/train_0.5M_CN": (
        "29c35e9b56c7fc91f04a613ed33896f7a19bad54", "Belle_open_source_0.5M.json", "single",
    ),
    "BelleGroup/multiturn_chat_0.8M": (
        "dc32eb8955e10ae1cd4189a50a566ef8f8254a65", "multiturn_chat_0.8M.json", "multi",
    ),
}
CMRC_REVISION = "137f2c45a24275fb68f6961c4d357f46288886aa"

HAN = re.compile(r"[\u3400-\u9fff]")
DIGITS = re.compile(r"\d+(?:[.,，．]\d+)*")
OUTSIDE_FACT = re.compile(
    r"https?://|www\.|据(?:报道|统计|研究|调查)|实验结果|最新(?:消息|排行|数据|政策|研究)|"
    r"(?:排名|票房|评分|排行榜|知名品牌|著名|创始于|生于|发表于|总部位于|出自于|发源于|起源于)|"
    r"\d{4}[年/-]|\d+(?:\.\d+)?\s*[%％]|\b(?:ChatGPT|OpenAI|BELLE|JSON|API|SQL)\b", re.I,
)
TECHNICAL_TASK = re.compile(
    r"代码|编程|正则表达式|人工智能|机器人|算法|方程|实验|化学反应|电子电路|"
    r"学术论文|政治|历史事件|流行病|临床|市值|营业额|推荐.{0,12}(?:品牌|店铺|酒店|电影|APP)|实时|今日行情", re.I,
)
CLAIM_OPENING = re.compile(r"^(?:据|研究(?:发现|显示)|作为(?:一名|一个)|我是(?:一个|一名)|我曾经|我已经帮你)")
EMPTY_ACK = re.compile(r"^(?:好的|没问题|当然|当然可以|谢谢|不客气|收到|是的)[，,。！!、\s]*$")
TRAILING_FILLER = re.compile(r"(?:希望(?:这些|我的).{0,14}(?:帮助|帮到)|如有任何问题|需要更多帮助|随时(?:问我|联系我))")
PROMPT_WRAPPER = re.compile(
    r"(?:情境|场景|给定|下面是|以下是|根据(?:以下|下面|输入|上下文)|此处提供|以下输入|用户(?:输入|说|提问|需要)|"
    r"提供(?:一条|一个|一项|[\u3400-\u9fff]{1,10}的)建议|输出|对话如下|输入示例|扮演|假设你|续写|模拟|请写|"
    r"写(?:一|个|封)|请列出|列举|描述|分析|解释|定义|翻译|总结|提取|分类|讲解)",
)
DAY_SITUATION = re.compile(
    r"收纳|整理(?:房间|书桌|书架|衣柜|资料|衣服|行李)|收拾|搬家|打包|室友|同事|朋友|邻居|"
    r"周末|房间|书桌|衣柜|客厅|行李|阳台|见面|过生日|散步|日记|早起|作息|拖延|通勤|做饭|"
    r"约好|约时间|礼物|家务|下班|上班|沟通|道歉|拒绝|迟到|见面|聚会|紧张|难过|开心|烦恼|有点累",
)
DIRECT_ASK = re.compile(r"怎么|怎样|如何|怎么办|有什么|合适吗|好不好|该不该|要不要|能不能|可不可以|你觉得")
FIRST_PERSON = re.compile(r"(?:^|[，。！；？,?!\s])(?:我|我们|家里|这几天|最近我|今天我|昨晚我|这周我)")
DIRECT_START = re.compile(r"^(?:请问[，,：:]?)?(?:怎么|怎样|如何|哪里|该怎么|要不要|家里|今天|这几天|不太确定|想(?:问|收拾|整理|约|和|跟|把))")
FEELING = re.compile(r"紧张|有点累|难过|委屈|很烦|疲惫|想哭|失落|开心|心情不好|担心|焦虑|尴尬")
EMPATHY = re.compile(r"(?:听起来|能理解|可以理解|确实|这种时候|辛苦了|挺不容易|慢慢来|给自己|替你高兴|真不错)")
CONCRETE_ACTION = re.compile(r"(?:先|再|可以|不妨|试试|要不)[\u3400-\u9fff]{0,7}(?:挑|找|约|留|把|放|收|腾|问|说|写|列|聊|分|贴|装|发|拿|搬|摆)")
ACTION_DURATION = re.compile(r"(?:先|可以先|试着先)?(?:拿出|花|留出|预留|留|腾出|用)\s*(?:约|大约)?\s*\d{1,2}\s*分钟(?=[，,。；;、]|来|把|收拾|整理|热身|准备)")
UNBOUNDED_CLAIM = re.compile(r"(?:能|会|可|可以).{0,18}(?:提升|提高|降低|节省|减少|缓解).{0,8}(?:率|分贝|卡路里|百分点|百分之|[%％]|[0-9])|效果.{0,4}(?:达到|提升)|(?:所有|绝对|必然|一定|保证).{0,10}(?:有效|成功|解决)")
CLARIFY_ASK = re.compile(r"(?:不太确定|不确定|不知道|没想好|有点拿不准|不知道该).{0,15}(?:怎么|怎样|如何|要不要|做什么)|^(?:怎么|怎样|如何).{0,16}(?:开口|沟通|商量|说)")
CLARIFY_REPLY = re.compile(r"^(?:你|先确认一下|方便说说).{2,75}(?:是|还是|哪|什么|多少|哪一类|具体).{0,42}[？?]$")
CLARIFY_TOPICS = (
    (re.compile(r"室友|合住"), re.compile(r"作息|卫生|噪音|家务|分工|边界|房租")),
    (re.compile(r"同事|上班|工作"), re.compile(r"分工|进度|会议|时间|任务|反馈|约时间")),
    (re.compile(r"搬家|行李|打包|整理"), re.compile(r"物品|空间|时间|行李|纸箱|衣服|带走|留下")),
    (re.compile(r"朋友|聚会|约好|见面"), re.compile(r"时间|地点|人数|活动|计划|想聊|误会")),
)
SOCIAL = (
    (re.compile(r"(?:你?好|嗨|哈喽|早上好|下午好|晚上好|早安|晚安)[！!。\s]*"),
     re.compile(r"(?:你好|嗨|哈喽|早上好|下午好|晚上好|早安|晚安)")),
    (re.compile(r"(?:谢谢(?:你|您)?|多谢(?:你|您)?|太感谢了|辛苦你了)[！!。～\s]*"),
     re.compile(r"不客气|不用谢|小事|不必客气|很高兴(?:能|可以)?帮")),
)
USER_TEXT = re.compile(
    r"(?:改写|重写|重新书写|改成|改为|润色|换种说法|换个说法|简化|精简|缩写|缩短|缩成|摘要|总结|概括|标题|纠正|修正|校对|提取|摘录)",
)
TEXT_NAME = re.compile(r"^\s*(?:原文|原句|句子|文本|段落|内容|输入(?:文本)?)\s*[:：]\s*")
SUMMARY = re.compile(r"精简|简化|缩短|缩成|改短|更简洁|更精炼|短一点|简短|简明")
AMBIGUOUS_EDIT = re.compile(r"纠正|修正|校对|错误|纠错|标题|语气|风格|受众|孩子|学生|重新组织|正式|口语|轻松|不超过|限制|分别|并且|同时|附上|理由|英文|英语|法语|西语|日语|韩语")
END_SENTENCE = re.compile(r"(?:首句|第一句|开头(?:的)?一句|末句|最后一句|结尾(?:的)?一句)")
EXACT_QUOTE = re.compile(r"引用|摘录|原样|原句|逐字|照抄|抄写")
FICTION = re.compile(r"^.{0,12}(?:写|编|创作)(?:一段|一个|一小段|一则|一篇).{0,20}(?:小故事|童话|片段|虚构故事)")
BRAINSTORM = re.compile(r"想(?:个|一个|几个|出|一想|一款).{0,12}(?:创意|点子|方案)|(?:思考|设计|构思|提出).{0,10}(?:创意|点子|方案|应用|游戏)")
BACKREF = re.compile(r"刚才|刚刚|前面|上面|上一(?:段|句|条|个)|你(?:说|提到|写)的|这(?:段|句|篇|个)|(?:它|把它)|其[中余]")
UPDATE = re.compile(r"(?:把|将)(?:上面|前面|刚才)(?:用的|说的|提到的|的)?(?P<old>[\w\u3400-\u9fff]{1,14})(?:换成|改成|改为|改用)(?P<new>[\w\u3400-\u9fff]{1,14})[。！!？?]*$")
CORRECT_VALUE = re.compile(r"(?:我说的|我说的是|我刚才说错了[，,])?不是(?P<old>[\w\u3400-\u9fff]{1,14})[，,、](?:而)?是(?P<new>[\w\u3400-\u9fff]{1,14})[。！!？?]*$")
OBJECT_REF = re.compile(r"(?:这个|这款|这套|该款|该|上述|你(?:提到|说)的)(?P<object>app|应用程序|应用|小程序|游戏|方案|计划|活动|故事|课程|功能|运动|工具|书架|家务安排)", re.I)
SPECIFIES = re.compile(r"(?:包含|需要|设计|加上|提供|有什么|有哪些).{0,10}(?:功能|环节|活动|步骤)|(?:什么|哪些)(?:功能|环节|活动|步骤)|具体(?:怎么|如何|做法|功能)")
FEATURE_PAYLOAD = re.compile(r"视频|计划|功能|提醒|记录|分享|互动|动作|先|再|步骤|环节|安排|打卡")
FORMAT_PREFIX = re.compile(r"^(?:答(?:案)?|原句|原文|总结|改写后|摘要|标题|改写|修改后|好的|没问题)\s*[:：，,]\s*")
COUNT_CUE = re.compile(r"(?:至少|最多|不超过|不少于|约|大约|限制|限定|控制|在)?\s*(\d{1,4})\s*个?\s*(字|句|段|行|条)")
ONE_SENTENCE = re.compile(r"一(?:个完整的)?句(?:话|子)|只写一句|用一句")
SENTENCE = re.compile(r"[^。！？!?\n]+")
DAY_STATE = re.compile(r"(?P<day>今天|今日|昨天|昨日|明天|明日)(?:是(?:一个|个))?(?P<body>.+?)(?:的日子)?$")
DAY_ATTRS = frozenset("充实 有意义 平静 愉快 开心 高兴 忙碌 轻松 顺利 普通 温暖 难忘".split())
CMRC_SLOTS = (
    ("date", re.compile(r"什么时候|什么时间|什么时期|哪个时期|何时|哪一年|哪年|哪一天|哪天|几月几[日号]")),
    ("place", re.compile(r"哪里|哪儿|何处|什么地方|哪(?:一)?个(?:国家|城市|省份|省|州|地区|地方|县|岛)")),
    ("name", re.compile(r"叫什么(?:名字|名称)?|称为什么|叫作什么|名为什么|用什么名称|什么(?:名字|名称)|叫什么")),
    ("count", re.compile(r"多少|几(?:个|人|名|部|集|届|次|篇|米|公里|年|岁|位|座|条|种|项|天|月|枚|本)")),
    ("author", re.compile(r"(?:原)?作者(?:是|为)?谁|由谁(?:编写|创作|撰写)|谁(?:写|创作|撰写)的")),
    ("reason", re.compile(r"为什么|为何|什么原因|原因是什么")),
    ("purpose", re.compile(r"目的是什么|什么目的|用来做什么|用来干什么|用途是什么")),
)
COMPLEX_QUESTION = re.compile(r"如何|怎样|哪些|哪两|哪三|有哪些|是不是|是否|除了|分别|(?:什么|哪些)(?:影响|意义|作用|效果)|主要|最(?:大|小|长|高|早|晚)|和谁|与谁")
DATE_VALUE = re.compile(r"(?:公元前)?[一二三四五六七八九十零〇\d]{1,6}年(?:代(?:[初中末](?:期)?)?|\s*\d{1,2}月(?:\s*\d{1,2}[日号])?)?|\d{1,2}月(?:\s*\d{1,2}[日号])?")
DATE_WITH_PREDICATE = re.compile(r"^(?:(?:始建|建成|建|成立|创立|开馆|创办|出版|发行|播出|推出|上映|出生|去世|开幕|举办|竣工|完工|建造|设立)于|自|从|于)?\s*")
COUNT_UNIT = r"公里|千米|[人名部集届次篇米岁位座条种项个天月枚年本]"
NUMBER_VALUE = re.compile(r"约?\s*[\d零〇一二三四五六七八九十百千万亿两点.,，．]+\s*(?:" + COUNT_UNIT + r")?")
ENTITY_VALUE = re.compile(r"[\w\u3400-\u9fff·・《》「」“”\s.-]{2,32}")
PLACE_BEFORE = re.compile(r"(?:位于|地处|坐落(?:于|在)|迁(?:往|至)|于|在|到|至|来自)\s*$")
TITLE_BEFORE = re.compile(r"(?:称(?:之)?为|又名|亦称|被称作|叫作|名为|取名(?:为)?|名字(?:是|为)|命名为|名称(?:是|为)|简称|俗称)\s*[《「“]?\s*$")
QUESTION_FILLER = re.compile(r"^(?:请问|根据(?:文章|资料|材料|原文)[，,：:]?)|[？?。\s]+$")
CLAUSE_SPLIT = re.compile(r"[，,；;。！？!?\n]")
SUBJECT_END = re.compile(r"的|(?=\d{2,})|实验(?=的|是)|(?:最初)?(?:建|创立|成立|建成|始建|开馆|出版|发行|上映|举办|设立|出生|去世|竣工)于|来自|位于|地处|又称|又叫|称为|简称|名为|叫作|叫做|坐落在|共有|一共有|总共有|有(?:约|共)?$")
RELATION_WRAPPER = re.compile(r"^(?:的|是|在|从|于|约|由|被)+|(?:的|了|是|在|于|从|为|由|叫|有|约|共|哪个|哪个国家|哪个城市)+$")
COREFERENCE = re.compile(r"(?:^|[，,；;])(?:\d{2,4}年(?:\d{1,2}月(?:\d{1,2}日)?)?)?\s*(?:她|他(?:本人)?|其|该[\u3400-\u9fff]{1,6}|本(?:书|校|馆|片|剧|队|鱼|物种)|它)")
INSTITUTION_CHAIN = re.compile(r"(?:大学|学院|学校)$")
AUTHOR_LEAD = re.compile(r"(?:作者(?:是|为)|由|(?:是|为)[^，。]{0,4}(?:作家|小说家|学者|诗人|编剧)?)\s*$")
AUTHOR_TAIL = re.compile(r"^\s*(?:所著|撰写|编写|编著|创作|的(?:小说|作品|长篇|书籍|著作|诗集|剧本))")
CAUSE_LEAD = re.compile(r"(?:^|[，,；;。])\s*(?:原因(?:是|在于)|由于|因为|因(?=[^此而缘]))")
PURPOSE_LEAD = re.compile(r"(?:目的是|目的在于|旨在|用来|用于)\s*$")
WIKI_RENDERING = re.compile(
    r"-\{(?P<plain>[\u3400-\u9fff]{0,16})\}-|[（(]\s*[,，/／、；;·]*\s*[）)]|"
    r"[（(](?:学名|英文|原名|英文名|外文名|谚文|朝鲜汉字|韩文|日语|罗马字)\s*[:：]\s*[）)]|"
    r"[,，]\s*(?:(?:朝鲜汉字|外文名|英文名|学名|罗马字)\s*[:：]\s*)?(?=[）)])|"
    r"[“‘「『《\"]\s*[”’」』》\"]",
)
WIKI_UNRESOLVED = re.compile(r"-\{|\}-|\bzh-(?:hans|hant|cn|tw|hk)\s*[:：]|\{\{|\}\}|\[\[")
MATERIAL_QUESTION = re.compile(
    r"^(?:(?:请)?(?:根据|依据|阅读|读完|只根据|仅根据)(?:下面的|以下的|下列|以下|下面)?(?:材料|文章|短文|文本|段落|资料)"
    r"(?:回答(?:下列|下面的|以下|下面)?(?:问题)?)?[。！!：:\s]+)?"
    r"(?:材料|文章|短文|文本|段落|资料|原文)\s*[:：]\s*(?P<material>.+?)\n+\s*(?:问题|问)\s*[:：]\s*(?P<question>[^\n]+?)\s*$",
    re.S,
)


@dataclass
class SourceRow:
    name: str
    group_id: str
    provenance: dict
    original: object


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def msg(role, content):
    return {"role": role, "content": content}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def json_rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def moss_rows():
    path = MOSS / "zh_helpfulness.prefix64m.jsonl"
    metadata_path = MOSS / "zh_helpfulness.prefix64m.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata["repo"], metadata["revision"], metadata["license_from_card"], metadata["source_file"],
            metadata["range_start"], metadata["range_end_inclusive"]) != (
        "OpenMOSS-Team/moss-002-sft-data", "d530c921add4c7551b4e2309e0e7fe9468278311",
        "cc-by-4.0", "zh_helpfulness.json", 0, 67_108_863,
    ):
        raise ValueError(f"Unexpected fixed MOSS helpfulness source: {metadata_path}")
    for line, row in enumerate(json_rows(path), 1):
        row_id = row.get("id")
        provenance = {
            "repo": metadata["repo"], "revision": metadata["revision"],
            "license": metadata["license_from_card"], "license_record": relative(MOSS / "README.md"),
            "source_file": metadata["source_file"], "local_file": relative(path),
            "source_row_id": row_id, "jsonl_line_1_based": line,
            "acquisition_record": relative(metadata_path), "original_byte_range_inclusive": [0, 67_108_863],
            "source_selection": metadata["selection_bias"],
        }
        group = f"moss-helpfulness:{row_id}" if type(row_id) in (str, int) else f"moss-helpfulness:line:{line}"
        original = {"messages": parse_moss(row), "source_num_turns": row.get("num_turns"), "source_prefix": row.get("prefix")}
        yield SourceRow("moss_helpfulness", group, provenance, original)


def belle_rows():
    for filename, location in (("candidates.sqlite3", "ordinal_1_based"),
                               ("candidates_spans.sqlite3", "source_byte_offset")):
        path = BELLE / filename
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            for repo, revision, cursor, old_family, text in db.execute(
                "SELECT repo, revision, ordinal, family, payload FROM candidates ORDER BY repo, ordinal",
            ):
                expected, source_file, kind = BELLE_REVISIONS[repo]
                if revision != expected:
                    raise ValueError(f"Unexpected BELLE revision for {repo}: {revision}")
                messages = json.loads(text)
                group = conversation_key(messages[:2])
                provenance = {
                    "repo": repo, "revision": revision, "source_file": source_file,
                    "usage": BELLE_USAGE, "license_record": relative(BELLE / "manifest.prepared.json"),
                    "local_file": relative(path), "cache_table": "candidates", "cache_ordinal": cursor,
                    location: cursor, "old_cache_family": old_family,
                    "source_selection": "Previously screened cache of complete normalized source conversations",
                }
                yield SourceRow(f"belle_{kind}", f"belle:{group}", provenance, {"messages": messages})


def belle_stage1_rows():
    path = BELLE_STAGE1 / "train.jsonl"
    for number, row in enumerate(json_rows(path), 1):
        repo = row["source"]
        revision, source_file, kind = BELLE_REVISIONS[repo]
        group = row["group_id"]
        messages = row["messages"] if kind == "single" else None
        if messages is not None and (len(messages) != 2 or conversation_key(messages) != group):
            raise ValueError(f"Unexpected Stage 1 single-turn conversation identity at line {number}")
        provenance = {
            "repo": repo, "revision": revision, "source_file": source_file,
            "usage": BELLE_USAGE, "license_record": relative(BELLE_STAGE1 / "manifest.json"),
            "local_file": relative(path), "jsonl_line_1_based": number,
            "old_train_id": row["id"], "old_group_id": group,
            "source_selection": "Legacy Stage 1 single-turn records, where the entire pair was retained",
        }
        yield SourceRow(f"belle_stage1_{kind}", f"belle:{group}", provenance, {"messages": messages})


def cmrc_rows(split):
    from pyarrow.parquet import ParquetFile

    filename = f"data/{split}-00000-of-00001.parquet"
    path = CMRC / filename
    index = 0
    for batch in ParquetFile(path).iter_batches(batch_size=256, columns=["id", "context", "question", "answers"]):
        for row in batch.to_pylist():
            index += 1
            group = digest(normalized_content(row["context"]))
            provenance = {
                "repo": "hfl/cmrc2018", "revision": CMRC_REVISION,
                "license": "cc-by-sa-4.0", "license_record": relative(CMRC / "README.md"),
                "source_file": filename, "local_file": relative(path),
                "source_row_1_based": index, "source_row_id": row["id"],
                "source_split": split, "source_document_id": row["id"].rsplit("_QUERY_", 1)[0],
            }
            yield SourceRow(f"cmrc_{split}", f"cmrc2018:{group}", provenance, row)


def pair_quality(question, answer, *, allow_action_parameters=False, prior_text=""):
    for content in (question, answer):
        if not isinstance(content, str) or not content.strip() or contains_reserved(content) or ROLE.search(content):
            return "invalid_or_reserved_content"
        if RISK.search(content):
            return "high_risk_topic"
        if IDENTITY.search(content):
            return "source_identity"
        if BAD_FORMAT.search(content) or re.search(r"(?:^|\n)[QA][：:]\s*(?:$|\n)|\b(?:XXX|TBD)\b|\{[^{}\n]*\}", content, re.I):
            return "broken_format_or_placeholder"
        if len(HAN.findall(content)) < min(2, len(content)):
            return "insufficient_chinese"
        if OUTSIDE_FACT.search(content) or TECHNICAL_TASK.search(content):
            return "outside_facts_or_complex_task"
    if (OVERCLAIM.search(answer) or MOSS_UNSUPPORTED.search(answer) or CLAIM_OPENING.search(answer)
            or UNBOUNDED_CLAIM.search(answer)):
        return "unsupported_claim"
    if TRAILING_FILLER.search(answer):
        return "generic_service_filler"
    # List indices are formatting. A duration in an explicit proposal is an action option;
    # bare quantities, rates and outcomes are still not supported by a suggestion template.
    body = re.sub(r"(?m)^\s*\d{1,2}[.、)）]\s*", "", answer)
    if allow_action_parameters:
        body, count = ACTION_DURATION.subn("", body)
        if count > 1:
            return "multiple_unchecked_action_parameters"
    if not set(DIGITS.findall(body)) <= set(DIGITS.findall(question + "\n" + prior_text)):
        return "ungrounded_number"
    return None


def anchors(text, other, *, min_chars=3, max_chars=10):
    """Return shared nontrivial Chinese substrings, excluding generic cue phrases."""
    generic = re.compile(r"(?:这个|那个|这些|那些|刚才|上面|上述|内容|方法|需要|可以|应该|建议|通过|我们|什么|哪些|如何|怎么)")
    left = re.split(r"[^\u3400-\u9fff]+", generic.sub(" ", text))
    right = re.split(r"[^\u3400-\u9fff]+", generic.sub(" ", other))
    matches = set()
    for chunk in left:
        for width in range(min(len(chunk), max_chars), min_chars - 1, -1):
            hit = False
            for start in range(len(chunk) - width + 1):
                span = chunk[start:start + width]
                if any(span in candidate for candidate in right):
                    matches.add(span)
                    hit = True
            if hit:
                break
    return sorted(matches, key=lambda value: (-len(value), value))


def natural_family(question, answer):
    concise = answer.strip()
    for prompt, reply in SOCIAL:
        if prompt.fullmatch(question.strip()) and len(concise) <= 42 and reply.search(concise):
            return "greeting_or_thanks"
    if len(question) > 88 or len(concise) > 200 or PROMPT_WRAPPER.search(question) or EMPTY_ACK.fullmatch(concise):
        return None
    if not (FIRST_PERSON.search(question) or DIRECT_START.search(question)) or not DAY_SITUATION.search(question):
        return None
    detail = anchors(question, answer, min_chars=3)
    if CLARIFY_ASK.search(question) and CLARIFY_REPLY.fullmatch(concise):
        topic_matched = any(topic.search(question) and clarification.search(answer) for topic, clarification in CLARIFY_TOPICS)
        if detail or topic_matched:
            return "situated_clarifying_question"
    if (FIRST_PERSON.search(question) and FEELING.search(question) and EMPATHY.search(answer)
            and detail and len(concise) <= 120):
        return "emotional_acknowledgment"
    same_topic = any(match.group() in answer for match in DAY_SITUATION.finditer(question))
    if (DIRECT_ASK.search(question) and CONCRETE_ACTION.search(answer)
            and (detail or same_topic) and re.search(r"[。！？.!?]$", concise)
            and not re.search(r"^\s*(?:作为|以下|首先我们需要|答案|建议用户|建议他|你可以尝试以下)", concise)):
        return "situated_daily_advice"
    return None


def text_instruction(question):
    split = re.split(r"\n+|(?:[:：]\s*)", question, maxsplit=1)
    if len(split) != 2:
        return None
    task, text = split
    text = TEXT_NAME.sub("", text).strip().strip('“”「」"')
    if (USER_TEXT.search(task) or EXACT_QUOTE.search(task)) and not AMBIGUOUS_EDIT.search(task) and len(HAN.findall(text)) >= 5:
        return task, text
    return None


def lexical_coverage(answer, material):
    source = set(re.findall(r"(?=([\u3400-\u9fff]{2}))", material))
    observed = re.findall(r"(?=([\u3400-\u9fff]{2}))", answer)
    return sum(phrase in source for phrase in observed) / len(observed) if observed else 0


def day_state(text):
    match = DAY_STATE.fullmatch(normalized_content(text))
    if not match:
        return None
    attributes = re.split(r"而且|而|并且|也|又|且", match["body"])
    if len(attributes) < 2 or any(attribute not in DAY_ATTRS for attribute in attributes):
        return None
    day = {"今日": "今天", "昨日": "昨天", "明日": "明天"}.get(match["day"], match["day"])
    return day, sorted(attributes)


def task_family(task, material, answer):
    clean = FORMAT_PREFIX.sub("", answer).strip().strip('“”「」"')
    if not clean or len(clean) > 240 or COUNT_CUE.search(task) or AMBIGUOUS_EDIT.search(task):
        return None
    if ONE_SENTENCE.search(task) and len(SENTENCE.findall(clean)) != 1:
        return None
    if ordinal := END_SENTENCE.search(task):
        sentences = list(sentence_spans(material))
        if EXACT_QUOTE.search(task) and len(sentences) >= 2:
            index = -1 if re.search(r"末|最后|结尾", ordinal.group()) else 0
            start, end = sentences[index]
            excerpt = material[start:end].strip().strip('“”「」"')
            if clean.rstrip("。！!？?") == excerpt.rstrip("。！!？?"):
                return "exact_requested_sentence"
    if SUMMARY.search(task) and len(clean) < len(material):
        if (meaning := day_state(material)) is not None and day_state(clean) == meaning:
            return "meaning_preserving_day_compaction"
    return None


def single_family(question, answer):
    if family := natural_family(question, answer):
        return "natural", family
    if parts := text_instruction(question):
        if family := task_family(*parts, answer):
            return "instruction", family
    return None


def independent_prefix(question, answer, single):
    if single or (FICTION.search(question) and 25 <= len(answer) <= 330 and re.search(r"[。！？!?]$", answer)):
        return True
    if BRAINSTORM.search(question) and len(answer) <= 230 and re.search(r"^(?:可以|不妨|试试|试着)", answer):
        return bool(anchors(question, answer, min_chars=3) or ("健康" in question and re.search(r"运动|锻炼|健身", answer)))
    return False


def dependency(previous_answer, question, answer):
    if EMPTY_ACK.fullmatch(answer) or len(question) > 115:
        return None
    if change := UPDATE.search(question) or CORRECT_VALUE.fullmatch(question):
        old, new = change.group("old", "new")
        if (old != new and previous_answer.count(old) == 1
                and normalized_content(FORMAT_PREFIX.sub("", answer)) == normalized_content(previous_answer.replace(old, new))):
            return "history_constraint_update", {"old": old, "new": new}
    if not BACKREF.search(question) and not OBJECT_REF.search(question):
        return None
    if USER_TEXT.search(question) or EXACT_QUOTE.search(question):
        if family := task_family(question, previous_answer, answer):
            return f"history_{family}", {"copied_or_preserved_value": answer}
    if (object_ref := OBJECT_REF.search(question)) and SPECIFIES.search(question) and len(answer) <= 230:
        target = object_ref["object"]
        carries = [span for span in anchors(previous_answer, answer, min_chars=4) if span not in question]
        if (target.casefold() in previous_answer.casefold() and carries and FEATURE_PAYLOAD.search(answer)
                and lexical_coverage(answer, previous_answer) < .65 and not re.search(r"^\s*(?:我不知道|请提供|不能确定)", answer)):
            return "history_object_specifics", {"object": target, "reused_details": carries[:3]}
    return None


def make_candidate(tokenizer, source, messages, category, family, transform):
    try:
        encoded = encode_conversation(tokenizer, messages, final_eos=False)
    except ValueError:
        return None, "invalid_rendered_conversation"
    size = len(encoded["input_ids"])
    if size > MAX_TOKENS:
        return None, "over_512_tokens"
    locator = json.dumps(source.provenance, ensure_ascii=False, sort_keys=True)
    return {
        "id": f"v4-{source.name}-{digest(locator)}", "source": source.name,
        "source_group_id": source.group_id, "group_id": source.group_id,
        "category": category, "task_family": family, "messages": messages,
        "source_record": source.provenance, "original": source.original, "transform": transform,
        "token_count": size, "supervised_tokens": sum(label != IGNORE_LABEL for label in encoded["labels"]),
    }, None


def dialogue_candidate(tokenizer, source):
    original = source.original.get("messages")
    if not isinstance(original, list) or not original or len(original) % 2:
        return None, "invalid_source_conversation"
    if [m.get("role") for m in original] != ["user", "assistant"] * (len(original) // 2):
        return None, "invalid_source_conversation"
    question, answer = (m["content"] for m in original[:2])
    if not isinstance(question, str) or not isinstance(answer, str):
        return None, "invalid_source_conversation"
    grounded, grounding_reason = grounded_dialogue_candidate(tokenizer, source, question, answer)
    if grounded is not None:
        return grounded, None
    first = single_family(question, answer)
    if reason := pair_quality(question, answer, allow_action_parameters=bool(first and first[0] == "natural")):
        return None, reason
    if len(original) < 4 or not independent_prefix(question, answer, first):
        if not first:
            return None, grounding_reason or "no_admitted_first_exchange"
    multi_candidate = None
    checks = []
    if len(original) >= 4 and independent_prefix(question, answer, first):
        for index in range(2, len(original), 2):
            next_question, next_answer = (m["content"] for m in original[index:index + 2])
            if not isinstance(next_question, str) or not isinstance(next_answer, str):
                break
            own_family = single_family(next_question, next_answer)
            prior_text = "\n".join(message["content"] for message in original[:index])
            if pair_quality(next_question, next_answer, prior_text=prior_text,
                            allow_action_parameters=bool(own_family and own_family[0] == "natural")):
                break
            found = None
            for assistant_index in range(index - 1, 0, -2):
                if relation := dependency(original[assistant_index]["content"], next_question, next_answer):
                    found = assistant_index, relation
                    break
            if not found:
                # A short on-topic social acknowledgment may connect complete dependent turns.
                if own_family != ("natural", "greeting_or_thanks") or not re.match(r"^(?:谢谢|多谢|太感谢)", next_question):
                    break
                continue
            assistant_index, (family, evidence) = found
            checks.append({"prior_assistant_index": assistant_index, "dependent_user_index": index,
                           "family": family, "evidence": evidence})
            row, reason = make_candidate(tokenizer, source, original[:index + 2], "multi", family, {
                "type": "complete_prefix", "message_range_zero_based_half_open": [0, index + 2],
                "dependency_checks": list(checks), "original_message_count": len(original),
            })
            if row is None:
                break
            multi_candidate = row
    if multi_candidate is not None:
        return multi_candidate, None
    if not first:
        return None, grounding_reason or "no_verified_history_dependency"
    return make_candidate(tokenizer, source, original[:2], *first, {
        "type": "complete_prefix", "message_range_zero_based_half_open": [0, 2],
        "original_message_count": len(original),
    })


def wiki_text(text, *, start=0):
    """Resolve only literal render markers; record the raw codepoint edits."""
    edits = []

    def replace(match):
        replacement = match.group("plain") or ""
        edits.append({"chars": [start + match.start(), start + match.end()],
                      "old": match.group(), "replacement": replacement})
        return replacement

    rendered = WIKI_RENDERING.sub(replace, text)
    return (None if WIKI_UNRESOLVED.search(rendered) else rendered), edits


def relation_spec(question):
    question = QUESTION_FILLER.sub("", question).strip()
    if COMPLEX_QUESTION.search(question):
        return None
    matches = [(name, match) for name, regex in CMRC_SLOTS for match in regex.finditer(question)]
    if len(matches) != 1 or "？" in question or "?" in question:
        return None
    family, slot = matches[0]
    before, after = question[:slot.start()], question[slot.end():]
    title = re.match(r"《([^》]+)》", before)
    if title:
        subject, before = title.group(1), before[title.end():]
    else:
        subject = SUBJECT_END.split(before, maxsplit=1)[0]
        subject = RELATION_WRAPPER.sub("", subject.strip())
        before = before[len(subject):]
    key = normalized_content(subject)
    if not 2 <= len(key) <= 28 or re.match(r"^(?:这[篇部本个种份]|那[篇部本个种份]|该|其|他(?:的|$)|她(?:的|$)|本文|材料|文章|什么|哪一个)", subject):
        return None
    before = RELATION_WRAPPER.sub("", before.strip())
    after = RELATION_WRAPPER.sub("", after.strip())
    return family, subject, before, after, slot.group()


def relation_support(spec, sentence, local_offset, gold):
    family, subject, before, after, slot = spec
    if normalized_content(gold) in normalized_content(f"{subject}{before}{after}"):
        return None
    full_left, full_right = sentence[:local_offset], sentence[local_offset + len(gold):]
    left, right = CLAUSE_SPLIT.split(full_left)[-1], CLAUSE_SPLIT.split(full_right)[0]
    nearby = left + gold + right
    if not right.strip() and family == "date" and re.match(r"[，,]", full_right):
        # A leading date is often set off by a comma before its event.
        nearby += CLAUSE_SPLIT.split(full_right)[1]
    if family in ("reason", "purpose"):
        nearby = sentence
    if family == "count":
        unit = re.match(COUNT_UNIT, after)
        expected_unit = slot[1:] if slot.startswith("几") else (unit.group() if unit else None)
        if not expected_unit or not NUMBER_VALUE.fullmatch(gold.strip()):
            return None
        if not (gold.endswith(expected_unit) or right.lstrip().startswith(expected_unit)):
            return None
    elif family == "date":
        if not DATE_VALUE.fullmatch(DATE_WITH_PREDICATE.sub("", gold.strip())):
            return None
        if DATE_VALUE.search(left) or DATE_VALUE.search(right):
            return None
    elif family == "place":
        if not ENTITY_VALUE.fullmatch(gold.strip()) or not PLACE_BEFORE.search(left):
            return None
    elif family == "name":
        if not ENTITY_VALUE.fullmatch(gold.strip()) or not TITLE_BEFORE.search(left):
            return None
    elif family == "author":
        # Assignment or authorship must directly border the name span.
        if not ENTITY_VALUE.fullmatch(gold.strip()) or not AUTHOR_LEAD.search(left):
            return None
        if not (AUTHOR_TAIL.search(right) or re.search(r"作者(?:是|为)\s*$", left)):
            return None
    elif family == "reason":
        if len(gold) > 115 or not (CAUSE_LEAD.search(gold) or CAUSE_LEAD.search(full_left[-70:])):
            return None
        if not any(len(normalized_content(part)) >= 2 for part in (before, after)):
            return None
    elif family == "purpose":
        if len(gold) > 100 or not (re.search(r"(?:目的是|目的在于|旨在|用来|用于)", gold) or PURPOSE_LEAD.search(left)):
            return None
    comparison = normalized_content(nearby)
    parts = [normalized_content(part) for part in (before, after)]
    required = [part for part in parts if len(part) >= 2 or (family == "date" and part == "建")]
    if family == "date" and not required:
        return None
    if any(part not in comparison for part in required):
        return None
    return {"question_subject": subject, "relation": family, "required_relation_text": required}


def subject_supported(subject, evidence, passage):
    key = normalized_content(subject)
    if key in normalized_content(evidence):
        return "named_in_evidence"
    if key not in normalized_content(passage):
        return None
    if COREFERENCE.search(evidence):
        # Avoid a different overtly introduced person between the named subject and a personal pronoun.
        bridge = passage[:-len(evidence)]
        if re.search(r"[^。！？；]{2,30}(?:所生|所著|的妻子|的丈夫|的女儿|的儿子)[，。]", bridge):
            return None
        return "named_antecedent_in_contiguous_prefix"
    if INSTITUTION_CHAIN.search(subject) and re.match(r"\s*\d{4}年(?:更名|改名|改称)", evidence):
        return "named_institution_and_visible_rename_chain"
    return None


def grounded_dialogue_candidate(tokenizer, source, question, answer):
    """Admit actual document QA, independent of BELLE's cached labels."""
    match = MATERIAL_QUESTION.fullmatch(question)
    if not match:
        return None, None
    material, query = match.group("material", "question")
    if (not re.search(r"[？?]$", query) or len(HAN.findall(material)) < 18
            or any(contains_reserved(text) or ROLE.search(text) or RISK.search(text) or BAD_FORMAT.search(text)
                   for text in (question, answer))):
        return None, "invalid_supplied_material_or_question"
    gold = answer.strip().rstrip("。！!？?")
    offset = material.find(gold)
    spec = relation_spec(query)
    if not gold or offset < 0 or material.find(gold, offset + 1) >= 0 or not spec:
        return None, "no_unambiguous_answer_in_supplied_material"
    for start, end in sentence_spans(material):
        if not start <= offset < offset + len(gold) <= end:
            continue
        evidence = material[start:end]
        support = relation_support(spec, evidence, offset - start, gold)
        if support and (subject_link := subject_supported(spec[1], evidence, material[:end])):
            offset_in_prompt = match.start("material")
            return make_candidate(tokenizer, source, source.original["messages"][:2], "grounded", f"document_{spec[0]}", {
                "type": "complete_prefix_supplied_document", "message_range_zero_based_half_open": [0, 2],
                "original_message_count": len(source.original["messages"]),
                "material_chars_in_user_zero_based_half_open": [match.start("material"), match.end("material")],
                "answer_chars_in_user_zero_based_half_open": [offset_in_prompt + offset, offset_in_prompt + offset + len(gold)],
                "evidence_chars_in_user_zero_based_half_open": [offset_in_prompt + start, offset_in_prompt + end],
                "subject_link": subject_link, **support,
            })
    return None, "supplied_material_does_not_support_relation"


def cmrc_candidate(tokenizer, source):
    row = source.original
    question, context, annotations = row["question"], row["context"], row["answers"]
    if not question or not context or contains_reserved(question) or ROLE.search(question) or RISK.search(question):
        return None, "invalid_cmrc_text"
    spec = relation_spec(question)
    if not spec:
        return None, "cmrc_no_explicit_local_relation"
    texts, starts = annotations["text"], annotations["answer_start"]
    if len(texts) != len(starts) or not texts:
        return None, "invalid_cmrc_annotation"
    verified = Counter((gold, start) for gold, start in zip(texts, starts)
                       if isinstance(gold, str) and gold.strip() and type(start) is int and start >= 0
                       and context[start:start + len(gold)] == gold)
    if sum(verified.values()) != len(texts):
        return None, "invalid_cmrc_answer_offset"
    if len(verified) != 1:
        return None, "cmrc_annotation_disagreement"
    (gold, offset), = verified
    spans = list(sentence_spans(context))
    for index, (start, end) in enumerate(spans):
        if not start <= offset < offset + len(gold) <= end:
            continue
        evidence, edits = wiki_text(context[start:end], start=start)
        if evidence is None or any(a < offset + len(gold) and b > offset for a, b in (edit["chars"] for edit in edits)):
            return None, "cmrc_unresolved_render_or_answer_edit"
        deleted_before_answer = sum((edit["chars"][1] - edit["chars"][0]) - len(edit["replacement"])
                                    for edit in edits if edit["chars"][1] <= offset)
        support = relation_support(spec, evidence, offset - start - deleted_before_answer, gold)
        if not support:
            return None, "cmrc_no_explicit_local_relation"
        subject_key = normalized_content(spec[1])
        candidate_starts = [start] if subject_key in normalized_content(evidence) else [
            first for first, stop in spans[:index] if subject_key in normalized_content(context[first:stop])
        ]
        for passage_start in candidate_starts:
            passage, visible_edits = wiki_text(context[passage_start:end], start=passage_start)
            if passage is None:
                continue
            if (contains_reserved(passage) or ROLE.search(passage) or RISK.search(passage)
                    or "\ufffd" in passage):
                continue
            subject_link = subject_supported(spec[1], evidence, passage)
            if not subject_link:
                continue
            messages = [msg("user", f"请根据材料回答问题。\n\n材料：{passage}\n\n问题：{question}"), msg("assistant", gold)]
            row, reason = make_candidate(tokenizer, source, messages, "grounded", f"cmrc_{spec[0]}", {
                "type": "visible_contiguous_source_sentences",
                "passage_chars_zero_based_half_open": [passage_start, end],
                "answer_chars_zero_based_half_open": [offset, offset + len(gold)],
                "evidence_chars_zero_based_half_open": [start, end],
                "wiki_render_edits": visible_edits, "subject_link": subject_link, **support,
            })
            if row is not None:
                return row, None
            if reason != "over_512_tokens":
                return None, reason
        return None, "cmrc_no_safe_complete_window_within_512"
    return None, "cmrc_answer_crosses_sentence"
