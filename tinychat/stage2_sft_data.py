"""Bounded, deterministic preparation of Chinese Stage 2 conversations."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import blake2b, sha256
import json
import math
from pathlib import Path
from random import Random
import re
import sys
import time
import unicodedata
from uuid import uuid4

from .chat_format import IGNORE_LABEL, SPECIAL_IDS, check_special_ids, contains_reserved, encode_conversation
from .stage2_runtime import ensure_disk_budget


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/stage1_v1/tokenizer.json"
EVERYDAY = ROOT / "data/public_sources/natural_sft_v4/smoltalk-chinese/everyday.parquet"
CMRC_ROOT = ROOT / "data/public_sources/natural_sft_v3/cmrc2018"
CMRC_FILE = "train-00000-of-00001.parquet"
CMRC_REVISION = "137f2c45a24275fb68f6961c4d357f46288886aa"
REPORTS = ROOT / "reports/stage2_500m"
OUTPUT = ROOT / "data/sft_stage2_natural"
RAW = OUTPUT / "sources"
SMOLTALK_REPO = "opencsg/smoltalk-chinese"
SMOLTALK_REVISION = "5edf5d0d7b06794755cf78b274a150a4efe1c9d4"
SMOLTALK_TERMS = "The data card requires the OpenCSG Community License; commercial use requires permission."
SMOLTALK_FILES = {
    "information-seeking": (25_153_384, "ea7f9947804c3146348394177ce1298912ea3056f129a2ccbc25d5ce498e4b40"),
    "creative-writing": (72_754_149, "8d07d04b490415c172b1e58a51bdc1dafcb842f13f39ce62921b9756a0f1b702"),
    "editing": (76_655_129, "105be65841999f5ae518abecc75d6221ccb0f14b793b781aea2c358b61d2ac75"),
    "summary": (10_539_315, "19148c19f502c01912e5a63d980b1ea83a997b2feb8047b44125d1c9eb7426a6"),
}
SOURCE_TARGETS = {"everyday.parquet": 6_000, "editing.parquet": 7_000, "summary.parquet": 3_000,
                  CMRC_FILE: 3_000, "creative-writing.parquet": 1_000, "information-seeking.parquet": 1_000}
PREPARATION_SCAN_LIMITS = {"editing.parquet": 22_000, "creative-writing.parquet": 4_000, "summary.parquet": 7_500}
SPLIT_COUNTS = {"train": 20_000, "validation": 500, "test": 500}
MAX_SEQ = 512
SEED = 381924
HAN = re.compile(r"[\u3400-\u9fff\U00020000-\U0002fa1f]")
OTHER_SCRIPT = re.compile(r"[\u0400-\u052f\u0590-\u08ff\u0e00-\u0e7f\u3040-\u30ff\uac00-\ud7af]")
NOISE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]|<\|[^\n]{1,48}\|>|</?think>|\[/?INST\]", re.I)
FILLER = re.compile(r"^(?:当然可以|当然|好的|没问题|好呀|是的|不是|明白了|收到|谢谢|不客气|很高兴帮你)[，,。！!、？?\s]*$")
UNRELIABLE = re.compile(
    r"作为(?:一名|一个)?(?:AI|人工智能|语言模型|医生|律师|投资顾问)|"
    r"我(?:曾经|上次|昨天|前几天|有个朋友|在家里|前阵子|刚刚)(?:亲自|自己)?(?:看了|去过|去了|买了|吃了|读了|用过|参加|遇到|做了|试了)|"
    r"我(?:已经|刚刚)帮你(?:完成|预订|发送|购买|查询|联系|打电话)|"
    r"(?:根据|据)(?:最新|近期|一项|某项)?(?:研究|调查|统计)(?:表明|显示|发现|数据)|"
    r"(?:202[0-9]|201[0-9])年.{0,10}(?:最新|目前|上周|上个月|发布|上映)|"
    r"(?:信息|上文|原文)(?:没有提供|未提供).{0,14}(?:我猜|可能是)", re.I
)
HIGH_STAKES = re.compile(
    r"处方药|用药剂量|吃什么药|用什么药|药物剂量|过量服药|确诊|诊断结果|手术方案|孕期用药|自杀|自残|"
    r"投资荐股|推荐股票|选哪只股票|股票代码|杠杆率|内幕交易|法律意见|诉讼策略|怎么起诉|规避监管|"
    r"制作炸弹|制造炸弹|制毒|窃取密码|勒索软件|信用卡号|身份证号|真实手机号|儿童色情"
)
SYMPTOM = re.compile(r"感冒|高烧|发烧|发热|胸痛|腹痛|呼吸困难|头晕|哮喘|过敏症状|急性|伤口|扭伤|焦虑症|抑郁症")
SELF_TREATMENT = re.compile(r"(?:我|自己).{0,16}(?:能不能|要不要|是不是|怎么办|该不该|应该|可以|需要).{0,16}(?:去健身|运动|锻炼|吃|喝|用药|就医|治疗)|(?:应该|需要)(?:怎么|如何)(?:治疗|用药|处理)")
USER_ROLE_SWAP = re.compile(r"^(?:[您你]好[，,。！!\s]*)?(?:在这个角色[下中][，,。\s]*)?(?:我将尽我所能|我(?:可以|愿意|能|将|会)为[您你](?:提供|解答)|[您你](?:有什么|需要什么)(?:问题|帮助)|请[您你]告诉我[，,。\s]*[您你](?:需要|想要))")
FOLLOWUP = re.compile(r"刚才|上面|前面|你(?:提到|刚说|所说的|建议的)|那(?:个|样|种)|这(?:个|些|样|类)|其中|继续(?:说|聊|讲|补充)|具体点|第[二三四]个")
GENERIC_FOLLOWUP = frozenset("你觉得 你知道 你认为 能不能 有没有 怎么办 怎么做 怎么样 可以说 有什么 一点吗 吗我想 我想问 我还想 我最近 你可以 以告诉 告诉我 什么是 谢谢你 很有趣 谢谢您 帮助我 推荐一".split())
GROUNDING = re.compile(r"根据(?:以下|下[列文述]|这[篇段份个]|所给|原文|材料|文章|上述)|阅读(?:下面|以下)|(?:文章|材料|文本|原文)(?:中|提到)|^文本[:：]")
WRITING = re.compile(r"(?:帮我|请|替我|给我|需要|想|如何|怎么).{0,12}(?:写|润色|改写|拟|起草)|(?:写|拟)(?:一封|一份|一个|一段|一则)")
PRACTICAL_WRITING = re.compile(r"通知|贺卡|邮件|回复|邀请|建议书|推荐信|信件|工作报告|周报|沟通|自我介绍|发言稿|演讲稿|标题|宣传|活动方案|宣传稿|申请|感谢信|简历|请假|文案|短信|演示稿|道歉|说明文|说明书|笔记|广告|润色|改写|介绍信|新闻稿|招募|留言|社交媒体")
FICTION = re.compile(r"小说|剧本|童话|奇幻故事|科幻故事|诗歌|一首诗|虚构故事")
PRACTICAL_REQUEST = re.compile(r"怎么|如何|怎样|该不该|能不能|有没有|有什么(?:好办法|建议|方法|技巧)|帮我|给我|建议|请问能|请问可")
PRACTICAL_TOPIC = re.compile(r"收纳|整理|清理|家务|做饭|沟通|协作|时间管理|做笔记|学习计划|复习|准备面试|写简历|安排(?:工作|时间|会议)|记账|规划(?:一日|周末|行程)|制定计划|制作手工|养花|环保|节约|锻炼习惯|读书|工作流程|桌面|文件夹|厨房|做菜|人际关系|聚会|打招呼|交朋友|生活习惯")
EXTERNAL_FACT = re.compile(r"现任|最新|最近.{0,8}(?:上市|发布|上映|热销|流行)|截至|历史上|距地球最|世界最|全球最|最远|最大|最早|首次|第一位|第[0-9一二三四五六七八九十]+任|谁(?:是|发明|发现|发起|创立)|是谁|哪一年|什么时候.{0,6}(?:成立|发生|发明|发现|出生|成为)|哪[个座]国家|规定的比例|科学原理|^什么是|(?:1[89]|20)[0-9]{2}\s*年")
NUMBER_TEXT = r"([0-9]{1,3}|[零〇一二两三四五六七八九十百]{1,5})"
SENTENCE_UNIT = r"\s*(?:句话|句子|句(?![话子]))"
SENTENCE_BOUND = re.compile(r"(不少于|大于等于|不低于|至少|不超过|不多于|不高于|最多|至多|低于|少于|超过|多于)\s*" + NUMBER_TEXT + SENTENCE_UNIT)
SENTENCE_SUFFIX = re.compile(NUMBER_TEXT + SENTENCE_UNIT + r"\s*(以内|以下|以上)")
SENTENCE_RANGE = re.compile(NUMBER_TEXT + r"\s*(?:到|至|[-–—~～])\s*" + NUMBER_TEXT + SENTENCE_UNIT)
SENTENCE_EXACT = re.compile(r"(?:用|写|总结成|总结为|总结到|概括成|概括为|提炼为|提炼成|概括到|浓缩成|只需|改成)\s*" + NUMBER_TEXT + SENTENCE_UNIT + r"(?!\s*(?:以内|以下|以上|左右|上下))")
LATER_EDIT = re.compile(r"替换|修改|改写|改成|改动|改为|删掉|删除|避免|措辞|换个|润色|补充|减少")
REPEAT_REQUEST = re.compile(r"(?:请|再)原样|原样(?:重复|复述|输出)|一字不(?:改|差)|不要做任何改动")
PUNCTUATION_EDIT = re.compile(r"标点|感叹号|句号|逗号|冒号|引号|问号")
QUOTED_TEXT = re.compile(r'[“「『"]([^”」』"\n]{10,200})[”」』"]')


@dataclass(frozen=True)
class Source:
    path: Path
    repo: str
    revision: str
    file: str
    license: str
    terms: str
    card_url: str

    @property
    def name(self):
        return f"{self.repo}:{self.file}"


def load_source(path):
    path = path.resolve()
    if path == (CMRC_ROOT / "data" / CMRC_FILE).resolve():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        return Source(path, "hfl/cmrc2018", CMRC_REVISION, CMRC_FILE, "cc-by-sa-4.0",
                      "CMRC 2018 human span annotations on Chinese Wikipedia, original train split only.",
                      "https://ymcui.com/cmrc2018/")
    meta = json.loads(path.with_name(path.name + ".source.json").read_text(encoding="utf-8-sig"))
    if not re.fullmatch(r"[0-9a-f]{40}", meta["revision"]) or not meta.get("license_metadata"):
        raise ValueError(f"Missing pinned source revision or license: {path}")
    if meta["file"] != path.name or not path.is_file() or path.stat().st_size != meta["bytes"]:
        raise ValueError(f"Local source does not match recorded filename or size: {path}")
    return Source(path, meta["repo"], meta["revision"], meta["file"], meta["license_metadata"],
                  meta.get("additional_source_terms", ""), meta["card_url"])


def acquire_smoltalk(subset, directory=RAW):
    from huggingface_hub.utils import get_session
    from pyarrow.parquet import ParquetFile

    expected_bytes, expected_hash = SMOLTALK_FILES[subset]
    file = f"{subset}.parquet"
    url = f"https://huggingface.co/datasets/{SMOLTALK_REPO}/resolve/{SMOLTALK_REVISION}/{file}"
    destination = directory / file
    partial = directory / f"{file}.part"
    metadata = directory / f"{file}.source.json"
    if destination.exists() and metadata.exists():
        return load_source(destination)
    if any(path.exists() for path in (destination, partial, metadata)):
        raise FileExistsError(f"Source or partial already exists: {destination}")
    ensure_disk_budget(directory, required_bytes=expected_bytes)
    directory.mkdir(parents=True, exist_ok=True)
    sha = sha256()
    total = 0
    start = time.perf_counter()
    with get_session().stream("GET", url, headers={"Accept-Encoding": "identity"},
                              follow_redirects=True, timeout=30) as response:
        response.raise_for_status()
        length = response.headers.get("content-length")
        if (response.status_code != 200 or response.headers.get("content-encoding", "identity").lower() != "identity"
                or length is not None and int(length) != expected_bytes):
            raise ValueError(f"Unexpected response for pinned source: {file}")
        with partial.open("xb") as stream:
            for chunk in response.iter_raw(chunk_size=256 * 1024):
                total += len(chunk)
                if total > expected_bytes or time.perf_counter() - start > 300:
                    raise ValueError(f"Pinned transfer exceeded its byte or time limit: {file}")
                ensure_disk_budget(partial, required_bytes=len(chunk))
                stream.write(chunk)
                sha.update(chunk)
    if total != expected_bytes or sha.hexdigest() != expected_hash:
        raise ValueError(f"Downloaded data differs from pinned file: {file}")
    with ParquetFile(partial) as parquet:
        rows = parquet.metadata.num_rows
        columns = parquet.schema_arrow.names
    if rows < 1 or "conversations" not in columns:
        raise ValueError(f"Unexpected empty or incomplete source schema: {file}")
    meta = {"repo": SMOLTALK_REPO, "revision": SMOLTALK_REVISION, "file": file, "url": url,
            "bytes": total, "sha256": expected_hash, "rows": rows, "columns": columns,
            "license_metadata": "apache-2.0", "additional_source_terms": SMOLTALK_TERMS,
            "card_url": f"https://huggingface.co/datasets/{SMOLTALK_REPO}/blob/{SMOLTALK_REVISION}/README.md",
            "elapsed_seconds": round(time.perf_counter() - start, 3)}
    with metadata.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    partial.rename(destination)
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return load_source(destination)


def normalize(text, convert):
    return convert(unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))).strip()


def prompt_key(text):
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum())


def digest(*fields):
    return blake2b("\x1f".join(map(str, fields)).encode("utf-8"), digest_size=16).hexdigest()


def grams(key, size=4):
    return frozenset(key[index:index + size] for index in range(len(key) - size + 1))


class PromptIndex:
    """Exact normalization and thresholded set Jaccard using prefix filtering."""

    def __init__(self, threshold, frequencies=None):
        self.threshold = threshold
        self.frequencies = frequencies or {}
        self.exact = defaultdict(list)
        self.postings = defaultdict(list)
        self.features = {}

    def prefix(self, features):
        # The sorted prefixes of two qualifying sets must share a shingle.
        return sorted(features, key=lambda feature: (self.frequencies.get(feature, 0), feature))[
            :len(features) - math.ceil(self.threshold * len(features)) + 1]

    def add(self, key, identifier):
        self.exact[key].append(identifier)
        if len(key) < 12:
            return
        features = grams(key)
        self.features[identifier] = features
        for feature in self.prefix(features):
            self.postings[feature].append(identifier)

    def matches(self, key):
        exact = set(self.exact.get(key, ()))
        if len(key) < 12:
            return exact
        features = grams(key)
        candidates = {identifier for feature in self.prefix(features)
                      for identifier in self.postings.get(feature, ())}
        for identifier in candidates - exact:
            other = self.features[identifier]
            if min(len(features), len(other)) < self.threshold * max(len(features), len(other)):
                continue
            overlap = len(features & other)
            if overlap >= self.threshold * (len(features) + len(other) - overlap):
                exact.add(identifier)
        return exact


def evaluation_index(directory, convert):
    paths = sorted(path for path in directory.rglob("*.jsonl")
                   if path.name in {"prompts.jsonl", "prompts_for_overlap.jsonl"})
    index = PromptIndex(0.75)
    for path in paths:
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                row = json.loads(line)
                for message in row["messages"]:
                    if message["role"] == "user":
                        key = prompt_key(normalize(message["content"], convert))
                        if key and key not in index.exact:
                            index.add(key, len(index.exact))
    if not index.exact:
        raise ValueError(f"No evaluation user prompts found in {directory}")
    return index, paths


def role_messages(row):
    messages = row.get("conversations") or row.get("messages")
    if messages is not None:
        return messages
    for question, answer in (("instruction", "output"), ("prompt", "response"), ("question", "answer")):
        if isinstance(row.get(question), str) and isinstance(row.get(answer), str):
            user = row[question]
            if isinstance(row.get("input"), str) and row["input"].strip():
                user = f"{user}\n\n{row['input']}"
            return [{"role": "user", "content": user}, {"role": "assistant", "content": row[answer]}]
    return []


def clean_messages(raw, convert):
    if not isinstance(raw, list) or not raw:
        return None
    roles = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant", "system": "system"}
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        raw_role = item.get("role", item.get("from"))
        role = roles.get(raw_role) if isinstance(raw_role, str) else None
        content = item.get("content", item.get("value"))
        if role is None or not isinstance(content, str):
            return None
        text = normalize(content, convert)
        if not text or contains_reserved(text) or NOISE.search(text):
            return None
        cleaned.append({"role": role, "content": text})
    offset = int(cleaned[0]["role"] == "system")
    turns = len(cleaned) - offset
    if turns < 2 or turns % 2 or any(message["role"] != ("assistant" if index % 2 else "user")
                                      for index, message in enumerate(cleaned[offset:])):
        return None
    return cleaned


def looks_linked(user, history):
    shared = grams(prompt_key(user), 3) & grams(prompt_key(history), 3)
    return bool(FOLLOWUP.search(user) or shared - GENERIC_FOLLOWUP)


def sentence_number(text):
    if text.isdecimal():
        return int(text)
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    total = 0
    digit = None
    for character in text:
        if character in "十百":
            total += (1 if digit is None else digit) * (10 if character == "十" else 100)
            digit = None
        else:
            digit = digits[character]
    return total + (digit or 0)


def sentence_constraint_met(user, answer):
    bounds = []
    for op, raw in SENTENCE_BOUND.findall(user):
        number = sentence_number(raw)
        if op in {"不少于", "大于等于", "不低于", "至少"}:
            bounds.append((number, math.inf))
        elif op in {"超过", "多于"}:
            bounds.append((number + 1, math.inf))
        else:
            bounds.append((0, number - int(op in {"低于", "少于"})))
    for raw, op in SENTENCE_SUFFIX.findall(user):
        number = sentence_number(raw)
        bounds.append((number, math.inf) if op == "以上" else (0, number))
    bounds.extend((sentence_number(low), sentence_number(high)) for low, high in SENTENCE_RANGE.findall(user))
    for raw in SENTENCE_EXACT.findall(user):
        number = sentence_number(raw)
        bounds.append((number, number))
    if not bounds:
        return True
    sentences = re.split(r"[。！？!?]+|\.(?=\s|$)", answer)
    count = sum(bool(HAN.search(part) or re.search(r"[A-Za-z]{2,}", part)) for part in sentences)
    return all(low <= count <= high for low, high in bounds)


def edit_progresses(user, previous, answer):
    if not LATER_EDIT.search(user) or REPEAT_REQUEST.search(user):
        return True
    if answer == previous:
        return False
    before, after = prompt_key(previous), prompt_key(answer)
    if before == after:
        return bool(PUNCTUATION_EDIT.search(user))
    anchors = [prompt_key(quote) for quote in QUOTED_TEXT.findall(user)]
    if (min(len(before), len(after)) < .97 * max(len(before), len(after))
            or not any(len(anchor) >= 10 and anchor in before and anchor in after for anchor in anchors)):
        return True
    return SequenceMatcher(None, before, after, autojunk=False).ratio() < .97


def practical_without_reference(user, answer):
    return bool(PRACTICAL_REQUEST.search(user) and PRACTICAL_TOPIC.search(user)
                and not EXTERNAL_FACT.search(user) and not EXTERNAL_FACT.search(answer))


def answer_issue(user, answer, grounded=False):
    user_han, answer_han = len(HAN.findall(user)), len(HAN.findall(answer))
    if user_han < 3 or answer_han < (2 if grounded else 8):
        return "non_chinese_or_short"
    if (user_han < len(prompt_key(user)) * 0.52 or answer_han < len(prompt_key(answer)) * 0.52
            or OTHER_SCRIPT.search(user) or OTHER_SCRIPT.search(answer)):
        return "non_chinese_or_short"
    if USER_ROLE_SWAP.search(re.sub(r"\s+", "", user)):
        return "user_role_swap"
    if FILLER.fullmatch(answer) or prompt_key(user) == prompt_key(answer):
        return "empty_or_echo"
    if (HIGH_STAKES.search(user) or HIGH_STAKES.search(answer)
            or not grounded and SYMPTOM.search(user) and SELF_TREATMENT.search(user)):
        return "high_stakes"
    if UNRELIABLE.search(answer):
        return "unsupported_claim_marker"
    if grounded and not set(re.findall(r"[0-9]+(?:[.,][0-9]+)*", answer)).issubset(
            re.findall(r"[0-9]+(?:[.,][0-9]+)*", user)):
        return "unsupported_number"
    if answer.endswith(("…", "...", ":", "：")) or answer.count("```") % 2:
        return "incomplete_answer_marker"
    if not sentence_constraint_met(user, answer):
        return "sentence_constraint"
    return None


def select_cmrc(row, source, row_number, tokenizer, convert, evaluation):
    record_id, raw_context, raw_question = row["id"], row["context"], row["question"]
    annotations = row["answers"]
    texts, offsets = annotations["text"], annotations["answer_start"]
    if (not texts or len(texts) != len(offsets) or len(set(texts)) != 1
            or any(type(start) is not int or start < 0 or raw_context[start:start + len(text)] != text
                   for text, start in zip(texts, offsets))):
        return None, "annotation_offset"
    context, question, answer = (normalize(text, convert) for text in (raw_context, raw_question, texts[0]))
    if (not answer or answer not in context or not question
            or any(contains_reserved(text) or NOISE.search(text) or HIGH_STAKES.search(text)
                   for text in (context, question, answer))):
        return None, "invalid_cmrc_span"
    user = f"材料：{context}\n\n问题：{question}"
    if evaluation.matches(prompt_key(question)) or evaluation.matches(prompt_key(user)):
        return None, "evaluation_overlap"
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    encoded = encode_conversation(tokenizer, messages, final_eos=False)
    if len(encoded["input_ids"]) > MAX_SEQ:
        return None, "over_context"
    document = record_id.rsplit("_QUERY_", 1)[0]
    return {"id": digest(source.name, source.revision, row_number), "source": source.name,
            "source_license": source.license, "source_revision": source.revision, "source_row": row_number,
            "source_group": digest(source.repo, source.revision, record_id),
            "split_group": digest(source.repo, "document", context), "category": "grounded",
            "source_metadata": {"source_record_id": record_id, "source_document_id": document,
                                "answer_offsets_raw": offsets}, "dedup_prompt": question,
            "turns": 1, "messages": messages, **encoded}, None


def select_conversation(row, source, row_number, tokenizer, convert, evaluation):
    if source.repo == "hfl/cmrc2018":
        return select_cmrc(row, source, row_number, tokenizer, convert, evaluation)
    messages = clean_messages(role_messages(row), convert)
    if not messages:
        return None, "invalid_messages"
    offset = int(messages[0]["role"] == "system")
    users = [message["content"] for message in messages[offset::2]]
    if any(evaluation.matches(prompt_key(user)) for user in users):
        return None, "evaluation_overlap"

    is_everyday = source.file == "everyday.parquet"
    is_information = source.file == "information-seeking.parquet"
    is_editing = source.file == "editing.parquet"
    grounded = bool(GROUNDING.search(users[0])) or source.file in {"document-qa.parquet", "summary.parquet"}
    if grounded and len(users[0]) < 80:
        return None, "missing_grounding_context"
    best = None
    issue = "no_complete_prefix"
    for index, user in enumerate(users):
        pair = offset + 2 * index
        answer = messages[pair + 1]["content"]
        issue = answer_issue(user, answer, grounded=grounded and index == 0)
        if issue:
            break
        if is_information and (not index and not practical_without_reference(user, answer)
                               or EXTERNAL_FACT.search(user) or EXTERNAL_FACT.search(answer)):
            issue = "unsupported_open_domain"
            break
        if is_everyday and (EXTERNAL_FACT.search(user) or EXTERNAL_FACT.search(answer)):
            issue = "unsupported_open_domain"
            break
        if is_editing and index and not edit_progresses(user, messages[pair - 1]["content"], answer):
            issue = "no_progress_on_revision"
            break
        if index and not looks_linked(user, "\n".join(message["content"] for message in messages[offset:pair])):
            issue = "unlinked_followup"
            break
        prefix = messages[:pair + 2]
        encoded = encode_conversation(tokenizer, prefix, final_eos=False)
        if len(encoded["input_ids"]) > MAX_SEQ:
            issue = "over_context"
            break
        best = (prefix, encoded, index + 1)
    if best is None:
        return None, issue or "no_complete_prefix"

    selected, encoded, turns = best
    source_id = row.get("id") or row.get("conversation_id") or digest(*(m["content"] for m in messages))
    source_group = digest(source.repo, source.revision, source_id)
    category = ("multiturn" if is_everyday and turns > 1 else "singleturn_natural" if is_everyday
                else "writing" if source.file in {"editing.parquet", "creative-writing.parquet"}
                else "practical_qa" if is_information
                else "grounded" if grounded else "writing" if WRITING.search(users[0])
                else "question_explanation")
    metadata = {name: row[name] for name in ("score", "difficulty", "classify", "magpie_model", "task_category")
                if row.get(name) is not None}
    if turns < len(users) and issue:
        metadata["prefix_stop_reason"] = issue
    if category == "writing":
        metadata["practical_writing"] = bool(PRACTICAL_WRITING.search(users[0]) and not FICTION.search(users[0]))
    return {"id": digest(source.name, source.revision, row_number), "source": source.name,
            "source_license": source.license, "source_revision": source.revision, "source_row": row_number,
            "source_group": source_group, "split_group": source_group, "category": category, "source_metadata": metadata,
            "turns": turns, "messages": selected, **encoded}, None


def parquet_rows(source, limit=None):
    from pyarrow.parquet import ParquetFile

    reader = ParquetFile(source.path)
    available = reader.metadata.num_rows
    positions = None
    if limit is not None and limit < available:
        positions = set(Random(digest(SEED, source.name)).sample(range(available), limit))
    row_number = 0
    for batch in reader.iter_batches(batch_size=512):
        if positions is None:
            for row in batch.to_pylist():
                yield row_number, row
                row_number += 1
            continue
        indices = [index for index in range(len(batch)) if row_number + index in positions]
        for index, row in zip(indices, batch.take(indices).to_pylist()) if indices else ():
            yield row_number + index, row
        row_number += len(batch)


def deduplicate(candidates, excluded=frozenset()):
    keys = [prompt_key(candidate.get("dedup_prompt") or next(message["content"] for message in candidate["messages"]
                                                           if message["role"] == "user"))
            for candidate in candidates]
    distinct_keys = list(dict.fromkeys(keys))
    sample = Random(SEED).sample(distinct_keys, min(2_000, len(distinct_keys)))
    frequencies = Counter(gram for key in sample for gram in grams(key))
    index = PromptIndex(0.80, frequencies)
    groups = {}
    roots = []

    def root(identifier):
        while roots[identifier] != identifier:
            roots[identifier] = roots[roots[identifier]]
            identifier = roots[identifier]
        return identifier

    for identifier, (candidate, key) in enumerate(zip(candidates, keys)):
        roots.append(identifier)
        related = index.matches(key)
        if candidate["source_group"] in groups:
            related.add(groups[candidate["source_group"]])
        groups[candidate["source_group"]] = identifier
        for neighbor in related:
            roots[root(neighbor)] = root(identifier)
        if key not in index.exact:
            index.add(key, identifier)

    clusters = defaultdict(list)
    for identifier, candidate in enumerate(candidates):
        clusters[root(identifier)].append(candidate)
    representatives = []
    for cluster in clusters.values():
        if any(item["id"] in excluded for item in cluster):
            continue

        def preference(item):
            score = item.get("source_metadata", {}).get("score")
            return -item["turns"], -(score if type(score) is int else 0), digest(SEED, item["id"])

        preferred = min(cluster, key=preference)
        representatives.append({**preferred, "cluster_id": min(item["id"] for item in cluster)})
    return representatives


def collect(sources, tokenizer, convert, evaluation, limit, excluded=frozenset()):
    from pyarrow.parquet import ParquetFile

    candidates = []
    stats = {}
    for source in sources:
        start = time.perf_counter()
        counts = Counter()
        scan_limit = PREPARATION_SCAN_LIMITS.get(source.file) if limit is None else limit
        for position, row in parquet_rows(source, scan_limit):
            counts["scanned"] += 1
            if limit is None and counts["scanned"] % 5_000 == 0:
                print(json.dumps({"source": source.name, "scanned": counts["scanned"],
                                  "passed_rules": counts["passed_rules"]}), file=sys.stderr, flush=True)
            result, issue = select_conversation(row, source, position, tokenizer, convert, evaluation)
            if result is not None and result["id"] in excluded:
                counts["independently_excluded"] += 1
            else:
                counts[issue or "passed_rules"] += 1
            if result is not None:
                candidates.append(result)
                if (reason := result["source_metadata"].get("prefix_stop_reason")):
                    counts[f"dropped_suffix_{reason}"] += 1
        stats[source.name] = {"available": ParquetFile(source.path).metadata.num_rows, "counts": dict(counts),
                              "seconds": round(time.perf_counter() - start, 3), "bytes": source.path.stat().st_size,
                              "scan_limit": scan_limit, "raw_file": str(source.path),
                              "revision": source.revision, "license": source.license, "terms": source.terms,
                              "card_url": source.card_url}
        if source.repo == "hfl/cmrc2018":
            stats[source.name]["license_record"] = str(CMRC_ROOT / "README.md")
    if limit is None:
        print(json.dumps({"status": "deduplicating", "candidate_count": len(candidates)}), file=sys.stderr, flush=True)
    representatives = deduplicate(candidates, excluded)
    for source in sources:
        kept = sum(item["source"] == source.name for item in representatives)
        source_stats = stats[source.name]
        source_stats["kept"] = kept
        source_stats["practical_writing_kept"] = sum(item["source"] == source.name
                                                     and item["source_metadata"].get("practical_writing", False)
                                                     for item in representatives)
        source_stats["estimated_yield_before_full_dedup"] = round(
            source_stats["available"] * kept / max(1, source_stats["counts"].get("scanned", 0)))
    return representatives, stats


def proportional_slots(sizes, size):
    available = sum(sizes.values())
    if not 0 <= size <= available:
        raise ValueError(f"Requested {size} samples from {available} available records")
    targets = {key: count * size / available for key, count in sizes.items()} if available else {}
    slots = {key: math.floor(target) for key, target in targets.items()}
    order = sorted(sizes, key=lambda key: (slots[key] - targets[key], key))
    for key in order[:size - sum(slots.values())]:
        slots[key] += 1
    return slots


def stratified_sample(rows, size, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"], row["category"]].append(row)
    slots = proportional_slots({key: len(group) for key, group in groups.items()}, size)
    sample = []
    for key, group in groups.items():
        sample.extend(sorted(group, key=lambda row: digest(seed, row["id"]))[:slots[key]])
    return sorted(sample, key=lambda row: digest("order", seed, row["id"]))


def jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def write_audit(rows, report_dir, stem, seed):
    blind, key = [], []
    for row in rows:
        blind_id = digest("blind", seed, row["id"])
        blind.append({"blind_id": blind_id, "messages": row["messages"]})
        key.append({"blind_id": blind_id, "id": row["id"], "source": row["source"], "source_row": row["source_row"],
                    "source_group": row["source_group"], "split_group": row["split_group"],
                    "category": row["category"], "tokens": len(row["input_ids"]), "split": row.get("split")})
    jsonl(report_dir / f"sft_{stem}_blind.jsonl", blind)
    jsonl(report_dir / f"sft_{stem}_key.jsonl", key)


def mixture_pools(rows):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"].rsplit(":", 1)[-1]].append(row)
    by_document = defaultdict(list)
    for row in by_source[CMRC_FILE]:
        by_document[row["split_group"]].append(row)
    by_source[CMRC_FILE] = [row for group in by_document.values()
                           for row in sorted(group, key=lambda item: digest("document", SEED, item["id"]))[:2]]
    return by_source


def mixture_quotas(by_source, size):
    caps = {name: len(by_source[name]) for name in SOURCE_TARGETS}
    if caps["everyday.parquet"] < math.ceil(size / 4):
        raise ValueError(f"Natural conversations limited to {caps['everyday.parquet']}; need at least {math.ceil(size / 4)}")
    caps["summary.parquet"] = min(caps["summary.parquet"], size // 5)
    caps["creative-writing.parquet"] = min(caps["creative-writing.parquet"], size // 20)
    caps["editing.parquet"] = min(caps["editing.parquet"], size * 35 // 100)
    caps["information-seeking.parquet"] = min(caps["information-seeking.parquet"], math.ceil(size / 21))
    target = {name: size * requested / sum(SOURCE_TARGETS.values()) for name, requested in SOURCE_TARGETS.items()}
    quotas = {name: min(caps[name], math.floor(desired)) for name, desired in target.items()}
    evidence = {"summary.parquet", CMRC_FILE}
    for _ in range(size - sum(quotas.values())):
        evidence_count = sum(quotas[name] for name in evidence)
        options = [name for name in quotas if quotas[name] < caps[name]
                   and (name not in evidence or evidence_count < size * 35 // 100)]
        if not options:
            raise ValueError(f"Need {size} in-scope conversations within proportion limits; capped capacity: {caps}")
        if evidence_count >= size * 30 // 100:
            options = [name for name in options if name not in evidence] or options
        name = max(options, key=lambda name: (target[name] - quotas[name], name))
        quotas[name] += 1
    return quotas


def select_mixture(rows, size):
    by_source = mixture_pools(rows)
    quotas = mixture_quotas(by_source, size)
    selected = []
    for name, quantity in quotas.items():
        if not quantity:
            continue
        pool = by_source[name]
        if name not in {"creative-writing.parquet", "editing.parquet"}:
            selected.extend(stratified_sample(pool, quantity, f"mix-{SEED}"))
            continue
        practical = [row for row in pool if row["source_metadata"].get("practical_writing")]
        others = [row for row in pool if not row["source_metadata"].get("practical_writing")]
        count = min(len(practical), max(round(0.6 * quantity), quantity - len(others)))
        if count:
            selected.extend(stratified_sample(practical, count, f"practical-{SEED}"))
        if quantity - count:
            selected.extend(stratified_sample(others, quantity - count, f"creative-{SEED}"))
    return selected


def grouped_holdout(rows, size, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("split_group", row["id"])].append(row)
    strata = defaultdict(list)
    for group_id, group in groups.items():
        kind = group[0]["source"], group[0]["category"]
        if any((row["source"], row["category"]) != kind for row in group):
            raise ValueError(f"Split group crosses strata: {group_id}")
        strata[kind].append((group_id, group))
    slots = proportional_slots({key: sum(len(group) for _, group in groups) for key, groups in strata.items()}, size)
    held_out = []
    spare = []
    for kind, groups in strata.items():
        available = slots[kind]
        for group_id, group in sorted(groups, key=lambda pair: digest(seed, pair[0])):
            if len(group) <= available:
                held_out.extend(group)
                available -= len(group)
            else:
                spare.append((group_id, group))
    for group_id, group in sorted(spare, key=lambda pair: digest("fill", seed, pair[0])):
        if len(held_out) == size:
            break
        if len(group) <= size - len(held_out):
            held_out.extend(group)
    if len(held_out) != size:
        raise ValueError(f"Cannot allocate an intact document holdout of {size} records")
    return held_out


def assign_splits(rows, counts):
    remaining = rows
    splits = {}
    for split in ("validation", "test"):
        held_out = grouped_holdout(remaining, counts[split], f"{split}-{SEED}")
        held_ids = {row["id"] for row in held_out}
        splits[split] = held_out
        remaining = [row for row in remaining if row["id"] not in held_ids]
    if len(remaining) != counts["train"]:
        raise ValueError("Train count does not match the requested split size")
    splits["train"] = remaining
    return splits


def file_identity(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prior_review_exclusions(review_path, key_path):
    review = json.loads(review_path.read_text(encoding="utf-8-sig"))["review"]
    covered = set(review["covered_ids"])
    flagged = {item["blind_id"] for item in review["issues"] if item["severity"] in {"MAJOR", "VERIFY"}}
    if not covered or not flagged <= covered:
        raise ValueError("Prior review is empty or flags IDs outside its coverage")
    with key_path.open(encoding="utf-8-sig") as stream:
        review_ids = {row["blind_id"]: row["id"] for row in map(json.loads, stream) if row["blind_id"] in covered}
    if covered != set(review_ids):
        raise ValueError("Prior review and blind mapping cover different IDs")
    row_ids = set(review_ids.values())
    groups = {}
    for split in SPLIT_COUNTS:
        with (OUTPUT / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for row in map(json.loads, stream):
                if row["id"] in row_ids:
                    groups[row["id"]] = row["source_group"]
    if set(groups) != row_ids:
        raise ValueError("Prior reviewed rows were not found in the current candidate")
    return {"prior_review": str(review_path.resolve()), "ids": sorted(row_ids),
            "source_groups": sorted(set(groups.values())), "flagged_ids": sorted(review_ids[item] for item in flagged)}


def supervision_summary(rows):
    all_targets = 0
    end_turn = 0
    end_sequence = 0
    content_lengths = []
    by_source = Counter()
    for row in rows:
        labels = row["labels"][1:]
        turns = labels.count(SPECIAL_IDS["<|end_turn|>"])
        eos = labels.count(SPECIAL_IDS["<|eos|>"])
        targets = len(labels) - labels.count(IGNORE_LABEL)
        all_targets += targets
        end_turn += turns
        end_sequence += eos
        content_lengths.append(targets - turns - eos)
        by_source[row["source"]] += targets
    ordered = sorted(content_lengths)
    percentiles = {f"p{p}": ordered[math.ceil(len(ordered) * p / 100) - 1] for p in (10, 25, 50, 75, 90, 100)}
    token_share = {name: round(count / all_targets, 6) for name, count in by_source.items()}
    return {"assistant_tokens": all_targets, "end_turn_tokens": end_turn, "eos_tokens": end_sequence,
            "end_turn_fraction": round(end_turn / all_targets, 6), "eos_fraction": round(end_sequence / all_targets, 6),
            "assistant_content_tokens_per_record": percentiles, "content_at_most_32_count": sum(n <= 32 for n in ordered),
            "source_assistant_token_share": token_share}


def write_candidate(splits, stats, tokenizer, evaluation, paths, audit_name, review_exclusions):
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["frozen"]:
        raise FileExistsError(f"Frozen data already exists at {path}")
    ensure_disk_budget(OUTPUT, required_bytes=512 * 1024 ** 2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    tokenizer_hash = sha256(TOKENIZER.read_bytes()).hexdigest()
    stage1_manifest = json.loads(TOKENIZER.with_name("manifest.json").read_text(encoding="utf-8"))
    if stage1_manifest["tokenizer_sha256"] != tokenizer_hash or stage1_manifest["vocab_size"] != tokenizer.get_vocab_size():
        raise ValueError("Tokenizer differs from the frozen Stage 1 identity")
    excluded_ids = set(review_exclusions.get("ids", ()))
    excluded_groups = set(review_exclusions.get("source_groups", ()))
    for_audit = [{**row, "split": split} for split, selected in splits.items() for row in selected
                 if row["id"] not in excluded_ids and row["source_group"] not in excluded_groups]
    audit_seed = f"{audit_name}-{SEED}"
    audit = stratified_sample(for_audit, 250, audit_seed)
    sources = Counter()
    categories = Counter()
    practical_writing = 0
    split_meta = {}
    files = {"tokenizer.json": file_identity(TOKENIZER)}
    for split, selected in splits.items():
        sources.update(row["source"] for row in selected)
        categories.update(row["category"] for row in selected)
        practical_writing += sum(row["source_metadata"].get("practical_writing", False) for row in selected)
        supervision = supervision_summary(selected)
        split_meta[split] = {"documents": len(selected), "tokens": sum(len(row["input_ids"]) for row in selected),
                             "assistant_tokens": supervision["assistant_tokens"], "supervision": supervision,
                             "source_mix": dict(Counter(row["source"] for row in selected)),
                             "category_mix": dict(Counter(row["category"] for row in selected))}
        final = OUTPUT / f"{split}.jsonl"
        temporary = final.with_name(f"{final.name}.{uuid4().hex}.partial")
        jsonl(temporary, selected)
        temporary.replace(final)
        files[final.name] = file_identity(final)
    write_audit(audit, REPORTS, audit_name, audit_seed)
    manifest = {"schema_version": 1, "corpus_id": "sft_stage2_natural", "frozen": False,
                "training_ready": False, "tokenizer_path": str(TOKENIZER.resolve()), "tokenizer_sha256": tokenizer_hash,
                "vocab_size": tokenizer.get_vocab_size(), "special_token_ids": SPECIAL_IDS,
                "counts": {split: meta["documents"] for split, meta in split_meta.items()}, "splits": split_meta,
                "files": files, "chat_format": {"final_eos": False, "max_seq": MAX_SEQ}, "candidate_sources": stats,
                "source_mix": dict(sources), "category_mix": dict(categories),
                "practical_writing_marker_count": practical_writing, "seed": SEED,
                "normalization": "Unicode NFC, OpenCC t2s, trimmed message boundaries",
                "deduplication": {"source_group": True, "user_prompt_shingle": 4, "candidate_jaccard": 0.8,
                                  "evaluation_jaccard": 0.75, "holdouts": "Whole document groups stratified by source and category",
                                  "cmrc_document_cap": 2},
                "evaluation_prompt_count": len(evaluation.exact),
                "evaluation_paths": [str(item.relative_to(ROOT)) for item in paths],
                "quality_review": {"status": "pending", "count": len(audit), "excluded_prior": review_exclusions,
                                   "blind_file": str((REPORTS / f"sft_{audit_name}_blind.jsonl").resolve())}}
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.partial")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest


def main():
    from opencc import OpenCC
    from tokenizers import Tokenizer

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", help="Pinned local parquet and its .source.json sidecar")
    parser.add_argument("--acquire", choices=SMOLTALK_FILES, help="Download one exact pinned SmolTalk subset and exit")
    parser.add_argument("--probe-rows", type=int, default=500, help="Random raw records inspected per source")
    parser.add_argument("--probe-sample", type=int, default=100, help="Blind examples to hand to the independent reviewer")
    parser.add_argument("--probe-name", default="probe", help="Report name after the sft_ prefix")
    parser.add_argument("--prepare", action="store_true", help="Prepare the full candidate set pending independent review")
    parser.add_argument("--candidate-name", default="candidate", help="Report name after the sft_ prefix")
    parser.add_argument("--exclude-file", type=Path, help="JSON array of known failing candidate IDs to exclude")
    parser.add_argument("--prior-review", type=Path, help="Independent prior review used to quarantine flagged rows and isolate the new audit")
    args = parser.parse_args()
    if args.acquire:
        acquire_smoltalk(args.acquire)
        return
    if args.probe_rows < 1 or args.probe_sample < 1:
        parser.error("Probe limits must be positive")
    if not re.fullmatch(r"probe(?:_[a-z0-9]+)*", args.probe_name):
        parser.error("Probe name must begin with 'probe' and use lowercase underscore-separated words")
    if not re.fullmatch(r"candidate(?:_[a-z0-9]+)*", args.candidate_name):
        parser.error("Candidate name must begin with 'candidate' and use lowercase underscore-separated words")
    paths = args.source or [EVERYDAY]
    if args.prepare and not args.source:
        paths.extend(RAW / f"{subset}.parquet" for subset in SMOLTALK_FILES)
        paths.append(CMRC_ROOT / "data" / CMRC_FILE)
    sources = [load_source(path) for path in paths]
    if len({source.name for source in sources}) != len(sources):
        parser.error("A source can appear only once")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_special_ids(tokenizer)
    convert = OpenCC("t2s").convert
    evaluation, paths = evaluation_index(ROOT / "evaluation", convert)
    start = time.perf_counter()
    excluded = frozenset()
    review_exclusions = {}
    if args.prior_review:
        if not args.prepare:
            parser.error("Use --prior-review when creating a new full candidate")
        review_exclusions = prior_review_exclusions(args.prior_review, REPORTS / "sft_candidate_key.jsonl")
        excluded = frozenset(review_exclusions["flagged_ids"])
    if args.exclude_file:
        items = json.loads(args.exclude_file.read_text(encoding="utf-8-sig"))
        if not isinstance(items, list) or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{32}", item)
                                               for item in items):
            parser.error("Exclusion file must contain a JSON array of candidate IDs")
        excluded |= frozenset(items)
    rows, stats = collect(sources, tokenizer, convert, evaluation, None if args.prepare else args.probe_rows, excluded)
    if args.prepare:
        splits = assign_splits(select_mixture(rows, sum(SPLIT_COUNTS.values())), SPLIT_COUNTS)
        manifest = write_candidate(splits, stats, tokenizer, evaluation, paths, args.candidate_name, review_exclusions)
        print(json.dumps({key: manifest[key] for key in ("frozen", "training_ready", "counts", "source_mix", "category_mix")},
                         ensure_ascii=False))
        return
    sample = []
    if rows:
        size = min(len(rows), args.probe_sample)
        if {source.file for source in sources} == set(SOURCE_TARGETS):
            selected = select_mixture(rows, size)
            sample = sorted(selected, key=lambda row: digest("probe", SEED, row["id"]))
        else:
            sample = stratified_sample(rows, size, SEED)
    ensure_disk_budget(REPORTS, required_bytes=16 * 1024 ** 2)
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_audit(sample, REPORTS, args.probe_name, SEED)
    report = {"mode": "probe", "seconds": round(time.perf_counter() - start, 3), "sources": stats,
              "accepted_unique": len(rows), "probe_samples": len(sample), "evaluation_prompts": len(evaluation.exact),
              "evaluation_paths": [str(path.relative_to(ROOT)) for path in paths],
              "categories": dict(Counter(row["category"] for row in rows)), "seed": SEED,
              "external_generation_calls": 0, "assistant_tokens_in_probe": sum(
                  token != IGNORE_LABEL for row in sample for token in row["labels"][1:]),
              "max_seq": MAX_SEQ, "final_eos": False, "independent_quality_review": "pending"}
    report_path = REPORTS / f"sft_{args.probe_name}_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if len(sample) < args.probe_sample:
        raise SystemExit(f"Only {len(sample)} of {args.probe_sample} blind examples available; see {report_path}")


if __name__ == "__main__":
    main()
