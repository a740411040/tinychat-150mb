"""Retrieve source-faithful candidates and validate, without deciding, admission."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from itertools import chain
import json
from pathlib import Path
import random
import re

from tokenizers import Tokenizer

from .chat_format import check_special_ids, encode_conversation
from .natural_sft_v4_rewrite_seeds import DEST as OLD_PILOT, locator_key, prior_review_locations
from .natural_sft_v4_scale_seeds import (
    BROKEN, DEFAULT_PROTECTED, DOCUMENT_ACTION, DOCUMENT_REFERENCE, EVERYDAY, IMPLICIT_HEALTH_OR_RISK,
    IMPLICIT_PRACTICAL_CONTEXT, IN_TURN_CHOICES, INSTRUCTION_PAYLOAD, PRIVATE, ROLE_TEXT, SCOPE_CHANGE, STRONG_TASK,
    V2_EXPLICIT_TRANSFORMATION, V2_PSEUDO_CONTENT,
    LexicalIndex, OUTPUT as SCALE_DATA, SCALE_REPORT, dependency, fixed_tokens, honesty_rows,
    implicit_dependency, load_protected, material_from_user, option_spans, reject_user, user_keys, v2_user_shape,
    write_rows,
)
from .natural_sft_v4_semantic_pool import (
    CMRC_PREFIX, CMRC_SEPARATOR, MAX_TOKENS, OUTPUT as OLD_POOL, ROOT, TOKENIZER,
    belle_rows, belle_stage1_rows, cmrc_rows, json_rows, moss_rows, relative, smol_rows, structural_reject,
    write_json,
)
from .stage1_data import normalized_content


OUTPUT = ROOT / "data/natural_sft_v4_revision_v3"
OUTPUT_CANDIDATES = OUTPUT / "candidate_pool_v1.jsonl"
OUTPUT_MICRO = OUTPUT / "micro_raw_v1.jsonl"
OUTPUT_HANDOFF = OUTPUT / "micro_raw_handoff_v1.jsonl"
HANDOFF_REPORT = SCALE_REPORT / "revision_seed_handoff_v1.json"
OUTPUT_ROUND2 = OUTPUT / "micro_raw_handoff_round2_v1.jsonl"
ROUND2_REPORT = SCALE_REPORT / "revision_seed_handoff_round2_v1.json"
ROUND2_FRAME = OUTPUT / "micro_sampling_frame_round2_v1.jsonl"
ROUND2_SCAN = SCALE_REPORT / "revision_seed_round2_scan_v1.json"
SEED_REPORT = SCALE_REPORT / "revision_seed_inventory_v1.json"
EXCLUSIONS_REPORT = SCALE_REPORT / "revision_seed_exclusions_v1.json"
CLASSES = ("natural", "grounded", "multi", "instruction")
SEED = 20261127
HANDOFF_SEED = SEED + 1
ROUND2_SEED = HANDOFF_SEED + 1
ROUND1_EXIT_STEMS = ("6f82dbae", "28bb9471", "0e27be3b", "243a612e")
CMRC_NOWIKI = re.compile(r"(?:<\s*/?\s*nowiki\b[^<>]*>|&lt;\s*/?\s*nowiki\b(?:(?!&gt;).)*&gt;)", re.I)
CMRC_EMPTY_ANNOTATION = re.compile(
    r"(?<=[\u3400-\u9fff])[ \t\u3000]*(?P<empty>（[ \t\u3000]*）|\([ \t\u3000]*\))"
    r"(?=\s*(?:[，。；：、,.!?！？]|是|为|指|又称|也叫|又名|$))"
)
NORMAL_PARENTHESES = re.compile(r"（[^（）\n]{1,60}）|\([^()\n]{1,60}\)")
NORMAL_QUOTE = re.compile(r"“[^”“\n]{2,80}”|《[^《》\n]{2,80}》|「[^「」\n]{2,80}」")
_REWORD = r"(?:改写|重写|重新(?:表达|写)|换(?:一种|种)说法|改述|编辑)"
_PLAIN_LANGUAGE = r"(?:通俗(?:易懂)?|简单(?:易懂)?|平实|直白|易懂|易于理解|容易理解)"
_EASIER_EXPRESSION = (
    r"(?:通俗(?:易懂)?|简单易懂|易懂|易于理解|容易理解|好理解|看得懂|听得懂|平实|直白"
    r"|小学生[^，。！？]{0,8}(?:懂|理解)|简单(?=(?:的)?(?:语言|表达|措辞|说法|词语|话)))"
)
# A style word in the source text or an output object's name is not a simplification request.
HANDOFF_SIMPLIFY = re.compile(
    r"^\s*(?:(?:请你?|麻烦(?:你)?|帮我|能否|please)\s*)*(?:"
    r"(?:简化|通俗化)(?:这|下面|以下|下列|原|句|文章|文本|内容|语言|措辞|表达)"
    r"|(?:用|以|使用)(?:更|比较)?"
    + _PLAIN_LANGUAGE
    + r"(?:的)?(?:语言|话|措辞|说法|表达(?:方式)?|方式|词语)(?:来)?"
    + _REWORD
    + r"|(?:(?:把|将)[^，,。！？；;]{1,24})?"
    + _REWORD
    + r"[^，,。！？；;“”《》\"]{0,26}[，,]?\s*(?:使|让|改得|写得|变得|得|成)"
    + r"[^，,。！？；;“”《》\"]{0,24}?"
    + _EASIER_EXPRESSION
    + r"|simplify\b|(?:rewrite|rephrase|paraphrase)\b[^.!?]{0,70}\bin (?:plain|simple) language\b)",
    re.I,
)
HANDOFF_REWRITE = re.compile(r"改写|重写|重新(?:表达|写)|换(?:一种|种)说法|改述|rewrite|paraphras", re.I)
ROUND2_FORMALIZE = re.compile(
    r"^\s*(?:请|帮我)?(?:用|以)(?:更)?(?:正式|专业|商务|书面)(?:的)?(?:语气|口吻|表达|语言|措辞|方式)"
    r".{0,6}(?:重写|改写|重新表达|编辑)"
)
ROUND2_TEXT_DELIVERABLE = re.compile(
    r"^\s*(?:(?:请|帮我|请你)\s*)?(?:生成|写|创作|构思|编写|起草|描述|改写|重写)"
    r"[^，。！？\n]{0,38}(?:段落|一段|文章|一篇|小说|故事|寓言|邮件|报告|通知|演讲|文案|论文|小作文|台词)"
)
ROUND2_DOCUMENT_FOLLOWUP = re.compile(
    r"(?:这(?:篇|段|份|个)文(?:章|案|本)|上[面述]的(?:文|内容|文章)|(?:刚才|你)写的|"
    r"这(?:篇|段|则)(?:故事|作文|邮件|报告)|(?:该|这)文(?:章|本)|文章中|前文中|(?:这|刚才的)(?:段内容|草稿|邮件|提纲)|"
    r"从(?:上文|文中|前文)|把(?:文章|邮件|信件|故事|段落|报告))"
)
ROUND2_DOC_EVALUATE = re.compile(r"好处|帮助|作用|有什么用|有什么效果|主旨|内容|评价|怎么样|要点|怎么写")
ROUND2_SETTING = re.compile(
    r"(?:我|我们)(?:家|最近|刚|下班|住|手上|手边|打算|在|有|周末|平时|每次|准备|只|没|同事|室友)"
    r"|室友|同事|家里|孩子|朋友|(?:周末|今晚|这几天|下周)(?:想|要|准备|打算|一直)"
)
ROUND2_CONCRETE = re.compile(
    r"只有|只能|没有|不能|不想|不便|必须|来不及|打算|总是|每次|刚开始|预算|小时|分钟|周末|"
    r"下班|刚搬|刚换|怕|选|比较|已经|针织|条款|考试|例会|屏幕|资料|家里|路线"
)
ROUND2_ACTION = re.compile(r"怎么|怎样|如何|建议|办法|需要|怎么办|帮我|想|着急|有点|担心|难过|纠结|紧张|开心|松了口气|烦")
ROUND2_FOLLOW_LINK = re.compile(
    r"^(?:那[么你我如果要]|那[，,]|那$|但是|不过|可我|如果|那样|那就|好[，,]|嗯[，,]|明白)"
    r"|(?:你|您)(?:刚才|刚)?(?:提到|说的|建议的)|上[面述]|之中|这(?:个|些|种|几项)|它们|那一种|刚才|前面|"
    r"还有其他|还有别的|除了|第一种|第二种|这一项|那一项"
)
ROUND2_CHANGE = re.compile(
    r"但|不过|如果|要是|只能|只有|没有|预算|时间|分钟|没法|不方便|加(?:上|入)|换(?:成|为)|改(?:变|成|为)|"
    r"去掉|除外|不能|不想|其他|别的|另一|哪(?:种|个|一个|项)|优先|先做|怎么(?:做|用|选)|更(?:合适|有效|方便|简单|省事)|具体"
)
ROUND2_BROKEN_USER = re.compile(
    r"\\[.，。]|\n\s*\n\s*\n|(?:^|\n)\s*(?:用户|助手|assistant|user)\s*[:：]|"
    r"[\u0400-\u04ff\u0600-\u06ff\u0900-\u097f\uac00-\ud7af]|(?:[A-Za-z]+ ){4}[A-Za-z]+",
    re.I,
)
ROUND2_FIELD_GROUPS = ("source_group", "original_source_group", "document_group", "exclusive_route_group", "pool_group", "group_id")
ROUND2_TARGETS = {"natural": 12, "grounded": 12, "instruction": 16, "daily": 20, "document": 4}
SITUATED = re.compile(
    r"最近|这几天|今晚|今天|下班|上班|我家|我们家|邻居|室友|同事|朋友|钥匙|做饭|收纳|书桌|周末|备考|开会|练习|工作中|学习时"
)
PRACTICAL_ASK = re.compile(r"怎么|怎样|如何|建议|办法|困扰|担心|有点|不知|纠结|有些|难过|开心|烦恼|怎么办")
MECHANICAL = re.compile(
    r"(?:提取|找出|列出|选出).{0,18}(?:列|姓名|名称|名单|城市|字段|格式|数字|短语)|"
    r"改成|改写|重写|翻译|排序|替换|匹配|转换|改标点|标题|格式化|用逗号分隔|翻成"
)
V2_NOISY_INPUTS = {
    "pseudo_answer_or_mixed_script_corruption",
    "explicit_minimum_length_exceeds_response_budget",
}


def _require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def candidate_prefix_options(tokenizer, messages):
    """Return all allowed leading user/assistant pairs, independent of old reply length."""
    options = []
    for end in range(2, min(len(messages), 4) + 1, 2):
        prefix = messages[:end]
        if structural_reject(prefix) or [m["role"] for m in prefix] != ["user", "assistant"] * (end // 2):
            break
        if any(PRIVATE.search(item["content"]) for item in prefix):
            break
        user_messages = prefix[::2]
        if (
            any(BROKEN.search(item["content"]) or ROLE_TEXT.search(item["content"])
                or V2_PSEUDO_CONTENT.search(item["content"]) for item in user_messages)
            or v2_user_shape(prefix) in V2_NOISY_INPUTS
        ):
            break
        fixed = fixed_tokens(tokenizer, prefix)
        budget = MAX_TOKENS - fixed
        if budget < end // 2:
            break
        options.append({
            "message_count": end, "fixed_tokens": fixed, "assistant_content_token_budget": budget,
        })
    return options


def admission_payload(candidate):
    """Send no local hints, locators, evidence guesses, or protected inputs."""
    return {
        "id": candidate["id"],
        "original_messages": [{"role": item["role"], "content": item["content"]}
                              for item in candidate["original_messages"]],
        "prefix_options": [{name: option[name] for name in (
            "message_count", "fixed_tokens", "assistant_content_token_budget",
        )} for option in candidate["prefix_options"]],
    }


def _entry_list(value, fields, name):
    _require(isinstance(value, list), f"{name} must be a list")
    for item in value:
        _require(isinstance(item, dict) and set(item) == set(fields), f"Malformed {name} entry")
    return value


def _index(index, end):
    return type(index) is int and 0 <= index < end


def validate_admission(candidate, response, tokenizer):
    """Validate the decision contract. A successful validation is only structural."""
    _require(isinstance(candidate, dict) and _text(candidate.get("id")), "Invalid candidate identity")
    _require(isinstance(response, dict), "Admission must be an object")
    identifier = candidate["id"]
    _require(response.get("id") == identifier, "Admission ID differs from candidate")
    fields = {"id", "decision", "category", "message_count", "reason", "requirements", "evidence", "dependencies", "multi_subtype"}
    _require(set(response) <= fields, "Unexpected admission field")
    _require(response.get("decision") in ("accept", "reject") and _text(response.get("reason")),
             "Admission needs an accept/reject decision and reason")
    decision = response["decision"]
    result = {"id": identifier, "decision": decision, "errors": [], "admitted_seed": None, "admission": response}
    if decision == "reject":
        for key in ("category", "message_count", "multi_subtype"):
            _require(response.get(key) is None, f"Rejected admission cannot set {key}")
        for key in ("requirements", "evidence", "dependencies"):
            _require(response.get(key, []) == [], f"Rejected admission cannot set {key}")
        return result

    _require(set(response) == fields, "Accepted admission is missing fields")
    category = response["category"]
    end = response["message_count"]
    _require(category in CLASSES, "Unknown admission category")
    _require(type(end) is int and end >= 2 and end % 2 == 0, "Admission must select a complete leading exchange")
    original = candidate.get("original_messages")
    _require(isinstance(original, list) and end <= len(original), "Selected prefix exceeds original source")
    option = next((opt for opt in candidate.get("prefix_options", ()) if opt["message_count"] == end), None)
    _require(option is not None, "Selected leading boundary is absent from the candidate")
    prefix = original[:end]
    _require(structural_reject(prefix) is None and [item["role"] for item in prefix] == ["user", "assistant"] * (end // 2),
             "Selected leading prefix has invalid roles or content")
    fixed = fixed_tokens(tokenizer, prefix)
    budget = MAX_TOKENS - fixed
    _require(option["fixed_tokens"] == fixed and option["assistant_content_token_budget"] == budget and budget >= end // 2,
             "Selected leading prefix has an invalid reply budget")

    requirements = _entry_list(response["requirements"], ("assistant_message_index", "text"), "requirements")
    assistants = list(range(1, end, 2))
    addressed = set()
    for item in requirements:
        index = item["assistant_message_index"]
        _require(_index(index, end) and prefix[index]["role"] == "assistant" and _text(item["text"]),
                 "Requirement must identify an original assistant turn and give text")
        addressed.add(index)
    _require(addressed == set(assistants), "Requirements must cover all selected assistant turns")

    evidence = _entry_list(response["evidence"], ("message_index", "quote"), "evidence")
    _require(category != "grounded" or bool(evidence), "Grounded admission must supply visible user evidence")
    for item in evidence:
        index = item["message_index"]
        quote = item["quote"]
        _require(_index(index, end) and prefix[index]["role"] == "user" and _text(quote)
                 and quote in prefix[index]["content"], "Evidence must quote exactly from a preserved user message")

    dependencies = _entry_list(response["dependencies"],
                               ("user_message_index", "history_message_index", "history_quote", "missing_without_history"),
                               "dependencies")
    subtype = response["multi_subtype"]
    if category != "multi":
        _require(end == 2, "Non-multi admission must select only the original first exchange")
        _require(not dependencies, "Non-multi admission cannot retain dependency metadata")
        _require(subtype is None, "Only multi admission can assign a multi subtype")
    covered = set()
    for item in dependencies:
        user, history = item["user_message_index"], item["history_message_index"]
        quote = item["history_quote"]
        _require(_index(user, end) and user > 0 and prefix[user]["role"] == "user" and _index(history, user)
                 and _text(quote) and quote in prefix[history]["content"] and _text(item["missing_without_history"]),
                 "Dependency must quote prior history exactly and explain what would be missing")
        covered.add(user)
    if category == "multi":
        _require(end >= 4 and subtype in ("daily", "document") and covered == set(range(2, end, 2)),
                 "Multi admission must cover later user turns with history and use a valid subtype")

    _require(_text(candidate.get("source_group")) and isinstance(candidate.get("sources"), list) and candidate["sources"],
             "Candidate is missing source-group lineage")
    seed = {
        "id": identifier, "candidate_id": identifier, "category": category,
        "original_messages": prefix, "user_messages": prefix[::2],
        "sources": candidate["sources"], "source_group": candidate["source_group"],
        "original_source_group": candidate.get("original_source_group", candidate["source_group"]),
        "rewrite_assistant_turn_indices": assistants,
        "selected_original_message_range_zero_based_half_open": [0, end],
        "assistant_content_token_budget": budget, "tokens_without_assistant_content": fixed,
        "max_context": MAX_TOKENS, "final_eos": False, "admission_reason": response["reason"],
        "admission_requirements": requirements, "admission_evidence": evidence, "admission_dependencies": dependencies,
        "multi_subtype": subtype, "old_assistant_text_status": "unverified_reference_only",
    }
    for key in ("document_group", "exclusive_route_group", "pool_prefix_id", "validation_challenges"):
        if key in candidate:
            seed[key] = candidate[key]
    result["admitted_seed"] = seed
    return result


def validate_checker_context(messages, response, expected_subtype) -> list[str]:
    """Validate v6 context evidence, without certifying the semantic dependency."""
    _require(isinstance(response, dict) and response.get("category") in (*CLASSES, "unsupported"),
             "Checker response has an invalid category for context audits")
    _require(expected_subtype in (None, "daily", "document"), "Invalid expected multi subtype")
    subtype = response.get("multi_subtype")
    audits = response.get("context_audits", [])
    if response["category"] != "multi":
        _require(subtype is None and audits == [], "Non-multi Checker results cannot carry context audits or a multi subtype")
        return ["checker_multi_subtype_disagreement"] if expected_subtype is not None else []
    _require(subtype in ("daily", "document"), "Multi Checker response needs a valid multi subtype")
    _require(structural_reject(messages) is None and len(messages) >= 4
             and [item["role"] for item in messages] == ["user", "assistant"] * (len(messages) // 2),
             "Context audits need a complete original-order sequence of user and assistant messages")
    _require(all("message_index" not in item or (type(item["message_index"]) is int and item["message_index"] == i)
                 for i, item in enumerate(messages)), "Message index differs from its zero-based position")
    fields = ("user_message_index", "history_message_index", "history_quote", "missing_without_history", "answer_quote")
    audits = _entry_list(audits, fields, "context_audits")
    covered = set()
    for audit in audits:
        user, history = audit["user_message_index"], audit["history_message_index"]
        _require(_index(user, len(messages) - 1) and user >= 2 and messages[user]["role"] == "user"
                 and _index(history, user) and messages[user + 1]["role"] == "assistant",
                 "A context audit needs a later user, a previous message, and the next assistant")
        _require(_text(audit["history_quote"]) and audit["history_quote"] in messages[history]["content"],
                 "Context history quote must be exact in the indexed earlier message")
        _require(_text(audit["missing_without_history"]), "Context audit needs what would be missing without history")
        _require(_text(audit["answer_quote"]) and audit["answer_quote"] in messages[user + 1]["content"],
                 "Context answer quote must be exact in the later user's adjacent assistant")
        covered.add(user)
    _require(covered == set(range(2, len(messages), 2)), "Context audits must cover every later user message")
    return ["checker_multi_subtype_disagreement"] if subtype != expected_subtype else []


def load_candidates(path):
    """Read a frozen local candidate JSONL and reject ambiguous identities."""
    seen = set()
    for row in json_rows(Path(path)):
        identifier = row.get("id")
        _require(_text(identifier) and identifier not in seen, f"Invalid or duplicate candidate ID in {path}")
        _require(isinstance(row.get("prefix_options"), list) and row["prefix_options"], f"Missing prefix options for {identifier}")
        seen.add(identifier)
        yield row


def source_key(source):
    for field in ("source_row_id", "old_train_id", "json_array_ordinal_1_based", "jsonl_line_1_based",
                  "source_row_1_based", "source_byte_offset", "ordinal_1_based", "cache_ordinal"):
        if source.get(field) is not None:
            return (source["repo"], source["revision"], source["local_file"], field, str(source[field]))
    raise ValueError("Missing source locator")


class GroupRoots:
    def __init__(self):
        self.parents = {}

    def root(self, group):
        while group in self.parents:
            parent = self.parents[group]
            if parent in self.parents:
                self.parents[group] = self.parents[parent]
            group = parent
        return group

    def alias(self, group, known_group):
        left, right = self.root(group), self.root(known_group)
        if left != right:
            self.parents[left] = right
        return right


@dataclass
class PriorFrame:
    groups: dict
    locators: dict
    coarse_locators: dict
    first_users: LexicalIndex
    trajectories: dict
    files: Counter


def prior_frame():
    coarse, old_reports = prior_review_locations()
    old = PriorFrame({}, {}, coarse, LexicalIndex(), {}, Counter())
    for name in old_reports:
        old.files[name] = sum(name in files for files in coarse.values())

    def add(row, path, *, identifier=None):
        identifier = identifier or row.get("id") or row.get("sample_id") or row.get("pool_prefix_id", "")
        groups = [row[key] for key in ("original_source_group", "source_group", "pool_group", "group_id",
                                       "exclusive_route_group") if row.get(key)]
        for group in groups:
            old.groups.setdefault(group, identifier)
        sources = row.get("sources") or ([row["source"]] if isinstance(row.get("source"), dict) else [])
        for source in sources:
            old.locators.setdefault(source_key(source), identifier)
        messages = row.get("original_messages") or row.get("messages") or row.get("user_messages")
        if messages:
            if isinstance(messages[0], str):
                messages = [{"role": "user", "content": item} for item in messages]
            users, trajectory = user_keys(messages)
            if users and users[0]:
                old.first_users.add(users[0], identifier, groups[0] if groups else "seen_diagnostic")
            if len(users) > 1:
                old.trajectories.setdefault(trajectory, identifier)
        old.files[relative(path)] += 1

    for path in (OLD_PILOT, SCALE_DATA / "probe_seeds_v1.jsonl", SCALE_DATA / "probe_implicit_multi_v2.jsonl"):
        for row in json_rows(path):
            add(row, path)
    path = SCALE_REPORT / "multi_recall_diagnostic.json"
    for row in json.loads(path.read_text(encoding="utf-8"))["sampled_records"]:
        add(row, path)
    path = SCALE_REPORT / "local_scale_inventory.json"
    for row in json.loads(path.read_text(encoding="utf-8"))["examples"]:
        add(row, path, identifier=row.get("pool_prefix_id") or f"honesty-source:{row['source']['source_row_id']}")
    path = SCALE_REPORT.parent / "semantic_pool_examples.json"
    for rows in json.loads(path.read_text(encoding="utf-8"))["inputs"].values():
        for row in rows:
            add(row, path)
    return old


def pool_lineage(roots):
    known = {}
    for row in json_rows(OLD_POOL / "pool.jsonl"):
        for source in row["sources"]:
            known[source_key(source)] = (row["group_id"], source["source_group_id"], row["id"])
    for row in json_rows(SCALE_DATA / "seed_pool_v2.jsonl"):
        original = row["original_source_group"]
        if original != row["source_group"]:
            roots.alias(original, row["source_group"])
    return known


def local_inputs():
    pooled = chain(moss_rows(), belle_rows(), belle_stage1_rows(), cmrc_rows("train"), cmrc_rows("validation"), smol_rows())
    for source in pooled:
        if source.reject:
            yield source.name, None, source.reject
            continue
        if not isinstance(source.messages, list):
            yield source.name, None, "invalid_original_message_array"
            continue
        provenance = {"input": source.name, **source.provenance, "transform": source.transform,
                      "original_message_count": len(source.messages)}
        yield source.name, {"messages": source.messages[:4], "sources": [provenance], "document": source.document}, None
    counts = Counter()
    for row, reject in honesty_rows(counts):
        yield "moss_honesty", row, reject


def _first_material_question(messages):
    user = messages[0]["content"]
    if user.startswith(CMRC_PREFIX) and CMRC_SEPARATOR in user:
        return user.rsplit(CMRC_SEPARATOR, 1)[1]
    return None


def cmrc_material_findings(candidate):
    """Locate explicit rendering defects only inside a traceable full CMRC passage."""
    source = next((item for item in candidate["sources"] if item["input"].startswith("cmrc_")), None)
    if source is None:
        return []
    first = candidate["original_messages"][0]
    text = first["content"]
    span = source.get("transform", {}).get("passage_chars_zero_based_half_open")
    valid_span = (isinstance(span, list) and len(span) == 2
                  and all(type(value) is int for value in span) and span[0] == 0 and span[1] > 0)
    start = len(CMRC_PREFIX)
    end = start + span[1] if valid_span else start
    if (not valid_span or first["role"] != "user" or not text.startswith(CMRC_PREFIX)
            or not text[end:].startswith(CMRC_SEPARATOR) or not text[end + len(CMRC_SEPARATOR):].strip()):
        return [{"reason": "cmrc_unverifiable_full_passage_boundary", "message_index": 0, "chars_zero_based_half_open": None}]
    material = text[start:end]
    findings = []
    for pattern, reason, gid in (
        (CMRC_NOWIKI, "cmrc_unrendered_nowiki", 0),
        (CMRC_EMPTY_ANNOTATION, "cmrc_empty_prose_annotation", "empty"),
    ):
        for match in pattern.finditer(material):
            left, right = match.span(gid)
            findings.append({"reason": reason, "message_index": 0,
                             "chars_zero_based_half_open": [start + left, start + right]})
    return findings


def cmrc_document_keys(candidate):
    """Return immutable lineage keys used to quarantine all questions on a damaged document."""
    return {
        (source["repo"], source["revision"], source["source_document_id"])
        for source in candidate["sources"] if source.get("source_document_id")
    }


def sampling_signals(messages, source):
    """Local-only selection hints. A completely unhinted source row stays eligible."""
    first = messages[0]["content"]
    head = first.split("\n", 1)[0][:140]
    document = source.get("source_document_id") is not None
    hints = []
    operation = (STRONG_TASK.search(head) or V2_EXPLICIT_TRANSFORMATION.search(head) or MECHANICAL.search(head))
    if not document and operation and INSTRUCTION_PAYLOAD.search(first):
        hints.append("instruction")
    material = material_from_user(first)
    if document or (material and material.mode in (
        "labeled_material_then_question", "labeled_question_then_material", "visible_dialogue_then_question",
        "visible_quote_and_question", "untitled_paragraph_then_question",
    ) and not operation and not MECHANICAL.search(material.question)):
        hints.append("grounded")
    if not operation and not material and not document and len(first) <= 130 and SITUATED.search(first) and PRACTICAL_ASK.search(first):
        hints.append("natural")
    multi_kind, has_anchor = None, False
    if len(messages) >= 4:
        follow = messages[2]["content"]
        if DOCUMENT_REFERENCE.search(follow):
            hints.append("multi")
            multi_kind = "document"
        elif EVERYDAY.search(first) or IMPLICIT_PRACTICAL_CONTEXT.search(first):
            hints.append("multi")
            multi_kind = "daily"
        if multi_kind:
            has_anchor = bool(dependency(messages[:2], messages[2]) or implicit_dependency(messages)[0])
    return hints, multi_kind, has_anchor


def key_fingerprint(value):
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return blake2b(payload, digest_size=16).hexdigest()


def guard_prefixes(messages, options, protected_index, protected_paths, prior):
    accepted = []
    bare = _first_material_question(messages)
    if bare and (hit := protected_index.match(normalized_content(bare))):
        return accepted, ("protected_material_question_" + hit[0], hit[1])
    if bare and (hit := prior.first_users.match(normalized_content(bare))):
        return accepted, ("prior_material_question_" + hit[0], hit[1])
    users, _ = user_keys(messages[:options[-1]["message_count"]])
    for i, opt in enumerate(options):
        last_user = users[opt["message_count"] // 2 - 1]
        if not last_user:
            return accepted, ("invalid_normalized_user_key", None)
        trajectory = "|".join(users[:opt["message_count"] // 2])
        for family, index, trajectories in (
            ("protected", protected_index, protected_paths),
            ("prior", prior.first_users, prior.trajectories),
        ):
            if (i == 0 or len(last_user) >= 24) and (hit := index.match(last_user)):
                return accepted, (family + "_user_" + hit[0], hit[1])
            if trajectory in trajectories:
                return accepted, (family + "_exact_user_trajectory", trajectories[trajectory])
        accepted.append(opt)
    return accepted, None


def build_catalog(tokenizer, *, limit=None):
    roots = GroupRoots()
    sources = pool_lineage(roots)
    protected, paths, protected_count = load_protected(DEFAULT_PROTECTED)
    prior = prior_frame()
    prior_groups = {roots.root(group): ref for group, ref in prior.groups.items()}
    protected_document_groups = {}
    exact_questions, exact_documents, document_contexts, fingerprints = {}, {}, {}, {}
    seen_references, counts, rejected = [], Counter(), defaultdict(Counter)
    available = []
    aliases = []
    for name, raw, rejection in local_inputs():
        if limit and counts[name] >= limit:
            continue
        counts[name] += 1
        if rejection:
            rejected[name][rejection] += 1
            continue
        messages = raw["messages"]
        if not isinstance(messages, list) or not messages:
            rejected[name]["no_valid_leading_exchange"] += 1
            continue
        source = raw["sources"][0]
        key = source_key(source)
        identifier = "revision-" + key_fingerprint(key)
        previous = sources.get(key)
        group = previous[0] if previous else raw.get("group_id", identifier)
        original = previous[1] if previous else source.get("source_group_id", group)
        source["source_group_id"] = original
        first = messages[0]
        document = raw.get("document")
        if document is not None:
            doc = (source["repo"], source["revision"], source["source_document_id"])
            group = roots.alias(group, exact_documents.setdefault(doc, group))
            group = roots.alias(group, document_contexts.setdefault(key_fingerprint(document), group))
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            normalized = normalized_content(first["content"])
            if normalized:
                group = roots.alias(group, exact_questions.setdefault(normalized, group))
        if key in prior.locators or roots.root(group) in prior_groups:
            ref = prior.locators.get(key, prior_groups.get(roots.root(group), "prior_group"))
            seen_references.append({"id": identifier, "source_group": group, "reason": "prior_source_or_group", "conflicts_with": ref})
            prior_groups[roots.root(group)] = ref
            rejected[name]["prior_source_or_group"] += 1
            continue
        if name != "moss_honesty" and locator_key(source) in prior.coarse_locators:
            ref = sorted(prior.coarse_locators[locator_key(source)])
            prior_groups[roots.root(group)] = ref
            seen_references.append({"id": identifier, "source_group": group, "reason": "prior_report_locator", "conflicts_with": ref})
            rejected[name]["prior_report_locator"] += 1
            continue
        options = candidate_prefix_options(tokenizer, messages)
        if not options:
            rejected[name]["no_valid_leading_exchange"] += 1
            continue
        options, conflict = guard_prefixes(messages, options, protected, paths, prior)
        if conflict:
            reason, ref = conflict
            seen_references.append({
                "id": identifier, "source_group": group, "reason": reason, "conflicts_with": ref,
                "remaining_leading_options": [opt["message_count"] for opt in options],
            })
            rejected[name][("suffix_" if options else "") + reason] += 1
            if not options:
                if reason.startswith("prior_"):
                    prior_groups[roots.root(group)] = ref
                if reason.startswith("protected_") and document is not None:
                    protected_document_groups[roots.root(group)] = ref
                continue
        messages = messages[:options[-1]["message_count"]]
        fingerprint = key_fingerprint([(entry["role"], entry["content"]) for entry in messages])
        if first_candidate := fingerprints.get(fingerprint):
            original_roles_and_text = [(item["role"], item["content"]) for item in first_candidate["original_messages"]]
            _require(original_roles_and_text == [(item["role"], item["content"]) for item in messages],
                     "Conflicting exact-prefix digest")
            roots.alias(group, first_candidate["source_group"])
            first_candidate["sources"].append(source)
            aliases.append({"candidate": identifier, "representative": first_candidate["id"], "source": key})
            rejected[name]["exact_prefix_alias_merged"] += 1
            continue
        hints, multi_hint, lexical = sampling_signals(messages, source)
        candidate = {
            "id": identifier, "source_group": group, "original_source_group": original,
            "sources": [source], "original_messages": messages, "prefix_options": options,
            "sampling_hint": None, "possible_sampling_hints": hints,
            "multi_sampling_subtype": multi_hint, "multi_has_lexical_hint": lexical,
        }
        if previous:
            candidate["pool_prefix_id"] = previous[2]
        if document is not None:
            candidate["document_group"] = group
        available.append(candidate)
        fingerprints[fingerprint] = candidate
        if counts[name] in (10_000, 40_000, 80_000, 120_000):
            print(f"Read {counts[name]:,} {name} rows; {len(available):,} structural candidates.", flush=True)
    seen_roots = {roots.root(group): ref for group, ref in prior_groups.items()}
    protected_doc_roots = {roots.root(group): ref for group, ref in protected_document_groups.items()}
    retained = []
    for row in available:
        row["source_group"] = root = roots.root(row["source_group"])
        if root in seen_roots:
            rejected[row["sources"][0]["input"]]["prior_connected_source_group"] += 1
            seen_references.append({"id": row["id"], "source_group": root, "reason": "prior_connected_source_group",
                                    "conflicts_with": seen_roots[root]})
            continue
        if root in protected_doc_roots:
            rejected[row["sources"][0]["input"]]["protected_connected_document_group"] += 1
            seen_references.append({"id": row["id"], "source_group": root, "reason": "protected_connected_document_group",
                                    "conflicts_with": protected_doc_roots[root]})
            continue
        if "document_group" in row:
            row["document_group"] = root
        retained.append(row)
    counts_report = {
        "status": "local_candidates_only", "protected_projection": relative(DEFAULT_PROTECTED),
        "protected_projection_rows": protected_count,
        "prior_and_diagnostic_provenance_counts_nonadditive": dict(prior.files),
        "raw_by_input": dict(counts), "structural_or_protection_reasons_by_input": rejected,
        "catalog_rows": len(retained), "catalog_distinct_groups": len({row["source_group"] for row in retained}),
        "catalog_rows_by_primary_input": Counter(row["sources"][0]["input"] for row in retained),
        "leading_boundaries": Counter(str(len(row["original_messages"]) // 2) for row in retained),
        "local_hints_nonadditive": Counter(hint for row in retained for hint in row["possible_sampling_hints"]),
        "local_hint_groups_nonadditive": {
            category: len({row["source_group"] for row in retained if category in row["possible_sampling_hints"]})
            for category in CLASSES
        },
    }
    excludes = {"prior_or_protected_conflicts": seen_references, "exact_literal_prefix_aliases": aliases,
                "known_previously_seen_groups": sorted(seen_roots), "alias_group_edges": roots.parents}
    return retained, counts_report, excludes


class MicroSampler:
    def __init__(self, seed=SEED):
        self.rng = random.Random(seed)
        self.groups, self.firsts, self.paths = set(), LexicalIndex(), LexicalIndex()
        self.conflicts = []

    def accept(self, row):
        group = row["source_group"]
        if group in self.groups:
            return False
        users, trajectory = user_keys(row["original_messages"])
        if hit := self.firsts.match(users[0]) or self.paths.match(trajectory):
            self.conflicts.append({"id": row["id"], "source_group": group, "reason": "new_micro_" + hit[0],
                                   "conflicts_with": hit[1], "conflict_source_group": hit[2]})
            return False
        self.groups.add(group)
        self.firsts.add(users[0], row["id"], group)
        self.paths.add(trajectory, row["id"], group)
        return True

    def sample(self, rows, target, *, prefer_longer=False):
        strata = defaultdict(list)
        for row in rows:
            if row["source_group"] in self.groups:
                continue
            source = row["sources"][0]["input"]
            option = row["prefix_options"][-1 if prefer_longer else 0]
            # Enough fixed content for range coverage while keeping the first writer gate manageable.
            length = option["fixed_tokens"] // 100
            strata[(source, length)].append(row)
        for rows in strata.values():
            self.rng.shuffle(rows)
        order = sorted(strata)
        self.rng.shuffle(order)
        picked = []
        while order and len(picked) < target:
            for cell in tuple(order):
                bucket = strata[cell]
                while bucket:
                    row = bucket.pop()
                    if self.accept(row):
                        picked.append(row)
                        break
                if not bucket:
                    order.remove(cell)
                if len(picked) == target:
                    break
        return picked


def make_micro(candidates):
    sampler = MicroSampler()
    by_hint = defaultdict(list)
    for row in candidates:
        for hint in row["possible_sampling_hints"]:
            by_hint[hint].append(row)
    grounded = by_hint["grounded"]
    cmrc = [row for row in grounded if "document_group" in row]
    non_cmrc = [row for row in grounded if "document_group" not in row]
    grounded_picks = sampler.sample(non_cmrc, 10)
    grounded_picks += sampler.sample(cmrc, min(6, 16 - len(grounded_picks)))
    grounded_picks += sampler.sample(non_cmrc, 16 - len(grounded_picks))
    multi = by_hint["multi"]
    daily = [row for row in multi if row["multi_sampling_subtype"] == "daily"]
    document = [row for row in multi if row["multi_sampling_subtype"] == "document"]
    # Explicitly test histories for which the previous lexical dependency rules furnished no anchor.
    multi_picks = sampler.sample([row for row in daily if not row["multi_has_lexical_hint"]], 6, prefer_longer=True)
    multi_picks += sampler.sample([row for row in daily if row["multi_has_lexical_hint"]],
                                  12 - len(multi_picks), prefer_longer=True)
    multi_picks += sampler.sample(daily, 12 - len(multi_picks), prefer_longer=True)
    multi_picks += sampler.sample(document, min(len(multi_picks), 8, 16 - len(multi_picks)), prefer_longer=True)
    multi_picks += sampler.sample(daily, 16 - len(multi_picks), prefer_longer=True)
    instructions = sampler.sample(by_hint["instruction"], 16)
    natural = sampler.sample(by_hint["natural"], 16)
    strata = {"natural": natural, "grounded": grounded_picks, "multi": multi_picks, "instruction": instructions}
    micro = [{**row, "sampling_hint": hint} for hint, values in strata.items() for row in values]
    sampler.rng.shuffle(micro)
    summary = {
        "micro_rows": len(micro), "micro_seed": SEED,
        "micro_unique_groups": len({row["source_group"] for row in micro}),
        "micro_by_hint": Counter(row["sampling_hint"] for row in micro),
        "hint_shortfalls": {key: 16 - len(rows) for key, rows in strata.items() if len(rows) < 16},
        "micro_by_primary_input": Counter(row["sources"][0]["input"] for row in micro),
        "micro_fixed_token_bands": Counter(str(row["prefix_options"][-1]["fixed_tokens"] // 100) for row in micro),
        "micro_cmrc_count": sum("document_group" in row for row in micro),
        "micro_multi_hints_by_subtype": Counter(row["multi_sampling_subtype"] for row in multi_picks),
        "micro_multi_without_lexical_hint": sum(not row["multi_has_lexical_hint"] for row in multi_picks),
        "submission_count": 0,
    }
    return micro, summary, sampler.conflicts


def validation_challenges(candidate):
    """Return local sampling tags from visible structure, not semantic labels."""
    first = candidate["original_messages"][0]["content"]
    flags = []
    if "document_group" in candidate:
        flags.append("clean_cmrc")
    if any(re.search(r"\w", match[0][1:-1]) for match in NORMAL_PARENTHESES.finditer(first)):
        flags.append("normal_parentheses")
    if NORMAL_QUOTE.search(first):
        flags.append("normal_quote")
    if "instruction" in candidate["possible_sampling_hints"]:
        head, _, body = first.partition("\n")
        direction = head.split("：", 1)[0].split(":", 1)[0][:160]
        if HANDOFF_SIMPLIFY.search(direction):
            flags.append("simplify")
        elif HANDOFF_REWRITE.search(direction):
            flags.append("full_rewrite")
            paragraphs = [line for line in body.splitlines() if len(normalized_content(line)) >= 18]
            if len(paragraphs) >= 2:
                flags.append("multi_paragraph_rewrite")
            if candidate["prefix_options"][0]["fixed_tokens"] >= 100:
                flags.append("longer_full_rewrite")
    return flags


def known_handoff_groups(*, include_handoff=False):
    """Use the latest micro inputs plus the independent defect records as exclusions."""
    earlier_groups, earlier_locators, bad_documents = set(), set(), set()
    earlier_firsts, earlier_trajectories = LexicalIndex(), LexicalIndex()
    seen_count = 0
    paths = (OUTPUT_MICRO, OUTPUT_HANDOFF) if include_handoff else (OUTPUT_MICRO,)
    for row in chain.from_iterable(load_candidates(path) for path in paths):
        seen_count += 1
        group = row["source_group"]
        earlier_groups.update(row[key] for key in ("source_group", "original_source_group", "document_group",
                                                    "exclusive_route_group") if row.get(key))
        earlier_locators.update(source_key(source) for source in row["sources"])
        if include_handoff:
            bad_documents.update(cmrc_document_keys(row))
            earlier_groups.update(source["source_group_id"] for source in row["sources"] if source.get("source_group_id"))
        for opt in row["prefix_options"]:
            users, trajectory = user_keys(row["original_messages"][:opt["message_count"]])
            earlier_firsts.add(users[0], row["id"], group)
            for user in users[1:]:
                if len(user) >= 24:
                    earlier_firsts.add(user, row["id"], group)
            if len(users) > 1:
                earlier_trajectories.add(trajectory, row["id"], group)
    report_path = SCALE_REPORT / "revision_micro_semantic_review_v1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for row in report["records"]:
        document_keys = cmrc_document_keys(row)
        if document_keys and any(item["message_index"] == 0 and item["severity"] in ("MINOR", "MAJOR") for item in row["issues"]):
            bad_documents.update(document_keys)
            earlier_groups.add(row["source_group"])
    return earlier_groups, earlier_locators, earlier_firsts, earlier_trajectories, bad_documents, seen_count


def handoff_frame(*, limit=None):
    groups, locators, old_firsts, old_paths, bad_docs, old_seen = known_handoff_groups()
    protected, protected_paths, protected_count = load_protected(DEFAULT_PROTECTED)
    pending, exclusions, defective_roots = [], [], {}
    counts = Counter()
    for candidate in load_candidates(OUTPUT_CANDIDATES):
        counts["catalog_rows_read"] += 1
        if not candidate["possible_sampling_hints"]:
            continue
        counts["rows_with_sampling_hints"] += 1
        if limit and counts["rows_with_sampling_hints"] > limit:
            break
        root = candidate["source_group"]
        document_keys = cmrc_document_keys(candidate)
        findings = cmrc_material_findings(candidate)
        if findings or document_keys & bad_docs:
            defective_roots.setdefault(root, candidate["id"])
            bad_docs.update(document_keys)
            counts["cmrc_material_or_known_document_defect"] += 1
            exclusions.append({
                "id": candidate["id"], "source_group": root, "reason": "cmrc_material_or_known_document_defect",
                "document_keys": sorted(document_keys), "findings": findings,
            })
            continue
        candidate_groups = {
            candidate.get(key) for key in ("source_group", "original_source_group", "document_group", "exclusive_route_group")
        }
        if candidate_groups & groups or any(source_key(source) in locators for source in candidate["sources"]):
            counts["seen_micro_raw_source_or_group"] += 1
            exclusions.append({"id": candidate["id"], "source_group": root, "reason": "seen_micro_raw_source_or_group"})
            continue
        reason = None
        users, _ = user_keys(candidate["original_messages"])
        for index, user in enumerate(users):
            trajectory = "|".join(users[:index + 1])
            compare_user = index == 0 or len(user) >= 24
            if old_hit := (old_firsts.match(user) if compare_user else None) or old_paths.match(trajectory):
                reason = ("seen_micro_raw_lexical_" + old_hit[0], old_hit[1])
                break
            if compare_user and (hit := protected.match(user)):
                reason = ("protected_user_" + hit[0], hit[1])
                break
            if trajectory in protected_paths:
                reason = ("protected_exact_user_trajectory", protected_paths[trajectory])
                break
        if reason:
            counts[reason[0]] += 1
            exclusions.append({"id": candidate["id"], "source_group": root, "reason": reason[0], "conflicts_with": reason[1]})
            continue
        pending.append(candidate)
    frame = []
    for candidate in pending:
        root = candidate["source_group"]
        if root in defective_roots or cmrc_document_keys(candidate) & bad_docs:
            counts["damaged_connected_document_group"] += 1
            exclusions.append({"id": candidate["id"], "source_group": root, "reason": "damaged_connected_document_group",
                               "conflicts_with": defective_roots.get(root)})
            continue
        candidate["validation_challenges"] = validation_challenges(candidate)
        frame.append(candidate)
    counts["eligible_hinted_candidates"] = len(frame)
    return frame, counts, exclusions, old_seen, protected_count


def make_handoff_micro(frame):
    sampler = MicroSampler(HANDOFF_SEED)
    by_hint = defaultdict(list)
    for row in frame:
        for hint in row["possible_sampling_hints"]:
            by_hint[hint].append(row)
    grounded = by_hint["grounded"]
    cmrc = [row for row in grounded if "document_group" in row]
    local_grounded = [row for row in grounded if "document_group" not in row]
    def tag(row, name):
        return name in row["validation_challenges"]
    # Explicitly cover normal prose punctuation and source renderings as positive controls.
    grounded_picks = sampler.sample([row for row in local_grounded if tag(row, "normal_parentheses")], 3)
    grounded_picks += sampler.sample([row for row in local_grounded if tag(row, "normal_quote")], 3)
    grounded_picks += sampler.sample(local_grounded, 10 - len(grounded_picks))
    grounded_picks += sampler.sample(cmrc, min(6, 16 - len(grounded_picks)))
    grounded_picks += sampler.sample(local_grounded, 16 - len(grounded_picks))

    instructions = by_hint["instruction"]
    instruction_picks = sampler.sample([row for row in instructions if tag(row, "simplify")], 4)
    instruction_picks += sampler.sample([row for row in instructions if tag(row, "multi_paragraph_rewrite")], 2)
    instruction_picks += sampler.sample([row for row in instructions if tag(row, "longer_full_rewrite")], 1)
    instruction_picks += sampler.sample([
        row for row in instructions if tag(row, "full_rewrite") and not tag(row, "longer_full_rewrite")
    ], 2)
    instruction_picks += sampler.sample(instructions, 16 - len(instruction_picks))

    multi = by_hint["multi"]
    daily = [row for row in multi if row["multi_sampling_subtype"] == "daily"]
    document = [row for row in multi if row["multi_sampling_subtype"] == "document"]
    multi_picks = sampler.sample([row for row in daily if not row["multi_has_lexical_hint"]], 6, prefer_longer=True)
    multi_picks += sampler.sample(daily, 12 - len(multi_picks), prefer_longer=True)
    multi_picks += sampler.sample(document, min(len(multi_picks), 8, 16 - len(multi_picks)), prefer_longer=True)
    multi_picks += sampler.sample(daily, 16 - len(multi_picks), prefer_longer=True)
    lanes = {"natural": sampler.sample(by_hint["natural"], 16), "grounded": grounded_picks,
             "multi": multi_picks, "instruction": instruction_picks}
    micro = [{**row, "sampling_hint": hint} for hint, rows in lanes.items() for row in rows]
    sampler.rng.shuffle(micro)
    return micro, lanes, sampler.conflicts


def build_handoff(*, limit=None):
    if limit is None and any(path.exists() for path in (OUTPUT_HANDOFF, HANDOFF_REPORT)):
        raise FileExistsError("Refusing to overwrite a frozen handoff batch or report")
    frame, counts, exclusions, old_seen, protected_count = handoff_frame(limit=limit)
    micro, lanes, conflicts = make_handoff_micro(frame)
    tags = Counter(tag for row in micro for tag in row["validation_challenges"])
    expectations = {"clean_cmrc": 1, "normal_parentheses": 1, "normal_quote": 1, "full_rewrite": 1, "simplify": 1,
                    "multi_paragraph_rewrite": 1, "longer_full_rewrite": 1}
    report = {
        "status": "raw_source_selection_not_adjudicated", "handoff_phase": "round1",
        "previous_semantic_revise_history_preserved": 2, "external_author_calls": 0,
        "sampling_seed": HANDOFF_SEED, "previous_micro_raw_rows_indexed": old_seen, "protected_user_trajectories": protected_count,
        "prechecks": counts, "hinted_frame_by_route_nonadditive": Counter(hint for row in frame for hint in row["possible_sampling_hints"]),
        "micro_count": len(micro), "micro_source_groups": len({row["source_group"] for row in micro}),
        "micro_by_hint": Counter(row["sampling_hint"] for row in micro),
        "hint_shortfalls": {label: 16 - len(rows) for label, rows in lanes.items() if len(rows) < 16},
        "challenge_tags_nonadditive": tags,
        "challenge_tag_shortfalls": {label: 1 - tags[label] for label in expectations if not tags[label]},
        "micro_sources": Counter(row["sources"][0]["input"] for row in micro),
        "grounded_cmrc": sum("document_group" in row for row in lanes["grounded"]),
        "grounded_non_cmrc": sum("document_group" not in row for row in lanes["grounded"]),
        "multi_hints": Counter(row["multi_sampling_subtype"] for row in lanes["multi"]),
        "multi_without_old_lexical_hint": sum(not row["multi_has_lexical_hint"] for row in lanes["multi"]),
        "fixed_token_bands": Counter(str(row["prefix_options"][-1]["fixed_tokens"] // 100) for row in micro),
        "all_clean_cmrc_are_preflight_tags_only": True,
        "mechanical_or_old_group_exclusions": exclusions, "within_batch_lexical_conflicts": conflicts,
    }
    if limit is None:
        write_rows(OUTPUT_HANDOFF, micro)
        write_json(HANDOFF_REPORT, report)
    print(json.dumps({key: report[key] for key in (
        "micro_count", "micro_by_hint", "hint_shortfalls", "micro_sources", "challenge_tags_nonadditive",
        "challenge_tag_shortfalls", "grounded_cmrc", "grounded_non_cmrc", "multi_hints",
    )}, ensure_ascii=False))


def round2_instruction_challenges(candidate):
    first = candidate["original_messages"][0]["content"]
    head, _, body = first.partition("\n")
    direction = re.split(r"[:：]", head, maxsplit=1)[0][:160]
    flags = set(validation_challenges(candidate))
    if HANDOFF_REWRITE.search(direction):
        flags.add("full_rewrite")
    if ROUND2_FORMALIZE.search(direction):
        flags.update(("formalize", "full_rewrite"))
    if "full_rewrite" in flags:
        if sum(len(normalized_content(line)) >= 18 for line in body.splitlines()) >= 2:
            flags.add("multi_paragraph_rewrite")
        if candidate["prefix_options"][0]["fixed_tokens"] >= 100:
            flags.add("longer_full_rewrite")
    return sorted(flags)


def round2_multi_signal(candidate):
    """Rank an intact original two-exchange lead by inspectable local cues."""
    messages = candidate["original_messages"]
    if len(messages) != 4 or candidate["prefix_options"][-1]["message_count"] != 4:
        return None, 0, "no_complete_second_exchange"
    first, reply, follow = (messages[i]["content"] for i in (0, 1, 2))
    if any(ROUND2_BROKEN_USER.search(text) for text in (first, follow)) or v2_user_shape(messages):
        return None, 0, "apparent_user_surface_damage"
    if SCOPE_CHANGE.search(follow):
        return None, 0, "explicit_topic_switch"
    first_head = first.split("\n", 1)[0][:160]
    follow_head = follow.split("\n", 1)[0][:160]
    text_first = bool(ROUND2_TEXT_DELIVERABLE.search(first_head) or
                      HANDOFF_REWRITE.search(first_head) or HANDOFF_SIMPLIFY.search(first_head))
    doc_follow = bool(ROUND2_DOCUMENT_FOLLOWUP.search(follow_head) or DOCUMENT_REFERENCE.search(follow_head))
    doc_action = bool(DOCUMENT_ACTION.search(follow_head) or ROUND2_DOC_EVALUATE.search(follow_head))
    if doc_follow and doc_action and len(normalized_content(reply)) >= 35:
        return "document", 4 + int(text_first) + int(bool(DOCUMENT_ACTION.search(follow_head))), "prior_text_operation"
    if text_first or doc_follow or re.search(r"^(?:请|帮我)?\s*(?:总结|提取|找出|翻译|写出|选出|列出|摘要)", first_head):
        return None, 0, "text_deliverable_or_document_reference"
    if IMPLICIT_HEALTH_OR_RISK.search(first + "\n" + follow):
        return None, 0, "health_or_professional_risk"
    if len(first) > 180 or len(follow) > 130 or len(normalized_content(follow)) < 6:
        return None, 0, "not_a_short_clear_practical_followup"
    explicit = dependency(messages[:2], messages[2])
    implicit, _ = implicit_dependency(messages)
    setting = bool(ROUND2_SETTING.search(first))
    concrete = bool(ROUND2_CONCRETE.search(first))
    practical = bool(EVERYDAY.search(first) or IMPLICIT_PRACTICAL_CONTEXT.search(first) or setting)
    change = bool(ROUND2_CHANGE.search(follow))
    link = bool(ROUND2_FOLLOW_LINK.search(follow))
    useful_follow = bool(ROUND2_ACTION.search(follow) or change or re.search(r"[？?]", follow))
    if practical and useful_follow and (explicit or implicit or (link and (change or setting))):
        score = int(setting) * 3 + int(concrete) * 2 + int(link) * 2 + int(change) * 2
        score += int(bool(explicit)) * 3 + int(bool(implicit)) * 2 + int(len(follow) <= 45)
        cue = explicit[1] if explicit else "prior_action_set" if implicit else "practical_followup"
        return "daily", score, cue
    return None, 0, "no_visible_action_link"


def round2_group_keys(row):
    return {row[key] for key in ROUND2_FIELD_GROUPS if row.get(key)} | {
        source["source_group_id"] for source in row["sources"] if source.get("source_group_id")
    }


def round2_lexical_conflict(candidate, old_firsts, old_paths, protected, protected_paths):
    messages = candidate["original_messages"]
    bare = _first_material_question(messages)
    if bare:
        user = normalized_content(bare)
        for family, index in (("protected", protected), ("prior", old_firsts)):
            if hit := index.match(user):
                return f"{family}_material_question_{hit[0]}", hit[1]
    users, _ = user_keys(messages)
    for index, user in enumerate(users):
        trajectory = "|".join(users[:index + 1])
        if index == 0 or len(user) >= 24:
            for family, index_ in (("protected", protected), ("prior", old_firsts)):
                if hit := index_.match(user):
                    return f"{family}_user_{hit[0]}", hit[1]
        if trajectory in protected_paths:
            return "protected_exact_user_trajectory", protected_paths[trajectory]
        if index and (hit := old_paths.match(trajectory)):
            return f"prior_user_trajectory_{hit[0]}", hit[1]
    return None


def prepare_round2_frame(*, limit=None):
    if limit is None and any(path.exists() for path in (ROUND2_FRAME, ROUND2_SCAN, OUTPUT_ROUND2, ROUND2_REPORT)):
        raise FileExistsError("Refusing to overwrite frozen round2 inputs or results")
    groups, locators, old_firsts, old_paths, forbidden_docs, old_seen = known_handoff_groups(include_handoff=True)
    prior = prior_frame()
    groups.update(prior.groups)
    locators.update(prior.locators)
    for text, identifier, group in prior.first_users.records:
        old_firsts.add(text, identifier, group)
    for trajectory, identifier in prior.trajectories.items():
        old_paths.add(trajectory, identifier, "prior")
    protected, protected_paths, protected_count = load_protected(DEFAULT_PROTECTED)
    counts, candidates, defective_groups, exclusions = Counter(), defaultdict(list), set(), []
    rng = random.Random(ROUND2_SEED)

    def excluded(row, reason, conflict=None):
        counts[reason] += 1
        if len(exclusions) < 250:
            exclusions.append({"id": row["id"], "source_group": row["source_group"],
                               "reason": reason, "conflicts_with": conflict})

    for candidate in load_candidates(OUTPUT_CANDIDATES):
        counts["catalog_rows_read"] += 1
        sources = candidate["sources"]
        documents = cmrc_document_keys(candidate)
        group_keys = round2_group_keys(candidate)
        if findings := cmrc_material_findings(candidate):
            defective_groups.update(group_keys)
            forbidden_docs.update(documents)
            excluded(candidate, "cmrc_material_defect", findings)
        elif (group_keys & groups or documents & forbidden_docs or any(
            source_key(source) in locators or (source["input"] != "moss_honesty" and
             locator_key(source) in prior.coarse_locators) for source in sources
        )):
            excluded(candidate, "prior_source_group_locator_or_document")
        else:
            first = candidate["original_messages"][0]["content"]
            hints = candidate["possible_sampling_hints"]
            searches = []
            if "natural" in hints and not (ROUND2_TEXT_DELIVERABLE.search(first) or ROUND2_BROKEN_USER.search(first)
                                            or IMPLICIT_HEALTH_OR_RISK.search(first)):
                searches.append(("natural", 2 * int(bool(ROUND2_SETTING.search(first))), "situated_practical_request"))
            if "grounded" in hints:
                searches.append(("grounded", int(bool(documents)), "visible_source_material"))
            if "instruction" in hints:
                searches.append(("instruction", 1, "requested_text_operation"))
            if len(candidate["original_messages"]) == 4:
                lane, score, cue = round2_multi_signal(candidate)
                counts[f"multi_screen_{cue}"] += 1
                if lane:
                    searches.append((lane, score, cue))
            if searches:
                candidate["validation_challenges"] = round2_instruction_challenges(candidate)
                for lane, score, cue in searches:
                    cohort = ((bool(documents),) if lane == "grounded" else
                              tuple(tag for tag in ("simplify", "full_rewrite", "formalize")
                                    if tag in candidate["validation_challenges"]) if lane == "instruction" else ())
                    source = sources[0]["input"]
                    candidates[(lane, source, cohort)].append((score, rng.random(), cue, candidate))
        if limit is not None and counts["catalog_rows_read"] >= limit:
            break
        if counts["catalog_rows_read"] % 50_000 == 0:
            print(f"Read {counts['catalog_rows_read']:,} existing catalogue rows.", flush=True)

    frame, eligible = {}, Counter()
    limits = {"natural": 22, "grounded": 22, "instruction": 24, "daily": 100, "document": 16}
    checked = {}
    for (lane, _, _), rows in sorted(candidates.items()):
        eligible[lane] += len(rows)
        shortlisted = 0
        for score, _, cue, candidate in sorted(rows, reverse=True):
            identifier = candidate["id"]
            if identifier not in checked:
                if round2_group_keys(candidate) & defective_groups or cmrc_document_keys(candidate) & forbidden_docs:
                    checked[identifier] = ("damaged_or_prior_connected_document", None)
                else:
                    checked[identifier] = round2_lexical_conflict(
                        candidate, old_firsts, old_paths, protected, protected_paths)
                if checked[identifier]:
                    excluded(candidate, *checked[identifier])
            if checked[identifier]:
                continue
            item = frame.setdefault(identifier, {**candidate, "round2_sampling_cues": {}})
            item["round2_sampling_cues"][lane] = {"score": score, "cue": cue}
            shortlisted += 1
            if shortlisted >= limits[lane]:
                break
    summary = {
        "status": "local_shortlist_only", "catalogue": relative(OUTPUT_CANDIDATES),
        "catalogue_bytes": OUTPUT_CANDIDATES.stat().st_size,
        "previous_micro_raw_rows_indexed": old_seen,
        "previous_frames": {relative(OUTPUT_MICRO): 64, relative(OUTPUT_HANDOFF): 64, **dict(prior.files)},
        "protected_projection": relative(DEFAULT_PROTECTED), "protected_user_trajectories": protected_count,
        "prechecks": counts, "route_cues_before_shortlist_checks_nonadditive": eligible,
        "shortlisted_by_local_cue_nonadditive": Counter(lane for row in frame.values() for lane in row["round2_sampling_cues"]),
        "shortlisted_distinct_candidates": len(frame), "exclusion_examples_bounded": exclusions,
    }
    if limit is None:
        write_rows(ROUND2_FRAME, frame.values())
        write_json(ROUND2_SCAN, summary)
    print(json.dumps({key: summary[key] for key in (
        "prechecks", "route_cues_before_shortlist_checks_nonadditive", "shortlisted_by_local_cue_nonadditive",
    )}, ensure_ascii=False), flush=True)


def round2_regression_exits():
    path = SCALE_REPORT / "revision_handoff_semantic_review_v1.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    topics = {
        "6f82dbae": "指定段落按交付归类，并区分静态冥想与涉及体式的益处",
        "28bb9471": "正式化保留邀请方同时是活动举办方",
        "0e27be3b": "保留竞争力与学习为必要条件，不强化为竞争优势",
        "243a612e": "保留设备承担的职能，不弱化为仅具能力",
    }
    records = []
    for row in old["records"]:
        stem = row["id"].removeprefix("revision-")[:8]
        if stem in ROUND1_EXIT_STEMS:
            records.append({
                "id": row["id"], "source_group": row["source_group"],
                "old_grade": row["grade"], "old_content_grade": row["content_grade"],
                "old_category_dependency_grade": row["category_dependency_grade"],
                "v6_rule": topics[stem], "decision": "exclude_from_future_training_candidates",
                "verification": "static_prompt_and_exit_review_only",
            })
    _require(len(records) == len(ROUND1_EXIT_STEMS), "Prior regression records have changed")
    return relative(path), records


def build_round2(selection_path):
    outputs = (OUTPUT_ROUND2, ROUND2_REPORT, ROUND2_REPORT.with_suffix(".txt"))
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen round2 output {path}")
    frame = {row["id"]: row for row in load_candidates(ROUND2_FRAME)}
    scan = json.loads(ROUND2_SCAN.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _require(isinstance(selection, dict) and set(selection) == set(ROUND2_TARGETS),
             "Round2 selection must list every frozen lane")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_special_ids(tokenizer)
    _, protected_paths, _ = load_protected(DEFAULT_PROTECTED)
    protected_multi = LexicalIndex()
    for trajectory, identifier in protected_paths.items():
        protected_multi.add(trajectory, identifier, "protected")
    prior_review, exits = round2_regression_exits()
    exit_ids = {item["id"] for item in exits}
    exit_groups = {item["source_group"] for item in exits}
    sampler = MicroSampler(ROUND2_SEED)
    groups, locators, documents = set(), set(), set()
    raw, ledger, exclusions = [], [], []
    accepted_counts = Counter()
    for lane, target in ROUND2_TARGETS.items():
        entries = _entry_list(selection[lane], ("id", "note"), f"round2 {lane} selection")
        _require(len(entries) <= target, f"Round2 {lane} exceeds its frozen quota")
        for entry in entries:
            identifier, note = entry["id"], entry["note"]
            _require(_text(identifier) and _text(note) and identifier in frame, "Invalid curated round2 entry")
            row = frame[identifier]
            _require(lane in row["round2_sampling_cues"], f"{identifier} was not shortlisted in {lane}")
            row_groups = round2_group_keys(row)
            row_locators = {source_key(source) for source in row["sources"]}
            row_documents = cmrc_document_keys(row)
            messages = row["original_messages"]
            _, trajectory = user_keys(messages)
            protected_hit = protected_multi.match(trajectory) if len(messages) >= 4 else None
            reason = None
            if identifier in exit_ids or row_groups & exit_groups:
                reason = "old_regression_exit"
            elif row_groups & groups or row_locators & locators or row_documents & documents:
                reason = "duplicate_source_group_locator_or_document_in_batch"
            elif protected_hit:
                reason = f"protected_user_trajectory_{protected_hit[0]}"
            elif not sampler.accept(row):
                reason = "new_batch_lexical_conflict"
            if reason:
                exclusions.append({"id": identifier, "lane": lane, "reason": reason,
                                   "conflicts_with": protected_hit[1] if protected_hit else None})
                continue
            _require(candidate_prefix_options(tokenizer, messages) == row["prefix_options"],
                     f"{identifier} original prefix choices or tokenizer budget differ")
            _require(not cmrc_material_findings(row), f"{identifier} source rendering conflicts with the scan")
            groups.update(row_groups)
            locators.update(row_locators)
            documents.update(row_documents)
            hint = "multi" if lane in ("daily", "document") else lane
            raw_row = {key: value for key, value in row.items() if key != "round2_sampling_cues"}
            raw_row["sampling_hint"] = hint
            _require("sampling_hint" not in admission_payload(raw_row), "Sampling hint leaked into admission")
            if hint == "multi":
                raw_row["multi_sampling_subtype"] = lane
            raw.append(raw_row)
            accepted_counts[lane] += 1
            ledger.append({"id": identifier, "lane": lane, "source_group": row["source_group"],
                           "source_locators": sorted(row_locators),
                           "prefix_options": row["prefix_options"], "sampling_note": note})
    shortfalls = {lane: target - accepted_counts[lane] for lane, target in ROUND2_TARGETS.items()
                  if accepted_counts[lane] < target}
    tags = Counter(tag for row in raw for tag in row["validation_challenges"])
    required_tags = {"clean_cmrc", "full_rewrite", "simplify", "formalize"}
    missing_tags = sorted(required_tags - tags.keys())
    ready = not shortfalls and not missing_tags
    sampler.rng.shuffle(raw)
    report = {
        "status": "raw_source_selection_not_adjudicated" if ready else "insufficient_frozen_raw_selection",
        "handoff_phase": "round2", "sampling_seed": ROUND2_SEED,
        "raw_input": relative(OUTPUT_ROUND2) if ready else None,
        "selection_input": relative(selection_path), "shortlist_input": relative(ROUND2_FRAME),
        "source_and_exclusion_scan": relative(ROUND2_SCAN),
        **{key: scan[key] for key in ("catalogue", "catalogue_bytes", "previous_micro_raw_rows_indexed",
                                      "previous_frames", "protected_projection", "protected_user_trajectories",
                                      "prechecks", "route_cues_before_shortlist_checks_nonadditive",
                                      "shortlisted_by_local_cue_nonadditive", "shortlisted_distinct_candidates")},
        "micro_count": len(raw), "micro_source_groups": len({row["source_group"] for row in raw}),
        "micro_by_hint": Counter(row["sampling_hint"] for row in raw),
        "multi_hints": {lane: accepted_counts[lane] for lane in ("daily", "document")},
        "micro_sources": Counter(row["sources"][0]["input"] for row in raw),
        "hint_shortfalls": shortfalls, "challenge_tags_nonadditive": tags,
        "challenge_tag_shortfalls": missing_tags, "selected_provenance": ledger,
        "curated_selection_exclusions": exclusions, "within_batch_lexical_conflicts": sampler.conflicts,
        "old_independent_review": prior_review, "old_failure_family_regression_exit": exits,
        "old_execution_chain_revise_count": 2, "handoff_chain_revise_count_before_round2": 1,
        "external_author_calls": 0, "student_model_inference": 0, "training": False, "new_corpus_download": False,
        "limitations": [
            "A sampling cue and a local curation note do not determine a semantic label or certify source integrity.",
            "Daily candidates are plausible original-history dependencies with no confirmed independent daily pass.",
            "Local lexical checks cover exact, containment and five-gram rules where indexed; semantic paraphrases are adjudicated later.",
            "The four previous failures have a static v6 rule mapping and an explicit exit. No v6 model reran their answers.",
            "A final independent 16 requires four rows per category and exactly two daily and two document multi rows.",
        ],
    }
    if ready:
        write_rows(OUTPUT_ROUND2, raw)
    write_json(ROUND2_REPORT, report)
    count_text = "、".join(f"{lane} {accepted_counts[lane]}/{target}" for lane, target in ROUND2_TARGETS.items())
    lines = [
        "交接链第二轮原题与排除记录",
        "",
        f"原题{('冻结' if ready else '缺额停止')}。{count_text}，独立来源组 {report['micro_source_groups']}。",
        f"本地 catalogue 正式流式读取 {scan['prechecks']['catalog_rows_read']:,} 条（{scan['catalogue_bytes']:,} 字节）。"
        f"另外有 2,000 条有界探路。实际入围供本地复核 {scan['shortlisted_distinct_candidates']} 条。",
        f"排除旧 64/96/24、诊断以及两批 micro 全部 {scan['previous_micro_raw_rows_indexed']} 条。"
        f"继承冻结用户投影 {scan['protected_user_trajectories']:,} 条，检查已知来源 locator、文档组、首问和完整用户轨迹。",
        "逐字保留原题、原始开头、全部可用 prefix 及来源和回复预算。源回复仅用于定位原指代。",
        "类别与多轮子类是本地取样提示，正式准入、改写、Checker 和独立语义门禁均待完成。",
        "至少要有 2 个日常和 2 个文档多轮进入独审并通过，旧轮尚无独审日常通过证据。",
        "",
        "四个旧失败案例仅做静态规则和退出对照，旧分数保留，未发 v6 重跑。",
        *[f"- {record['id']}，旧综合 {record['old_grade']}，内容 {record['old_content_grade']}。"
          f"{record['v6_rule']}。退出后续训练候选。"
          for record in exits],
        "",
        f"来源、逐题原始 locator 与预算见 {relative(ROUND2_REPORT)}。",
        f"逐类排除计数和可追溯样例见 {relative(ROUND2_SCAN)}。",
        "旧扩量链保留两次 REVISE。交接链本轮之前有一次 REVISE。本轮再 REVISE 或门槛不足则停止。",
        "本次外部作者调用、新语料下载、学生模型推理、训练均为零。",
    ]
    if not ready:
        lines.append(f"冻结配额缺额 {shortfalls}，缺少挑战题 {missing_tags}。未写新 raw 文件。")
    ROUND2_REPORT.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "micro_count", "micro_by_hint", "hint_shortfalls", "multi_hints", "micro_sources",
        "challenge_tags_nonadditive", "challenge_tag_shortfalls",
    )}, ensure_ascii=False))
    if not ready:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-limit", type=int, help="Bound accepted raw rows per input for an in-memory diagnostic")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--handoff", action="store_true", help="Sample an isolated new handoff batch from the existing catalog")
    mode.add_argument("--prepare-handoff-round2", action="store_true", help="Stream the catalogue into a bounded, local round2 review frame")
    mode.add_argument("--handoff-round2", type=Path, metavar="SELECTION", help="Freeze a curated round2 batch from the local frame")
    args = parser.parse_args()
    if args.quick_limit is not None and args.quick_limit < 1:
        raise ValueError("Quick limit must be positive")
    if args.prepare_handoff_round2:
        prepare_round2_frame(limit=args.quick_limit)
        return
    if args.handoff_round2:
        if args.quick_limit is not None:
            parser.error("--quick-limit applies to catalogue scanning only")
        build_round2(args.handoff_round2.resolve())
        return
    if args.handoff:
        build_handoff(limit=args.quick_limit)
        return
    if args.quick_limit is None:
        for path in (OUTPUT_CANDIDATES, OUTPUT_MICRO, SEED_REPORT, EXCLUSIONS_REPORT):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_special_ids(tokenizer)
    candidates, report, exclusions = build_catalog(tokenizer, limit=args.quick_limit)
    micro, micro_summary, conflicts = make_micro(candidates)
    report.update(micro_summary)
    report["micro_max_fixed_tokens"] = max((row["prefix_options"][-1]["fixed_tokens"] for row in micro), default=None)
    report["micro_min_assistant_budget"] = min(
        (row["prefix_options"][-1]["assistant_content_token_budget"] for row in micro), default=None)
    report["micro_old_prefix_over_512_retained"] = sum(
        len(encode_conversation(tokenizer, row["original_messages"], final_eos=False)["input_ids"]) > MAX_TOKENS
        for row in micro
    )
    exclusions["new_micro_lexical_conflicts"] = conflicts
    report["limits"] = [
        "Local sampling hints do not appear in admission payloads and are not adjudicated categories.",
        "Only available first and second complete exchanges were enumerated; the reply budget excludes old answers.",
        "Catalog uses normalized first questions, original groups, source aliases and exact full CMRC documents; its surviving rows may have lexical or semantic near neighbors.",
        "Lexical exact, containment and Jaccard checks protect this micro batch against itself and known old/frozen inputs; this is not a final corpus splitting gate.",
        "The old groups and reports listed here cannot establish novelty against untracked prior data or semantic paraphrases.",
    ]
    if args.quick_limit is None:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_rows(OUTPUT_CANDIDATES, candidates)
        write_rows(OUTPUT_MICRO, micro)
        write_json(SEED_REPORT, report)
        write_json(EXCLUSIONS_REPORT, exclusions)
    print(json.dumps({key: report[key] for key in (
        "catalog_rows", "catalog_distinct_groups", "local_hints_nonadditive", "micro_rows", "micro_by_hint", "hint_shortfalls",
        "micro_by_primary_input", "micro_cmrc_count", "micro_multi_hints_by_subtype",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
