"""Build the cleaned, globally shuffled corpus for the comparison pretraining run."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import time

import numpy as np
from tokenizers import Tokenizer

from .chat_format import SPECIAL_IDS, check_special_ids, contains_reserved
from .stage1_data import (
    EVAL_SUBSTRING_CHARS, HAN, REPEATED_CHARACTER, SPLITS, build_lock, clean_text, eval_matches, eval_prompt_keys,
    file_identity, heartbeat, json_line, normalized_content, write_json,
)


CORPUS_ID = "pretrain_clean_v2"
SEED = 42
QUALITY_SAMPLE_SEED = 20261122
MIN_TRAIN_TOKENS = 120_000_000
PREFIX_TOKENS = 100_000_000
COVERAGE_WINDOW = 10_000_000
EVAL_SHORT_SUBSTRING_LETTERS = 6
TOKENIZER_SHA256 = "1a03e395cbed783a213d145efff03b7534589ea8847b5009609dabe0d15845c5"
SHINGLE_SIZE = 5
BOTTOM_COUNT = 16
SHARED_KEYS = 2
MAX_POSTINGS = 256
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_COLUMNS = ("id", "url", "title", "text")
WIKI_LICENSES = ("CC-BY-SA-3.0", "GFDL")
WIKI_LICENSE_LINKS = ("https://creativecommons.org/licenses/by-sa/3.0/", "https://www.gnu.org/licenses/fdl-1.3.html")
WIKI_CARD = f"https://huggingface.co/datasets/wikimedia/wikipedia/tree/{WIKI_REVISION}/20231101.zh"
QUALITY_RULES = {
    "min_han": 120, "min_han_fraction_of_alphanumeric": 0.45, "max_document_bytes": 131072,
    "min_han_for_sentence_rich_text": 80, "min_han_fraction_for_sentence_rich_text": 0.40,
    "sentence_rich_minimum_punctuation_marks": 3,
    "max_document_characters": 40000, "max_repeated_line_characters_ratio": 0.30,
    "max_short_line_characters_ratio": 0.68, "malformed_markers_threshold": 2,
    "max_boilerplate_characters_ratio": 0.25, "max_url_count": 5,
    "repeated_reply_8gram_fraction_threshold": 0.10, "unsupported_software_fields_threshold": 4,
    "malformed_question_mark_sequences_threshold": 2, "news_blurb_quote_threshold": 8,
    "deposit_return_ad_minimum_investor_address_count": 2,
    "topic_block_minimum_characters": 100, "topic_minimum_document_characters": 550,
    "topic_maximum_disconnected_overlap": 0.025,
}

NAVIGATION = re.compile(
    r"^(?:首页|首頁|网站首页|網站首頁|上一页|下一页|上一頁|下一頁|返回首页|返回首頁|"
    r"返回顶部|返回頂部|手机端|手機端|收藏本站|打印本文|关闭窗口|關閉窗口|登录|登入|注册|"
    r"註冊|广告|廣告|相关文章|相關文章|热门推荐|熱門推薦|网站地图|網站地圖|展开全文|展開全文|"
    r"责任编辑|責任編輯|点击查看|點擊查看)[:：。\s]*$"
)
FOOTER = re.compile(
    r"^(?:版权声明|版權聲明|免责声明|免責聲明|版权所有|版權所有|Copyright\b|"
    r"\S{0,8}ICP备[\d-]+号|责任编辑[：:]|責任編輯[：:]|转载请注明|轉載請注明|"
    r"扫一扫关注|掃一掃關注|点击\S{0,8}(?:关注|領取|领取)|點擊\S{0,8}(?:關注|領取)|"
    r"下载APP|下載APP|立即咨询|立即諮詢|微信公众号[：:]|微信公眾號[：:])", re.IGNORECASE,
)
KEYWORDS = re.compile(r"^(?:关键[词字]|關鍵[詞字]|搜索关键词|搜尋關鍵字|标签|標籤)[：:]", re.MULTILINE)
COMMERCIAL = re.compile(r"(?:加微信|添加微信|微信[号號]|QQ[：:]|招收代理|送货上门|送貨上門|点击购买|點擊購買|"
                        r"立即下单|立即下單|货到付款|貨到付款|限时优惠|限時優惠|包邮|包郵|咨询热线|諮詢熱線)", re.IGNORECASE)
MALFORMED = re.compile(r"[�\ud800-\udfff\ue000-\uf8ff]|(?:锟斤拷|烫烫烫)|(?:&(?:nbsp|amp|lt|gt|quot);){2,}")
SURROGATE = re.compile(r"[\ud800-\udfff]")
URL = re.compile(r"(?:https?://|www\.)[^\s<>，。]+", re.IGNORECASE)
SENTENCE_END = re.compile(r"[。！？；.!?;]")
REFERENCE_REPLIES = re.compile(r"(?:参考回复|參考回覆)[：:]|(?:其它|其他)回[复覆]")
SOFTWARE_FIELDS = re.compile(r"^(功能|使用说明|使用說明|软件介绍|軟件介紹|软件名称|軟件名稱|主要功能)[：:]", re.MULTILINE)
SOFTWARE_CONTENT = re.compile(r"(?:下载|下載|安装|安裝|软件|軟件|版本|可执行文件|設定|系统|系統|插件|應用程式|界面|"
                              r"Windows|Linux|Android|iOS|macOS)", re.IGNORECASE)
MALFORMED_QUESTIONS = re.compile(r"[?？]\s*(?:吗|嗎|如何|什[么麼])\s*[?？]")
NEWS_SOURCE = re.compile(r"^(?:来源|來源|稿源)[：:]")
DIRECT_QUOTES = re.compile(r"[“”「」]")
DAILY_FIXED_RETURN = re.compile(r"(?:每日|每天|日均)[^。！？；\n]{0,20}(?:固定收益|固定回[报報]|固定報酬|保本|[稳穩][赚賺])")
SIGNUP_DEPOSIT = re.compile(r"(?:注册|註冊)[^。！？；\n]{0,8}(?:充值|儲值|入金)")
ADDRESSES_INVESTOR = re.compile(r"(?:每位投[资資]者|投[资資]者(?:在|当天|當天))")
HTML_EDGE = re.compile(r"^(?:[\w-]{8,}(?:\s*=\s*['\"])?>|[\w-]{8,}['\"]>)\s*|['\"]>\s*$")
WEB_BYLINE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+[\d:]+\s*(?:来源|來源)[：:]")
FAKE_COUNTRY = re.compile(r"(?:中华夏(?:儿女|兒女|族)民|中夏族民|中华儿女民)共和国")
SPUN_PHRASES = re.compile(r"(?:并异国|並異國|睁开全文|睜開全文|宠喜欢|寵喜歡|之后生可畏|之风姿洒脱|"
                         r"排郁闷解难|别名矮级|首点虽矮|除夜[学家国學國量大门門]|不问不闻时期)")
BROKEN_FOREIGN_NAME = re.compile(r"[A-Z]{2,}[?？][A-Z]{2,}")
BETTING = re.compile(r"(?:博彩|赌场|賭場|百家乐|百家樂|在线投注|在線投注|投注平台|体育投注|體育投注|"
                     r"让球|讓球|盘口|盤口|赔率|賠率|串关|串關|大小球|竞彩|競彩|下注|买码|買碼)")
GAMBLING_PROMO = re.compile(r"(?:(?:注册|註冊|开户|開戶)[^，。；！？\n]{0,8}(?:领取|領取|赠送|贈送|享受|返利)|"
                            r"送彩金|送彩金|赠彩金|贈彩金|返利|充值返现|儲值返現|即时提现|即時提現)")
LOTTERY_CODES = re.compile(r"(?:热[门碼码]号|熱門號|冷[门碼码]号|冷門號|[冷热熱](?:代[码碼]|号[码碼])|[胆膽](?:汁)?代[码碼]|单双球|單雙球)")
LOCKED_EXAM = re.compile(r"(?:题目|題目|试题|試題|题干|題幹|考核|试卷|試卷)")
PAYWALL = re.compile(r"(?:查看价格|查看價格|付费查看|付費查看|售价|售價|价格为|價格為|购买答案|購買答案|需支付)")
PAYWALL_CTA = re.compile(r"(?:金币|金幣|积分|積分|微信客服|请先登录|請先登入|开通会员|開通會員)")
SEVERE_ILLNESS = re.compile(r"(?:肿瘤|腫瘤|癌症|恶性癌|惡性癌|艾滋病|愛滋病|白血病)")
MARKETING_HEALTH = re.compile(r"(?:消费[者群]|消費[者群]|使用者|神奇|超[千万千萬\d]+人|有效率|代理|订购|訂購)")
HEALTH_PRODUCT_SALES = re.compile(r"(?:消费[者群]|消費[者群]|本产品|本產品|产品代理|產品代理|品牌代理|订购|訂購|下单|下單)")
TREATMENT_PROMISE = re.compile(r"(?:延长.{0,8}(?:存活|生命)|延長.{0,8}(?:存活|生命)|杀死癌细胞|殺死癌細胞|"
                               r"治[愈癒]|恢复.{0,8}免疫|恢復.{0,8}免疫|(?:改善|提高).{0,8}生存质量)")
NEGATED_PROMISE = re.compile(r"(?:不能|不足以|无法|無法|没有|沒有|未能|并非|並非|无证据|無證據|否认|否認)"
                             r"[^，。；！？\n]{0,8}$")
MEDIA_SEO = re.compile(r"(?:免费在线观看|免費在線觀看|在线播放|在線播放|app下载|app下載|免安装|免安裝|高清下载|高清下載)", re.IGNORECASE)
ADULT_SEO = re.compile(r"(?:邪恶acg|邪惡acg|成人视频|成人視頻|成人影片|无码影片|AV片|色情电影|色情電影)", re.IGNORECASE)
ARTICLE_RESET = re.compile(r"^(?:[“「][^”」]{3,36}[”」][，。！？!]?\s*)?(?:近日|日前|北京时间|北京時間|"
                           r"原标题|原標題|(?:中新网|中新網|新华社|新華社)\S{0,8}电|据\S{0,12}(?:报道|消息)|"
                           r"\d{4}年.{0,12}(?:消息|报道|報道))")
FIXTURE_TITLE = re.compile(r"(?:vs\.?|对阵|對陣|迎战|迎戰)", re.IGNORECASE)
FIXTURE_SWITCH = re.compile(r"(?:赛季首次较量|賽季首次較量|阵容方面|陣容方面|联赛球队|聯賽球隊|全场比赛|全場比賽|"
                            r"赛前分析|賽前分析|正式宣布|比赛结束|比賽結束)")
COLLECTION_TITLE = re.compile(r"(?:汇总|匯總|汇编|彙編|盘点|盤點|合集|精选|精選|回顾|回顧|资料整理|資料整理|问答|問答|访谈|訪談)")


def digest(value):
    return sha256(value.encode("utf-8")).digest()


def seeded_key(purpose, record_id):
    return digest(f"{SEED}:{purpose}:{record_id}")


def document_id(corpus, source, group, row):
    return digest(f"{corpus}\n{source}\n{group}\n{row}").hex()


def strip_navigation(text):
    lines = text.split("\n")
    retained, removed, commercial = [], 0, 0
    for index, line in enumerate(lines):
        if index < 5 and WEB_BYLINE.match(line) and len(line) <= 140:
            removed += len(line)
        elif len(line) <= 28 and NAVIGATION.fullmatch(line):
            removed += len(line)
        elif len(line) <= 180 and (index < 4 or index >= len(lines) - 5) and FOOTER.match(line):
            removed += len(line)
        else:
            if index < 5:
                cleaned = HTML_EDGE.sub("", line).strip()
                removed += len(line) - len(cleaned)
                line = cleaned
            if line:
                retained.append(line)
        if len(line) <= 180 and COMMERCIAL.search(line):
            commercial += 1
    return retained, removed, commercial


def quality_filter(raw):
    if not isinstance(raw, str) or len(raw) > QUALITY_RULES["max_document_characters"]:
        return "invalid_or_oversize", None, 0.0, 0
    text = clean_text(raw)
    if len(text.encode("utf-8", errors="surrogatepass")) > QUALITY_RULES["max_document_bytes"]:
        return "invalid_or_oversize", None, 0.0, 0
    if contains_reserved(text):
        return "reserved_tokens", None, 0.0, 0
    if len(MALFORMED.findall(text)) >= QUALITY_RULES["malformed_markers_threshold"] or SURROGATE.search(text):
        return "malformed_text", None, 0.0, 0
    if REPEATED_CHARACTER.search(text):
        return "repeated_characters", None, 0.0, 0

    retained, removed, commercial = strip_navigation(text)
    if removed > QUALITY_RULES["max_boilerplate_characters_ratio"] * len(text):
        return "navigation_or_boilerplate", None, 0.0, removed

    text = "\n".join(retained)
    length = len(text)
    if quality_reason := reject_low_quality(text, retained, commercial):
        return quality_reason, None, 0.0, removed

    # Prefer documents with a smaller proportion of removable boilerplate.
    score = round(1 - removed / (length + removed), 6)
    return None, text, score, removed


def reject_low_quality(text, lines, commercial):
    alnum = normalized_content(text)
    han_count = len(HAN.findall(alnum))
    sentences = len(SENTENCE_END.findall(text))
    if sentences >= QUALITY_RULES["sentence_rich_minimum_punctuation_marks"]:
        min_han = QUALITY_RULES["min_han_for_sentence_rich_text"]
        min_fraction = QUALITY_RULES["min_han_fraction_for_sentence_rich_text"]
    else:
        min_han = QUALITY_RULES["min_han"]
        min_fraction = QUALITY_RULES["min_han_fraction_of_alphanumeric"]
    if han_count < min_han or han_count < min_fraction * len(alnum):
        return "insufficient_chinese"
    line_lengths = Counter(line for line in lines if len(line) >= 12)
    if sum((count - 1) * len(line) for line, count in line_lengths.items()) > (
        QUALITY_RULES["max_repeated_line_characters_ratio"] * len(text)
    ):
        return "repeated_lines"
    short_chars = sum(len(line) for line in lines if len(line) <= 18)
    if len(lines) >= 12 and short_chars > QUALITY_RULES["max_short_line_characters_ratio"] * len(text) and sentences <= 2:
        return "short_line_list"
    keywords = [line for line in lines if KEYWORDS.match(line)]
    if len(keywords) >= 3 or any(len(line) >= 90 and len(re.findall(r"[、，,|｜]", line)) >= 8 for line in keywords):
        return "keyword_stuffing"
    if len(URL.findall(text)) > QUALITY_RULES["max_url_count"] and (commercial >= 2 or sentences <= 4):
        return "link_directory"
    if commercial >= 4 and len(set(match.group().casefold() for match in COMMERCIAL.finditer(text))) >= 3 and sentences <= 7:
        return "advertising_copy"
    if DAILY_FIXED_RETURN.search(text) and SIGNUP_DEPOSIT.search(text) and (
        len(ADDRESSES_INVESTOR.findall(text)) >= QUALITY_RULES["deposit_return_ad_minimum_investor_address_count"]
    ):
        return "deposit_return_ad"
    if reason := reject_distinctive_noise(text, alnum):
        return reason
    software_fields = set(SOFTWARE_FIELDS.findall(text))
    if len(software_fields) >= QUALITY_RULES["unsupported_software_fields_threshold"]:
        description = SOFTWARE_FIELDS.sub("", text)
        if len(SOFTWARE_CONTENT.findall(description)) < 2:
            return "unsupported_software_fields"
    if len(REFERENCE_REPLIES.findall(text)) >= 2:
        prefix = alnum[:3000]
        positions = len(prefix) - 7
        unique = len({prefix[i:i + 8] for i in range(positions)})
        if 1 - unique / positions >= QUALITY_RULES["repeated_reply_8gram_fraction_threshold"]:
            return "repeated_reference_replies"
    if len(MALFORMED_QUESTIONS.findall(text)) >= QUALITY_RULES["malformed_question_mark_sequences_threshold"]:
        return "malformed_question_sequences"
    title = lines[0]
    if 12 <= len(title) <= 55 and len(lines) >= 5:
        source_index = next((index for index in range(1, 4) if NEWS_SOURCE.match(lines[index])), None)
        if source_index is not None:
            body = "\n".join(lines[source_index + 1:])[:2000]
            if len(DIRECT_QUOTES.findall(body)) >= QUALITY_RULES["news_blurb_quote_threshold"]:
                title_words = normalized_content(title)
                normalized_body = normalized_content(body)
                title_bigrams = {title_words[i:i + 2] for i in range(len(title_words) - 1)}
                linked_bigrams = sum(word in normalized_body for word in title_bigrams)
                if linked_bigrams < 2 and not any(title_words[i:i + 3] in normalized_body for i in range(len(title_words) - 2)):
                    return "unrelated_news_blurb"
    if has_spliced_topics(text, lines):
        return "unrelated_spliced_sections"
    return None


def reject_distinctive_noise(text, alnum):
    spun = SPUN_PHRASES.findall(text)
    if FAKE_COUNTRY.search(text) or (len(spun) >= 3 and len(set(spun)) >= 2):
        return "seo_synonym_spinning"
    if len(BROKEN_FOREIGN_NAME.findall(text)) >= 2 and text.count("？") + text.count("?") >= 5:
        return "malformed_foreign_names"
    betting = set(BETTING.findall(text))
    if len(betting) >= 2 and GAMBLING_PROMO.search(text):
        return "gambling_promotion"
    if FIXTURE_TITLE.search(text.split("\n", 1)[0]) and len(set(LOTTERY_CODES.findall(text))) >= 2:
        return "sports_lottery_splice"
    if len(alnum) <= 700 and LOCKED_EXAM.search(text) and PAYWALL.search(text) and PAYWALL_CTA.search(text):
        return "locked_exam_fragment"
    if SEVERE_ILLNESS.search(text) and HEALTH_PRODUCT_SALES.search(text) and len(MARKETING_HEALTH.findall(text)) >= 3:
        for promise in TREATMENT_PROMISE.finditer(text):
            preceding = text[max(0, promise.start() - 16):promise.start()]
            if not NEGATED_PROMISE.search(preceding):
                return "serious_illness_product_promotion"
    if MEDIA_SEO.search(text) and ADULT_SEO.search(text):
        return "adult_media_seo"
    return None


def has_spliced_topics(text, lines):
    if len(text) < QUALITY_RULES["topic_minimum_document_characters"] or len(lines) < 4:
        return False
    title = lines[0]
    if not 8 <= len(title) <= 75 or COLLECTION_TITLE.search(title) or SENTENCE_END.search(title):
        return False
    tail = lines[2:]
    fixture = FIXTURE_TITLE.search(title)
    signals = (ARTICLE_RESET, FIXTURE_SWITCH) if fixture else (ARTICLE_RESET,)
    resets = [index for index, line in enumerate(tail) if len(line) >= QUALITY_RULES["topic_block_minimum_characters"]
              and any(signal.search(line[:150]) for signal in signals)]
    if not resets:
        return False
    title_key = normalized_content(title)
    title_grams = {title_key[index:index + 3] for index in range(len(title_key) - 2)}
    lead = normalized_content("\n".join(lines[1:])[:650])
    if sum(gram in lead for gram in title_grams) < 2:
        return False
    disconnected = 0
    for index in resets:
        block = normalized_content(tail[index][:700])
        if any(gram in block for gram in title_grams):
            continue
        block_grams = {block[offset:offset + 3] for offset in range(len(block) - 2)}
        overlap = sum(gram in lead for gram in block_grams) / max(len(block_grams), 1)
        if overlap < QUALITY_RULES["topic_maximum_disconnected_overlap"]:
            disconnected += 1
    return disconnected >= 2


def bottom_shingles(text):
    """Hash the exact set of character 5-grams, then retain the 16 lowest hashes."""
    code_points = np.frombuffer(text.encode("utf-32le"), dtype="<u4").astype(np.uint64)
    count = code_points.size - SHINGLE_SIZE + 1
    if count < 1:
        return 0, ()
    values = code_points[:count].copy()
    for offset in range(1, SHINGLE_SIZE):
        values *= np.uint64(1_000_003)
        values += code_points[offset:offset + count]
    # An invertible 64-bit mixer avoids a lexical bias in the bottom hashes.
    values ^= values >> 30
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> 27
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> 31
    unique = np.unique(values)
    return unique.size, tuple(unique[:BOTTOM_COUNT].tolist())


def shingles(text):
    return {text[index:index + SHINGLE_SIZE] for index in range(len(text) - SHINGLE_SIZE + 1)}


def sample_record(record, head=1000, tail=500):
    text = record["text"] if isinstance(record["text"], str) else repr(record["text"])
    safe = text.encode("utf-8", errors="replace").decode("utf-8")
    excerpt = safe if len(safe) <= head + tail else f"{safe[:head]}\n[省略]\n{safe[-tail:]}"
    return {**record, "text": excerpt, "text_characters": len(text)}


def comparison_eval_matches(keys):
    ordinary = eval_matches(keys)
    short_keys = tuple(sorted(key for key in keys if len(key) < EVAL_SUBSTRING_CHARS
                              and sum(char.isalpha() for char in key) >= EVAL_SHORT_SUBSTRING_LETTERS))

    def match(normalized):
        if result := ordinary(normalized):
            return result
        if any(key in normalized for key in short_keys):
            return "evaluation_short_substring_overlap"
        return None

    return match


@dataclass(slots=True)
class Document:
    record: dict
    quality: float = 0.0
    old_holdout: bool = False
    shingle_count: int = 0
    evaluation_conflict: str | None = None
    token_bytes: bytes | None = None

    @property
    def id(self):
        return self.record["id"]

    @property
    def text(self):
        return self.record["text"]

    @property
    def source(self):
        return self.record["source"]


class DuplicateIndex:
    def __init__(self):
        self.documents = []
        self.parents = []
        self.exact = {}
        self.postings = {}
        self.counts = Counter()
        self.examples = defaultdict(list)

    def root(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def connect(self, current, previous):
        first, second = self.root(current), self.root(previous)
        if first != second:
            self.parents[max(first, second)] = min(first, second)

    def join(self, current, previous, match, jaccard):
        self.connect(current, previous)
        a, b = self.documents[current], self.documents[previous]
        if a.id == b.id:
            return
        kind = f"old_holdout_{match}" if a.old_holdout != b.old_holdout else match
        self.counts[kind] += 1
        if len(self.examples[kind]) < 12:
            self.examples[kind].append({"id": a.id, "other_id": b.id, "jaccard": round(jaccard, 6),
                                        "source": a.source, "other_source": b.source,
                                        "text": a.text[:360], "other_text": b.text[:360]})

    def add(self, document):
        index = len(self.documents)
        self.documents.append(document)
        self.parents.append(index)
        normalized = normalized_content(document.text)
        exact_key = digest(normalized)
        existing = self.exact.get(exact_key)
        if existing is not None:
            self.join(index, existing, "exact", 1.0)
            return index
        self.exact[exact_key] = index
        document.shingle_count, keys = bottom_shingles(normalized)
        candidates = Counter()
        for key in keys:
            posting = self.postings.get(key, ())
            if posting is not None:
                candidates.update(posting)
        current_set = None
        for candidate, count in candidates.items():
            if count < SHARED_KEYS:
                continue
            other = self.documents[candidate]
            low, high = sorted((document.shingle_count, other.shingle_count))
            if 5 * low < 4 * high:
                continue
            self.counts["verified_candidates"] += 1
            if current_set is None:
                current_set = shingles(normalized)
            previous_set = shingles(normalized_content(other.text))
            overlap = len(current_set & previous_set)
            union = len(current_set) + len(previous_set) - overlap
            if union and 5 * overlap >= 4 * union:
                self.join(index, candidate, "near", overlap / union)
        for key in keys:
            posting = self.postings.get(key)
            if posting is None:
                if key not in self.postings:
                    self.postings[key] = [index]
            elif len(posting) >= MAX_POSTINGS:
                self.postings[key] = None
                self.counts["suppressed_frequent_keys"] += 1
            else:
                posting.append(index)
        return index

    def select(self):
        groups = defaultdict(list)
        for index, document in enumerate(self.documents):
            groups[self.root(index)].append(document)
        kept = {split: [] for split in SPLITS}
        discarded = defaultdict(Counter)
        summaries = []
        for group in groups.values():
            eligible = [document for document in group if not document.old_holdout]
            if not eligible:
                continue
            reason = None
            if len(eligible) != len(group):
                reason = "old_holdout_duplicate_component"
            elif any(document.evaluation_conflict for document in group):
                reason = "evaluation_conflict_component"
            if reason is not None:
                for document in eligible:
                    discarded[document.source][reason] += 1
                continue
            anchor = min(doc.id for doc in group)
            bucket = int.from_bytes(seeded_key("split", anchor)[:8], "big") % 10_000
            split = SPLITS[(bucket >= 9800) + (bucket >= 9900)]
            chosen = min(group, key=lambda doc: (-doc.quality, -len(doc.text), doc.id))
            kept[split].append(chosen)
            if len(group) > 1:
                if len(summaries) < 24:
                    summaries.append({"anchor": anchor, "representative": chosen.id, "documents": len(group),
                                      "representative_source": chosen.source, "split": split})
                for doc in group:
                    if doc is not chosen:
                        discarded[doc.source]["duplicate_component_nonrepresentative"] += 1
        return kept, discarded, summaries


class CorpusBuild:
    def __init__(self, reviewed_exclusions=None):
        self.index = DuplicateIndex()
        self.counts = defaultdict(Counter)
        self.samples = defaultdict(list)
        self.evaluation_paths = []
        self.evaluation_keys = set()
        self.evaluation_match = None
        self.reviewed_exclusions = reviewed_exclusions or {}

    def sample(self, reason, record):
        sample_key = seeded_key("quality_sample", record["id"])
        group = self.samples[(record["source"], reason)]
        if len(group) < 4 or sample_key < group[-1][0]:
            group.append((sample_key, sample_record(record)))
            group.sort(key=lambda sample: sample[0])
            del group[4:]

    def find_eval_conflict(self, text):
        if self.evaluation_match is not None:
            return self.evaluation_match(normalized_content(text))
        return None

    def add_eval_prompts(self, paths, progress):
        if not paths:
            return
        for path in paths:
            keys, identity = eval_prompt_keys(path)
            self.evaluation_paths.append({**identity, "substring_min_chars": EVAL_SHORT_SUBSTRING_LETTERS,
                                          "short_substring_requires_letter_count": EVAL_SHORT_SUBSTRING_LETTERS,
                                          "limitation": "Full normalized user strings only. Shorter generic strings are "
                                                        "compared for whole-document equality but not scanned as substrings."})
            self.evaluation_keys.update(keys)
        self.evaluation_match = comparison_eval_matches(self.evaluation_keys)
        progress("screening existing documents against evaluation prompts")
        for document in self.index.documents:
            if not document.old_holdout and not document.evaluation_conflict:
                document.evaluation_conflict = self.find_eval_conflict(document.text)
                if document.evaluation_conflict:
                    self.sample(document.evaluation_conflict, document.record)

    def add(self, record, old_holdout=False):
        counts = self.counts[record["source"]]
        if old_holdout:
            counts["old_holdout_indexed"] += 1
            original = self.index.add(Document(record, old_holdout=True))
            retained_lines, removed, _ = strip_navigation(clean_text(record["text"]))
            if retained_lines and removed:
                view = {**record, "text": "\n".join(retained_lines), "old_holdout_view": "navigation_stripped"}
                self.index.connect(self.index.add(Document(view, old_holdout=True)), original)
                counts["old_holdout_navigation_stripped_views"] += 1
            return
        counts["input_documents"] += 1
        if note := self.reviewed_exclusions.get(record.get("legacy_id")):
            counts["legacy_audit_poor"] += 1
            self.sample("legacy_audit_poor", {**record, "legacy_audit_note": note})
            return
        reason, text, quality, removed = quality_filter(record["text"])
        if reason:
            counts[reason] += 1
            self.sample(reason, record)
            return
        record["text"] = text
        reason = self.find_eval_conflict(text)
        if reason:
            self.sample(reason, record)
        counts["passes_quality"] += 1
        if removed:
            counts["documents_trimmed"] += 1
            counts["boilerplate_characters_removed"] += removed
        self.index.add(Document(record, quality, evaluation_conflict=reason))

    def add_legacy(self, directory, progress):
        for split in ("validation", "test", "train"):
            path = directory / f"{split}.jsonl"
            ordinal = 0
            with path.open(encoding="utf-8") as handle:
                for ordinal, line in enumerate(handle, start=1):
                    record = json.loads(line)
                    record["legacy_id"] = record["id"]
                    record["legacy_split"] = split
                    record["id"] = document_id("stage1_v1", record["source"], record["row_group"], record["row_in_group"])
                    self.add(record, old_holdout=split != "train")
                    if ordinal % 4096 == 0:
                        progress(f"indexed {split}: {ordinal:,} documents")
            progress(f"indexed {split}: {ordinal:,} documents (complete)")

    def select_and_encode(self, tokenizer, progress):
        selected, discarded, groups = self.index.select()
        uncached = [doc for docs in selected.values() for doc in docs if doc.token_bytes is None]
        progress(f"encoding {len(uncached):,} retained documents with the frozen tokenizer")
        for offset in range(0, len(uncached), 128):
            batch = uncached[offset:offset + 128]
            encodings = tokenizer.encode_batch([doc.text for doc in batch], add_special_tokens=False)
            for doc, encoding in zip(batch, encodings, strict=True):
                doc.token_bytes = np.asarray([*encoding.ids, SPECIAL_IDS["<|eos|>"]], dtype="<u2").tobytes()
            if offset and offset % 8192 == 0:
                progress(f"encoded {offset:,}/{len(uncached):,} retained documents")
        tokens = sum(len(doc.token_bytes) // 2 for doc in selected["train"])
        progress(f"selected train: {len(selected['train']):,} documents, {tokens:,} tokens")
        return selected, discarded, groups, tokens

    def data_report(self, discarded, groups):
        return {
            "quality_rules": QUALITY_RULES,
            "source_ingest": {source: dict(counts) for source, counts in sorted(self.counts.items())},
            "source_duplicate_or_eval_discards": {source: dict(counts) for source, counts in sorted(discarded.items())},
            "duplicate_edges": dict(self.index.counts),
            "duplicate_examples": dict(self.index.examples),
            "duplicate_components_sample": groups,
            "rejected_samples": [
                {"source": source, "reason": reason, "records": [excerpt for _, excerpt in samples]}
                for (source, reason), samples in sorted(self.samples.items())
            ],
            "evaluation_screen": self.evaluation_paths,
            "evaluation_match_reasons": dict(Counter(document.evaluation_conflict for document in self.index.documents
                                                    if document.evaluation_conflict)),
            "legacy_quality_exclusions": self.reviewed_exclusions,
        }


def legacy_quality_exclusions(review_path, samples_path):
    if review_path is None and samples_path is None:
        return {}
    if review_path is None or samples_path is None:
        raise ValueError("Legacy quality exclusion requires both review and samples")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    if review["execution_pass"] is not True:
        raise ValueError("Legacy quality audit was not completed successfully")
    poor = {entry["case_id"]: entry["issue"] for entry in review["grades"]["cases"]
            if entry["case_id"].startswith("web_") and entry["grade"] == "poor"}
    ids = {sample["case_id"]: sample["id"] for sample in samples["web"]}
    return {ids[case_id]: issue for case_id, issue in poor.items()}


def ordered_documents(documents):
    return sorted(documents, key=lambda doc: (seeded_key("shuffle", doc.id), doc.id))


def wiki_metadata(spec_path):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))["wiki"]
    if spec["repo"] != "wikimedia/wikipedia" or spec["revision"] != WIKI_REVISION or spec["config"] != "20231101.zh":
        raise ValueError("Wikipedia source does not match the pinned snapshot")
    return {key: spec[key] for key in ("repo", "revision", "config", "files")} | {
        "licenses": WIKI_LICENSES, "license_urls": WIKI_LICENSE_LINKS, "attribution": WIKI_CARD,
        "selection": "Per-file SHA-256 order of row group indices, interleaved over parquet files in path order.",
        "transformations": "Document cleaning and deduplication. Article title and URL retained on each document.",
        "row_groups": [],
    }


def wiki_group_batches(metadata, progress):
    from huggingface_hub import HfFileSystem
    from pyarrow import parquet

    file_system = HfFileSystem(token=False)
    with ExitStack() as stack:
        files = []
        for entry in sorted(metadata["files"], key=lambda item: item["path"]):
            source = entry["path"]
            path = f'datasets/{metadata["repo"]}@{metadata["revision"]}/{source}'
            progress(f"opening Wikipedia parquet metadata: {source}")
            handle = stack.enter_context(file_system.open(path, "rb", block_size=4 << 20, cache_type="readahead"))
            if handle.size != entry["bytes"]:
                raise ValueError(f"Wrong size for the pinned Wikipedia parquet file: {source}")
            data = parquet.ParquetFile(handle)
            order = sorted(range(data.num_row_groups), key=lambda group: seeded_key("wiki_row_group", f"{source}:{group}"))
            files.append((source, data, iter(order)))
        while files:
            remaining = []
            for source, data, order in files:
                group = next(order, None)
                if group is None:
                    continue
                remaining.append((source, data, order))
                progress(f"reading Wikipedia {source} row group {group}")
                records = data.read_row_group(group, columns=list(WIKI_COLUMNS)).to_pylist()
                for row, record in enumerate(records):
                    yield_record = {"id": document_id(f"wiki:{WIKI_REVISION}", source, group, row),
                                    "source": f"wiki/{source}", "row_group": group, "row_in_group": row,
                                    "article_id": record["id"], "url": record["url"], "title": record["title"],
                                    "licenses": WIKI_LICENSES, "text": record["text"]}
                    records[row] = yield_record
                yield {"path": source, "row_group": group, "documents": len(records)}, records
            files = remaining


def write_splits(output, selected, progress, previous_sample_ids=()):
    stats = {}
    names = []
    for split in SPLITS:
        ordered = ordered_documents(selected[split])
        docs, tokens = 0, 0
        sources = defaultdict(Counter)
        prefix = Counter()
        target_prefix = Counter()
        windows = [Counter() for _ in range(PREFIX_TOKENS // COVERAGE_WINDOW)] if split == "train" else []
        quality_samples = defaultdict(list)
        jsonl_name, bin_name = f"{split}.jsonl", f"{split}.bin"
        names.extend((jsonl_name, bin_name))
        with (output / f"{jsonl_name}.partial").open("w", encoding="utf-8", newline="\n") as jsonl, \
                (output / f"{bin_name}.partial").open("wb") as binary:
            for document in ordered:
                source = document.source
                token_count = len(document.token_bytes) // 2
                jsonl.write(json_line(document.record))
                binary.write(document.token_bytes)
                sources[source].update(documents=1, tokens=token_count)
                if document.id not in previous_sample_ids:
                    key = digest(f"{QUALITY_SAMPLE_SEED}:{split}:{document.id}")
                    sample = quality_samples[source]
                    if len(sample) < 4 or key < sample[-1][0]:
                        sample.append((key, sample_record(document.record)))
                        sample.sort(key=lambda item: item[0])
                        del sample[4:]
                if split == "train" and tokens < PREFIX_TOKENS:
                    stop = min(tokens + token_count, PREFIX_TOKENS)
                    prefix[source] += stop - tokens
                    for window in range(tokens // COVERAGE_WINDOW, (stop - 1) // COVERAGE_WINDOW + 1):
                        start_in_window = max(tokens, window * COVERAGE_WINDOW)
                        stop_in_window = min(stop, (window + 1) * COVERAGE_WINDOW)
                        windows[window][source] += stop_in_window - start_in_window
                if split == "train" and tokens < PREFIX_TOKENS + 1:
                    target_prefix[source] += min(tokens + token_count, PREFIX_TOKENS + 1) - max(tokens, 1)
                docs += 1
                tokens += token_count
                if docs % 8192 == 0:
                    progress(f"packed {split}: {docs:,} documents, {tokens:,} tokens")
        if split == "train" and (
            sum(prefix.values()) != PREFIX_TOKENS or set(sources) != set(prefix)
            or sum(target_prefix.values()) != PREFIX_TOKENS or set(sources) != set(target_prefix)
        ):
            raise RuntimeError(f"The first {PREFIX_TOKENS:,} tokens do not cover every training source")
        for source in sources.values():
            source["token_fraction"] = source["tokens"] / tokens
        stats[split] = {"documents": docs, "tokens": tokens, "binary_bytes": 2 * tokens,
                        "sources": dict(sorted(sources.items())), "quality_samples": {
                            source: [record for _, record in sample] for source, sample in sorted(quality_samples.items())
                        }}
        if split == "train":
            stats[split].update(first_100m_source_tokens=dict(sorted(prefix.items())),
                                first_100m_target_source_tokens=dict(sorted(target_prefix.items())),
                                first_100m_target_source_fractions={source: count / PREFIX_TOKENS
                                                                    for source, count in sorted(target_prefix.items())},
                                first_100m_10m_window_source_tokens=[dict(sorted(window.items())) for window in windows])
        progress(f"packed {split}: {docs:,} documents, {tokens:,} tokens (complete)")
    return stats, names


def verify_legacy_inputs(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["corpus_id"] != "stage1_v1" or not manifest["frozen"] or manifest["vocab_size"] != 16384:
        raise ValueError("Expected the frozen stage1_v1 corpus and 16,384-token vocabulary")
    tokenizer_path = Path(manifest["tokenizer_path"])
    if tokenizer_path.resolve() != (directory / "tokenizer.json").resolve() or not tokenizer_path.is_absolute():
        raise ValueError("Expected the original absolute stage1_v1 tokenizer path")
    if sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256 or manifest["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("Frozen tokenizer checksum changed")
    for filename in ("tokenizer.json", *(f"{split}.jsonl" for split in SPLITS)):
        if file_identity(directory / filename) != manifest["files"][filename]:
            raise ValueError(f"Frozen legacy input identity changed: {filename}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    check_special_ids(tokenizer)
    return manifest, tokenizer


def wait_for_file(path, progress, timeout_seconds=1200):
    deadline = time.monotonic() + timeout_seconds
    if not path.is_file():
        progress(f"waiting for {path.resolve()}")
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Required file did not become available: {path.resolve()}")
        time.sleep(5)


def build_corpus(old_dir, output, report_path, wiki_spec, eval_prompts, review_ready=None, reviewed_exclusions=None,
                 previous_sample_ids=()):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    legacy, tokenizer = verify_legacy_inputs(old_dir)
    build = CorpusBuild(reviewed_exclusions)
    with build_lock(output), heartbeat("initializing cleaned comparison corpus") as progress:
        build.add_legacy(old_dir, progress)
        for path in eval_prompts:
            wait_for_file(path, progress)
        build.add_eval_prompts(eval_prompts, progress)
        selected, discarded, groups, old_tokens = build.select_and_encode(tokenizer, progress)
        wiki = None
        tokens = old_tokens
        counts_report = {"initial_legacy_train_tokens": legacy["splits"]["train"]["tokens"],
                         "clean_legacy_train_tokens": old_tokens, "min_train_tokens": MIN_TRAIN_TOKENS}
        if tokens < MIN_TRAIN_TOKENS:
            write_json(report_path, {**counts_report, "status": "clean_legacy_ready", **build.data_report(discarded, groups)})
            if wiki_spec is None:
                raise RuntimeError(f"The cleaned legacy corpus is short by {MIN_TRAIN_TOKENS - tokens:,} tokens")
            wiki = wiki_metadata(wiki_spec)
            batches = wiki_group_batches(wiki, progress)
            try:
                for entry, records in batches:
                    for record in records:
                        build.add(record)
                    wiki["row_groups"].append(entry)
                    # Encode once per round of source files so the sampled supplement spans the snapshot.
                    if len(wiki["row_groups"]) % len(wiki["files"]) == 0:
                        selected, discarded, groups, tokens = build.select_and_encode(tokenizer, progress)
                        write_json(report_path, {**counts_report, "status": "supplement_in_progress", "wiki": wiki,
                                                 "train_tokens": tokens, **build.data_report(discarded, groups)})
                        if tokens >= MIN_TRAIN_TOKENS:
                            break
            finally:
                batches.close()
        if tokens < MIN_TRAIN_TOKENS or any(not docs for docs in selected.values()):
            raise RuntimeError(f"Expected at least {MIN_TRAIN_TOKENS:,} train tokens and nonempty holdouts, found {tokens:,}")
        stats, names = write_splits(output, selected, progress, previous_sample_ids)
        if stats["train"]["tokens"] != tokens:
            raise RuntimeError("The number of packed tokens differs from the distinct-document count")
        input_sources = {source for source, counts in build.counts.items() if counts["input_documents"]}
        if set(stats["train"]["sources"]) != input_sources:
            raise RuntimeError(f"Input sources with no retained training documents: {input_sources - set(stats['train']['sources'])}")
        deduplication = {
            "exact": "SHA-256 on NFKC-casefolded alphanumeric text",
            "near": "Bottom-16 hashes of character 5-grams, at least two shared keys, then exact Jaccard >= 0.8",
            "shingle_length_ratio_minimum": 0.8,
            "maximum_hash_postings": MAX_POSTINGS,
            "scope": "Cleaned candidates and all old validation/test records in the same index before new splitting. "
                     "Old holdout originals and navigation-stripped views share a connected component; no quality "
                     "threshold can remove an old holdout from the conflict index.",
            "limitation": "Candidates with fewer than two surviving bottom hashes shared are unexamined. The approximate "
                          "candidate search can miss duplicate pairs, including texts sharing frequent suppressed keys.",
            "representative": "For each connected group, least removed boilerplate, then longest text, then minimum document ID",
        }
        shuffle = {"seed": SEED, "key": "SHA-256 UTF-8 of '42:shuffle:<doc_id>'", "order": "ascending digest, then document ID",
                   "packing": "One little-endian uint16 token stream per split, same JSONL ordering, with EOS after each document"}
        report = {**counts_report, "status": "awaiting_quality_review", "wiki": wiki, "train_tokens": tokens,
                  "shuffle": shuffle, "deduplication": deduplication, "splits": stats,
                  "quality_sample_seed": QUALITY_SAMPLE_SEED, "quality_sample_exclusions": len(previous_sample_ids),
                  **build.data_report(discarded, groups)}
        write_json(report_path, report)
        if review_ready is not None:
            wait_for_file(review_ready, progress)
        for name in names:
            (output / f"{name}.partial").replace(output / name)
        manifest = {
            "schema_version": 1, "corpus_id": CORPUS_ID, "frozen": True, "dtype": "<u2",
            "tokenizer_path": legacy["tokenizer_path"], "tokenizer_sha256": legacy["tokenizer_sha256"],
            "vocab_size": legacy["vocab_size"], "special_token_ids": SPECIAL_IDS,
            "splits": {split: {key: value for key, value in stats[split].items() if key != "quality_samples"} for split in SPLITS},
            "files": {"tokenizer.json": legacy["files"]["tokenizer.json"],
                      **{name: file_identity(output / name) for name in names}},
            "sources": {"legacy": {key: legacy["sources"][key] for key in ("repo", "revision")}, "wiki": wiki},
            "seed": SEED, "shuffle": shuffle, "deduplication": deduplication,
            "split_percentages": [98, 1, 1], "filter_settings": QUALITY_RULES,
            "evaluation_screen": build.evaluation_paths, "quality_report_path": str(report_path.resolve()),
        }
        write_json(output / "manifest.json", manifest)
        write_json(report_path, {**report, "status": "frozen"})
        progress(f"frozen comparison corpus: {output / 'manifest.json'}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=Path("data/stage1_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/pretrain_clean_v2"))
    parser.add_argument("--report", type=Path, default=Path("reports/comparison_v2/predata_build.json"))
    parser.add_argument("--wiki-spec", type=Path)
    parser.add_argument("--eval-prompts", type=Path, action="append", default=[])
    parser.add_argument("--review-ready", type=Path)
    parser.add_argument("--legacy-quality-review", type=Path)
    parser.add_argument("--legacy-quality-samples", type=Path)
    parser.add_argument("--previous-quality-report", type=Path)
    args = parser.parse_args()
    exclusions = legacy_quality_exclusions(args.legacy_quality_review, args.legacy_quality_samples)
    previous_sample_ids = set()
    if args.previous_quality_report is not None:
        previous = json.loads(args.previous_quality_report.read_text(encoding="utf-8"))
        previous_sample_ids = {record["id"] for split in previous["splits"].values()
                               for samples in split["quality_samples"].values() for record in samples}
    build_corpus(args.old_dir, args.output, args.report, args.wiki_spec, args.eval_prompts, args.review_ready,
                 exclusions, previous_sample_ids)


if __name__ == "__main__":
    main()
