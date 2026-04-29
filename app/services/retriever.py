from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from app.services.knowledge_base import ChunkRecord, DenseIndexRecord, RetrievalCorpusRecord


TOKEN_RE = re.compile(r"[A-Za-z0-9_+-]+|[\u4e00-\u9fff]+")
STEP_SIGNAL_RE = re.compile(r"(?:^|\n)\s*\d+[.)、]")
GENERIC_SECTION_RE = re.compile(r"(指南|目录|前言|内容|目标|重要信息)$")
NUMERIC_ANSWER_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:"
    r"kg|kgs?|kilogram(?:s)?|lb|lbs?|"
    r"千克|公斤|克|磅|"
    r"w|kw|瓦|千瓦|"
    r"v|kv|伏|伏特|"
    r"a|ma|安|毫安|"
    r"l|ml|升|毫升|"
    r"℃|°c|°f|摄氏度|华氏度|"
    r"小时|分钟|秒|h|min|rpm|%"
    r"|cm|mm|m|米|英寸|寸|公里/小时|km/h|mph"
    r")",
    re.IGNORECASE,
)
PARAMETER_HINTS = (
    "max",
    "maximum",
    "minimum",
    "min",
    "load",
    "weight",
    "capacity",
    "size",
    "spec",
    "specification",
    "power",
    "voltage",
    "current",
    "pressure",
    "temperature",
    "speed",
    "rpm",
    "尺寸",
    "参数",
    "载重",
    "重量",
    "容量",
    "功率",
    "电压",
    "电流",
    "压力",
    "温度",
    "速度",
    "转速",
    "最大",
    "最小",
)
ACTION_TERMS = {
    "battery",
    "charge",
    "delete",
    "images",
    "playback",
    "memory card",
    "insert",
    "shutter",
    "shutter button",
    "button",
    "settings",
    "replace",
    "fuse",
    "jet wash",
    "clean",
    "storage",
    "storage compartment",
    "maintenance setting",
    "factory reset",
    "start",
    "start engine",
    "engine start",
    "start switch",
    "lanyard",
    "connect",
    "base",
    "station",
    "fax",
    "line",
    "jack",
    "engine",
    "oil",
    "spark",
    "plug",
    "inspect",
    "quick",
    "release",
    "natural",
    "float",
    "valve",
    "steam",
    "vent",
    "venting",
    "充电",
    "删除",
    "图像",
    "回放",
    "存储卡",
    "装入",
    "插入",
    "快门",
    "按钮",
    "设置",
    "更换",
    "保险丝",
    "喷射清洗",
    "清洗",
    "储物",
    "维护设置",
    "出厂重置",
    "启动",
    "启动发动机",
    "发动机启动",
    "启动开关",
    "熄火绳",
}
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "properly",
    "should",
    "the",
    "to",
    "what",
    "when",
    "with",
    "you",
    "your",
}
GENERIC_SEARCH_KEYWORDS = {
    "air fryer",
    "base station",
    "boat",
    "camera",
    "ereader",
    "fax",
    "handset",
    "jet boat",
    "landline",
    "lawn mower",
    "microwave",
    "motherboard",
    "mower",
    "pressure cooker",
    "snowmobile",
    "telephone",
}
DENSE_VECTOR_DIM = 256
DENSE_CHAR_NGRAM = 3


def tokenize(text: str) -> list[str]:
    raw_tokens = TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.append(token)
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
            tokens.extend(token[index : index + 3] for index in range(len(token) - 2))
            tokens.extend(list(token))
        else:
            tokens.append(token)
    return tokens


def tokenize_query(text: str) -> list[str]:
    tokens: list[str] = []
    for token in tokenize(text):
        if token in QUERY_STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


def build_char_ngrams(text: str, n: int = 2) -> Counter[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return Counter()
    if len(compact) <= n:
        return Counter([compact])
    return Counter(compact[index : index + n] for index in range(len(compact) - n + 1))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    if not dot:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def dense_feature_weights(text: str) -> dict[str, float]:
    features: dict[str, float] = {}
    for token in tokenize(text):
        if not token:
            continue
        features[f"tok:{token}"] = features.get(f"tok:{token}", 0.0) + 1.0

    compact = re.sub(r"\s+", "", text.lower())
    if compact:
        n = DENSE_CHAR_NGRAM
        if len(compact) <= n:
            features[f"ng:{compact}"] = features.get(f"ng:{compact}", 0.0) + 0.35
        else:
            for index in range(len(compact) - n + 1):
                gram = compact[index : index + n]
                features[f"ng:{gram}"] = features.get(f"ng:{gram}", 0.0) + 0.35
    return features


def _stable_feature_hash(feature: str, salt: str) -> int:
    payload = f"{salt}:{feature}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


@dataclass(slots=True)
class SearchResult:
    chunk: ChunkRecord
    score: float
    bm25_score: float
    semantic_score: float
    matched_terms: list[str]


@dataclass(slots=True)
class QueryProfile:
    howto: bool
    list_style: bool
    range_style: bool
    warranty: bool
    component: bool
    parameter: bool


class DenseHashRetriever:
    def __init__(self, chunks: list[ChunkRecord], dims: int = DENSE_VECTOR_DIM) -> None:
        self.chunks = chunks
        self.dims = dims
        self.doc_feature_freqs: list[dict[str, float]] = []
        self.feature_df: Counter[str] = Counter()
        self.doc_vectors: list[list[float]] = []

        for chunk in chunks:
            basis = f"{chunk.manual_name} {chunk.product_name} {chunk.section_title} {chunk.text} {' '.join(chunk.keywords)}"
            features = dense_feature_weights(basis)
            self.doc_feature_freqs.append(features)
            self.feature_df.update(features.keys())

        for features in self.doc_feature_freqs:
            self.doc_vectors.append(self._vectorize(features))

    def search(
        self,
        query: str,
        top_k: int = 5,
        boost_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        return self.score_query(query, boost_manuals=boost_manuals)[:top_k]

    def score_query(
        self,
        query: str,
        boost_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        query_features = dense_feature_weights(query)
        if not query_features:
            return []

        query_vector = self._vectorize(query_features)
        boost_manuals = boost_manuals or []
        results: list[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            semantic_score = self._dense_cosine(query_vector, self.doc_vectors[index])
            score = semantic_score
            if boost_manuals:
                score += 0.08 if chunk.manual_name in boost_manuals else -0.02
            matched_terms = sorted(
                {
                    token
                    for token in tokenize_query(query)
                    if token in self.doc_feature_freqs[index]
                    or f"tok:{token}" in self.doc_feature_freqs[index]
                }
            )
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    bm25_score=0.0,
                    semantic_score=semantic_score,
                    matched_terms=matched_terms,
                )
            )

        results.sort(key=lambda item: (-item.score, -item.semantic_score, item.chunk.order))
        return results

    def _vectorize(self, features: dict[str, float]) -> list[float]:
        vector = [0.0] * self.dims
        doc_count = len(self.chunks) or 1
        for feature, frequency in features.items():
            if frequency <= 0:
                continue
            df = self.feature_df.get(feature, 0)
            idf = math.log(1.0 + (doc_count + 1.0) / (df + 1.0))
            weight = frequency * idf
            index = _stable_feature_hash(feature, "dense-index") % self.dims
            sign = 1.0 if _stable_feature_hash(feature, "dense-sign") & 1 else -1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]

    def _dense_cosine(self, left: list[float], right: list[float]) -> float:
        return sum(left[index] * right[index] for index in range(self.dims))


class HybridRetriever:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.doc_term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.doc_ngram_freqs: list[Counter[str]] = []
        self.term_df: Counter[str] = Counter()
        self.avg_doc_length = 0.0

        for chunk in chunks:
            basis = f"{chunk.manual_name} {chunk.product_name} {chunk.section_title} {chunk.text} {' '.join(chunk.keywords)}"
            term_freq = Counter(tokenize(basis))
            self.doc_term_freqs.append(term_freq)
            self.doc_lengths.append(sum(term_freq.values()))
            self.doc_ngram_freqs.append(build_char_ngrams(basis))
            self.term_df.update(term_freq.keys())

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

    def search(
        self,
        query: str,
        top_k: int = 5,
        boost_manuals: list[str] | None = None,
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        return self.score_query(query, boost_manuals=boost_manuals, allowed_manuals=allowed_manuals)[:top_k]

    def score_query(
        self,
        query: str,
        boost_manuals: list[str] | None = None,
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        query_tokens = tokenize_query(query)
        query_ngrams = build_char_ngrams(query)
        if not query_tokens:
            return []

        boost_manuals = boost_manuals or []
        allowed = set(allowed_manuals or [])
        profile = self._build_query_profile(query)
        alpha_terms = re.findall(r"[A-Za-z0-9_+-]{2,}", query.lower())
        results: list[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            if allowed and chunk.manual_name not in allowed:
                continue
            bm25_score = self._bm25(query_tokens, index)
            semantic_score = cosine_similarity(query_ngrams, self.doc_ngram_freqs[index])
            manual_bonus = 0.0
            if boost_manuals:
                manual_bonus = 0.25 if chunk.manual_name in boost_manuals else -0.05
            section_bonus = self._section_bonus(query_tokens, chunk)
            chunk_type_bonus = self._chunk_type_bonus(profile, chunk.chunk_type)
            special_bonus = self._special_term_bonus(alpha_terms, chunk)
            intent_bonus = self._intent_bonus(query_tokens, chunk)
            score = 0.65 * (bm25_score / (bm25_score + 5.0) if bm25_score > 0 else 0.0)
            score += 0.35 * semantic_score
            score += manual_bonus
            score += section_bonus
            score += chunk_type_bonus
            score += special_bonus
            score += intent_bonus
            matched_terms = sorted({token for token in query_tokens if token in self.doc_term_freqs[index]})
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                    matched_terms=matched_terms,
                )
            )

        results.sort(key=lambda item: (-item.score, -item.bm25_score, item.chunk.order))
        return results

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        boost_manuals: list[str] | None = None,
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        query_tokens = tokenize_query(query)
        if not query_tokens:
            return []

        boost_manuals = boost_manuals or []
        allowed = set(allowed_manuals or [])
        results: list[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            if allowed and chunk.manual_name not in allowed:
                continue
            bm25_score = self._bm25(query_tokens, index)
            manual_bonus = 0.25 if boost_manuals and chunk.manual_name in boost_manuals else 0.0
            matched_terms = sorted({token for token in query_tokens if token in self.doc_term_freqs[index]})
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=bm25_score + manual_bonus,
                    bm25_score=bm25_score,
                    semantic_score=0.0,
                    matched_terms=matched_terms,
                )
            )

        results.sort(key=lambda item: (-item.score, -item.bm25_score, item.chunk.order))
        return results[:top_k]

    def rerank_candidates(
        self,
        candidates: list[SearchResult],
        query_variants: dict[str, str],
        boost_manuals: list[str] | None = None,
        search_keywords: list[str] | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        boost_manuals = boost_manuals or []
        search_keywords = search_keywords or []
        query_text = " ".join(query_variants.values())
        profile = self._build_query_profile(query_text)
        query_tokens = set(tokenize_query(query_text))
        action_terms = {token for token in query_tokens if token in ACTION_TERMS}
        normalized_keywords = [
            keyword.strip()
            for keyword in search_keywords
            if keyword and len(keyword.strip()) >= 2
        ]
        normalized_keywords_lower = [keyword.lower() for keyword in normalized_keywords]
        critical_keywords = [
            keyword
            for keyword in normalized_keywords_lower
            if keyword not in GENERIC_SEARCH_KEYWORDS
            and not re.fullmatch(r"(?:connect|install|remove|replace|change|set|reset|inspect|use|open|close)", keyword)
        ]

        reranked: list[SearchResult] = []
        for rank, candidate in enumerate(candidates[: max(top_k * 5, 15)], start=1):
            chunk = candidate.chunk
            combined = f"{chunk.section_title}\n{chunk.text}"
            combined_lower = combined.lower()
            title_lower = chunk.section_title.lower()
            score = candidate.score

            if boost_manuals and chunk.manual_name in boost_manuals:
                score += 0.10

            if profile.parameter and NUMERIC_ANSWER_RE.search(combined):
                score += 0.24

            keyword_hits = 0
            title_keyword_hits = 0
            for keyword in normalized_keywords_lower:
                if keyword in title_lower:
                    title_keyword_hits += 1
                    keyword_hits += 1
                elif keyword in combined_lower:
                    keyword_hits += 1
            score += min(0.72, 0.36 * title_keyword_hits)
            score += min(0.36, 0.12 * keyword_hits)
            if critical_keywords:
                critical_hits = sum(1 for keyword in critical_keywords if keyword in combined_lower)
                title_critical_hits = sum(1 for keyword in critical_keywords if keyword in title_lower)
                if critical_hits:
                    score += min(1.4, 0.42 * critical_hits)
                    score += min(0.6, 0.30 * title_critical_hits)
                else:
                    score -= 2.2

            if action_terms:
                action_hits = sum(1 for term in action_terms if term in combined_lower)
                score += min(0.30, 0.10 * action_hits)
                if action_hits == 0 and chunk.chunk_type in {"general", "title_only", "toc"}:
                    score -= 0.10
                if re.search(r"important safeguards|safety instructions|warning|caution", title_lower) and title_keyword_hits == 0:
                    score -= 0.42

            if profile.howto:
                if chunk.chunk_type == "step" or STEP_SIGNAL_RE.search(combined):
                    score += 0.14
                elif chunk.chunk_type in {"toc", "title_only"} and not NUMERIC_ANSWER_RE.search(combined):
                    score -= 0.08

            compact_title = re.sub(r"\s+", "", chunk.section_title.strip())
            if GENERIC_SECTION_RE.search(compact_title) and keyword_hits == 0:
                score -= 0.12

            score += max(0.0, 0.04 - 0.005 * (rank - 1))

            reranked.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    bm25_score=candidate.bm25_score,
                    semantic_score=candidate.semantic_score,
                    matched_terms=candidate.matched_terms,
                )
            )

        reranked.sort(key=lambda item: (-item.score, -item.bm25_score, item.chunk.order))
        return reranked[:top_k]

    def _build_query_profile(self, query: str) -> QueryProfile:
        lowered_query = query.lower()
        howto = any(token in query for token in ("如何", "怎么", "怎样", "步骤", "安装", "拆卸", "设置", "清洁", "启动", "关闭", "调节", "更换", "操作")) or any(
            token in lowered_query
            for token in ("how", "steps", "install", "remove", "set", "clean", "start", "replace", "change", "connect", "inspect")
        )
        list_style = any(token in query for token in ("哪些", "包含", "包括", "有什么", "分别", "前五条", "前5条", "最后三个步骤", "前两个步骤"))
        range_style = any(token in query for token in ("前五条", "前5条", "最后三个步骤", "前两个步骤", "前六个步骤", "最后三步"))
        warranty = any(token in query for token in ("保修", "质保", "免费服务", "除外责任", "免责"))
        component = any(token in query for token in ("部件", "组成", "接口", "按键", "按钮", "结构", "开关"))
        parameter = any(token in query.lower() for token in PARAMETER_HINTS)
        return QueryProfile(
            howto=howto,
            list_style=list_style,
            range_style=range_style,
            warranty=warranty,
            component=component,
            parameter=parameter,
        )

    def _section_bonus(self, query_tokens: list[str], chunk: ChunkRecord) -> float:
        title_tokens = set(tokenize(chunk.section_title))
        if not title_tokens:
            return 0.0
        overlap = len(set(query_tokens) & title_tokens)
        return min(0.24, overlap * 0.06)

    def _chunk_type_bonus(self, profile: QueryProfile, chunk_type: str) -> float:
        bonus = 0.0
        if chunk_type == "toc":
            bonus -= 0.30
        elif chunk_type == "title_only":
            bonus -= 0.14

        if profile.howto:
            if chunk_type == "step":
                bonus += 0.24
            elif chunk_type == "menu":
                bonus += 0.18
            elif chunk_type in {"note", "warning"}:
                bonus -= 0.10
            elif chunk_type in {"toc", "title_only"}:
                bonus -= 0.12

        if profile.range_style:
            if chunk_type in {"list", "step"}:
                bonus += 0.18
            elif chunk_type == "troubleshoot":
                bonus -= 0.18
            elif chunk_type in {"title_only", "toc"}:
                bonus -= 0.12

        if profile.list_style and chunk_type == "list":
            bonus += 0.10
        if profile.list_style and chunk_type == "menu":
            bonus += 0.08

        if profile.warranty:
            if chunk_type == "warranty":
                bonus += 0.22
            elif chunk_type in {"note", "warning", "toc"}:
                bonus -= 0.08

        if profile.component:
            if chunk_type == "component":
                bonus += 0.16
            elif chunk_type == "menu":
                bonus += 0.12
            elif chunk_type == "general":
                bonus += 0.06
            elif chunk_type in {"note", "warning", "warranty", "troubleshoot"}:
                bonus -= 0.10
            elif chunk_type == "step":
                bonus -= 0.04

        if profile.parameter:
            if chunk_type == "title_only":
                bonus += 0.28
            elif chunk_type == "component":
                bonus += 0.10
            elif chunk_type == "menu":
                bonus += 0.06
            elif chunk_type == "general":
                bonus -= 0.04
            elif chunk_type in {"step", "troubleshoot"}:
                bonus -= 0.12
            elif chunk_type in {"toc", "note", "warning"}:
                bonus -= 0.10

        return bonus

    def _special_term_bonus(self, alpha_terms: list[str], chunk: ChunkRecord) -> float:
        if not alpha_terms:
            return 0.0
        combined = f"{chunk.section_title} {chunk.text}".lower()
        bonus = 0.0
        for term in alpha_terms:
            if term in combined:
                bonus += 0.22
        return min(0.44, bonus)

    def _intent_bonus(self, query_tokens: list[str], chunk: ChunkRecord) -> float:
        query_set = set(query_tokens)
        combined = f"{chunk.section_title} {chunk.text}"
        combined_lower = combined.lower()
        bonus = 0.0

        def has_any(*terms: str) -> bool:
            return any(term in combined or term.lower() in combined_lower for term in terms)

        if {"电池", "充电"} <= query_set:
            if has_any("充电", "usb线", "usb", "交流电源适配器", "装入/充电电池"):
                bonus += 0.34
                if has_any("电池充电", "充电时间", "使用随附的usb线充电", "装入/充电电池"):
                    bonus += 0.12
            elif has_any("电池"):
                bonus += 0.06
            if has_any("随附配件", "认证标志", "回收", "生活垃圾", "battery咨询", "1-800-8-battery"):
                bonus -= 0.26
        if "可充电电池" in query_set and "可充电电池" in combined:
            bonus += 0.14
        if "存储卡" in query_set and has_any("插入存储卡", "装入存储卡"):
            bonus += 0.34
        elif "存储卡" in query_set and has_any("存储卡"):
            bonus += 0.08
        if {"删除", "图像"} <= query_set and has_any("删除", "图像"):
            bonus += 0.30
            if has_any("全部", "单张", "全部删除图像", "单张或全部删除图像"):
                bonus += 0.22
        if {"delete", "images"} <= query_set and has_any("delete", "images"):
            bonus += 0.30
            if has_any("all", "single", "delete all images", "delete images"):
                bonus += 0.18
        if {"快门", "按钮"} <= query_set and has_any("快门按钮", "快门"):
            bonus += 0.22
        if {"电视", "图像"} <= query_set and has_any("电视", "图像", "回放"):
            bonus += 0.22
        if "保险丝" in query_set and "保险丝" in combined:
            bonus += 0.24
        if "喷射清洗" in query_set and "喷射清洗" in combined:
            bonus += 0.26
        if "心率" in query_set and ("目标" in query_set or "年龄" in query_set):
            if has_any("心率控制", "目标心率程序", "心率区间", "最大心率百分比", "50–60%", "50-60%"):
                bonus += 0.46
            if has_any("目标追踪统计", "运动统计数据"):
                bonus -= 0.22
        if {"启动", "发动机"} <= query_set:
            if has_any("启动发动机", "发动机启动", "启动开关", "熄火绳", "发动机熄火开关"):
                bonus += 0.34
            elif has_any("启动"):
                bonus += 0.08
            if has_any("重要信息", "前言", "如何使用本练习指南", "内容 前言"):
                bonus -= 0.20
        if "储物舱" in query_set and has_any("储物", "储物舱"):
            bonus += 0.22
        if "维护设置" in query_set and has_any("维护设置", "设置画面"):
            bonus += 0.24
        if "出厂重置" in query_set and has_any("出厂重置", "重置"):
            bonus += 0.24
        if {"quick", "release"} <= query_set and has_any("quick release", "quick release button", "vent position"):
            bonus += 0.38
        if {"natural", "release"} <= query_set and has_any("natural release", "depressurizes naturally"):
            bonus += 0.38
        if {"float", "valve"} <= query_set and has_any("float valve"):
            bonus += 0.36
        if {"engine", "oil"} <= query_set and has_any("engine oil", "oil drain", "oil fill", "change the engine oil"):
            bonus += 0.34
        if {"spark", "plug"} <= query_set and has_any("spark plug"):
            bonus += 0.34
        if {"base", "station"} <= query_set and has_any("connect the base station", "base station"):
            bonus += 0.30
        if "fax" in query_set and has_any("fax", "telephone line cord", "line jack", "ext jack"):
            bonus += 0.28

        return min(0.52, bonus)

    def _bm25(self, query_tokens: list[str], doc_index: int, k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        doc_freq = self.doc_term_freqs[doc_index]
        doc_length = self.doc_lengths[doc_index] or 1
        seen_tokens: set[str] = set()
        for token in query_tokens:
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            frequency = doc_freq.get(token, 0)
            if not frequency:
                continue
            doc_frequency = self.term_df.get(token, 0)
            numerator = len(self.chunks) - doc_frequency + 0.5
            denominator = doc_frequency + 0.5
            idf = math.log(1 + (numerator / denominator))
            score += idf * (
                frequency * (k1 + 1)
            ) / (frequency + k1 * (1 - b + b * doc_length / (self.avg_doc_length or 1.0)))
        return score


class DenseEmbeddingRetriever:
    def __init__(
        self,
        dense_index: list[DenseIndexRecord],
        chunk_lookup: dict[str, ChunkRecord],
    ) -> None:
        self._entries: list[tuple[ChunkRecord, list[float], set[str]]] = []
        for record in dense_index:
            chunk = chunk_lookup.get(record.chunk_id)
            if chunk is None:
                continue
            vector = self._normalize(record.vector)
            if not vector:
                continue
            tokens = set(tokenize(f"{chunk.manual_name} {chunk.product_name} {chunk.section_title} {chunk.text}"))
            self._entries.append((chunk, vector, tokens))

    def is_ready(self) -> bool:
        return bool(self._entries)

    def search_query_vector(
        self,
        query_vector: list[float],
        top_k: int = 5,
        boost_manuals: list[str] | None = None,
        query_text: str = '',
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        normalized_query = self._normalize(query_vector)
        if not normalized_query:
            return []

        boost_manuals = boost_manuals or []
        allowed = set(allowed_manuals or [])
        query_tokens = set(tokenize_query(query_text)) if query_text else set()
        results: list[SearchResult] = []
        for chunk, vector, token_set in self._entries:
            if allowed and chunk.manual_name not in allowed:
                continue
            semantic_score = self._dot(normalized_query, vector)
            score = semantic_score
            if boost_manuals:
                score += 0.10 if chunk.manual_name in boost_manuals else -0.02
            matched_terms = sorted(query_tokens & token_set)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    bm25_score=0.0,
                    semantic_score=semantic_score,
                    matched_terms=matched_terms,
                )
            )

        results.sort(key=lambda item: (-item.score, -item.semantic_score, item.chunk.order))
        return results[:top_k]

    def _normalize(self, vector: list[float]) -> list[float]:
        if not vector:
            return []
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if not norm:
            return []
        return [value / norm for value in values]

    def _dot(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            size = min(len(left), len(right))
            return sum(left[index] * right[index] for index in range(size))
        return sum(left[index] * right[index] for index in range(len(left)))


class SidecarHybridRetriever:
    def __init__(
        self,
        records: list[RetrievalCorpusRecord],
        chunk_lookup: dict[str, ChunkRecord],
    ) -> None:
        self.chunk_lookup = chunk_lookup
        synthetic_chunks: list[ChunkRecord] = []
        for order, record in enumerate(records, start=1):
            synthetic_chunks.append(
                ChunkRecord(
                    chunk_id=record.chunk_id,
                    manual_name=record.manual_name,
                    product_name=record.product_name,
                    section_title=record.title_en or record.title_zh,
                    page=None,
                    text="\n".join(
                        part
                        for part in (
                            record.retrieval_text_mix,
                            record.retrieval_text_en,
                            record.text_en_summary,
                        )
                        if part
                    ),
                    image_ids=[],
                    keywords=_dedupe_keep_order(record.keywords_en + record.product_aliases_en),
                    source_file=f"{record.manual_name}:retrieval_corpus",
                    order=order,
                    chunk_type=record.chunk_type,
                )
            )
        self._base = HybridRetriever(synthetic_chunks)
        self._dense = None

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        boost_manuals: list[str] | None = None,
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        return self._map_results(
            self._base.search_bm25(
                query,
                top_k=top_k,
                boost_manuals=boost_manuals,
                allowed_manuals=allowed_manuals,
            )
        )

    def score_query(
        self,
        query: str,
        boost_manuals: list[str] | None = None,
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        return self._map_results(
            self._base.score_query(
                query,
                boost_manuals=boost_manuals,
                allowed_manuals=allowed_manuals,
            )
        )

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        boost_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        if self._dense is None:
            return []
        return self._map_results(
            self._dense.search(
                query,
                top_k=top_k,
                boost_manuals=boost_manuals,
            )
        )

    def score_query_dense(
        self,
        query: str,
        boost_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        if self._dense is None:
            return []
        return self._map_results(
            self._dense.score_query(
                query,
                boost_manuals=boost_manuals,
            )
        )

    def _map_results(self, results: list[SearchResult]) -> list[SearchResult]:
        mapped: list[SearchResult] = []
        for result in results:
            original_chunk = self.chunk_lookup.get(result.chunk.chunk_id)
            if original_chunk is None:
                continue
            mapped.append(
                SearchResult(
                    chunk=original_chunk,
                    score=result.score,
                    bm25_score=result.bm25_score,
                    semantic_score=result.semantic_score,
                    matched_terms=result.matched_terms,
                )
            )
        return mapped


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(normalized)
    return ordered
