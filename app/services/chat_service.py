from __future__ import annotations

import re
import os
import time
from dataclasses import dataclass
from uuid import uuid4

from app.core.config import Settings
from app.services.generator import EvidenceGroundedGenerator, INSUFFICIENT_PHRASE
from app.services.guardrail import GuardrailService
from app.services.embedding_client import EmbeddingClient
from app.services.knowledge_base import KnowledgeBaseRepository
from app.services.llm_client import LLMClient
from app.services.llm_query_planner import LLMQueryPlan, LLMQueryPlanner
from app.services.reranker_client import RerankerClient
from app.services.multimodal_understanding import MultimodalUnderstandingService
from app.services.preprocess import (
    is_manual_access_question,
    is_general_support_question,
    looks_english_dominant,
    normalize_question,
    rewrite_with_context,
    split_sub_questions,
)
from app.services.query_rewriter import QueryRewriteService
from app.services.retriever import DenseEmbeddingRetriever, HybridRetriever, SearchResult, SidecarHybridRetriever, tokenize
from app.services.session_memory import SessionMemory

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
GENERIC_TITLE_RE = re.compile(r"(指南|目录|前言|内容|目标|练习\d+|练习\d+[:：].*|重要信息)$")
GENERIC_GAP_SECTION_RE = re.compile(
    r"(目标|重要信息|目录|前言|内容|概述|提示|警告|注意|指南|"
    r"练习\d+|训练\d+|练习\d+[:：].*|训练\d+[:：].*)$"
)
PARAMETER_TERMS = {
    "max",
    "maximum",
    "minimum",
    "min",
    "load",
    "weight",
    "capacity",
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
}
ACTION_HINT_TERMS = {
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
    "base station",
    "fax",
    "engine oil",
    "spark plug",
    "quick release",
    "natural release",
    "float valve",
    "steam release",
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
    "载重",
    "重量",
}
CRITICAL_ACTION_PHRASES = (
    "jet wash",
    "factory reset",
    "maintenance setting",
    "fuse",
    "start engine",
    "engine start",
    "memory card",
    "shutter button",
    "delete images",
    "battery charge",
    "storage compartment",
    "eyepiece cover",
    "quick release",
    "natural release",
    "float valve",
    "engine oil",
    "spark plug",
    "connect the base station",
    "喷射清洗",
    "出厂重置",
    "维护设置",
    "保险丝",
    "启动发动机",
    "发动机启动",
    "存储卡",
    "快门按钮",
    "删除图像",
    "电池充电",
)
TEXT_PRODUCT_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("jetski", "jet ski", "watercraft", "pwc", "boat", "ship"), "摩托艇"),
    (("camera", "dslr", "camcorder"), "相机"),
    (("air conditioner", "ac", "remote controller", "remote"), "空调"),
    (("refrigerator", "fridge"), "冰箱"),
    (("dishwasher",), "洗碗机"),
    (("generator",), "发电机"),
    (("water pump", "pump"), "水泵"),
    (("thermostat",), "可编程温控器"),
    (("hair dryer", "blow dryer", "dryer"), "吹风机"),
    (("steam cleaner",), "蒸汽清洁机"),
    (("drill",), "电钻"),
    (("air purifier", "purifier"), "空气净化器"),
    (("fitness tracker", "smartwatch", "watch", "tracker"), "健身追踪器"),
    (("exercise bike", "spin bike"), "健身单车"),
    (("ergonomic chair", "office chair"), "人体工学椅"),
    (("vr headset", "virtual reality headset"), "VR头显"),
    (("keyboard",), "功能键盘"),
    (("mouse",), "蓝牙激光鼠标"),
    (("motherboard", "mainboard", "bios", "uefi"), "motherboard"),
    (("coffee machine", "coffee maker"), "coffee machine"),
    (("multi-use pressure cooker", "pressure cooker", "quick release", "natural release", "float valve"), "multi-use pressure cooker"),
    (("air fryer", "airfryer"), "air fryer"),
    (("fax",), "fax"),
    (("telephone", "base station", "handset", "caller id"), "电话"),
    (("earphones", "headphones", "earbuds"), "耳机"),
    (("ereader", "e-reader"), "电子阅读器"),
    (("lawn mower", "mower"), "lawn mower"),
    (("snowmobile",), "snowmobile"),
    (("microwave",), "microwave"),
)
PRODUCT_MANUAL_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("jet boat", "boat", "ship", "sailing", "bimini", "swim platform", "bilge", "livewell", "fire extinguisher", "fire extinguishers"), ("英文喷射船手册", "摩托艇手册")),
    (("jetski", "jet ski", "watercraft", "pwc"), ("摩托艇手册", "英文水上摩托手册")),
    (("camera", "memory card", "shutter button", "af mode", "af point", "autofocus"), ("相机手册", "英文单反相机手册")),
    (("toothbrush",), ("英文电动牙刷手册",)),
    (("pressure cooker", "multi-use pressure cooker", "air fryer", "quick release", "natural release", "float valve", "sealing ring"), ("英文多功能压力锅空气炸锅手册", "英文空气炸锅手册")),
    (("fax",), ("英文传真机手册",)),
    (("landline", "telephone", "base station", "handset"), ("英文固定电话手册",)),
    (("ereader", "e-reader"), ("英文电子阅读器手册",)),
    (("lawn mower", "mower"), ("英文割草机手册",)),
    (("snowmobile",), ("英文雪地摩托手册",)),
    (("microwave",), ("英文微波炉手册",)),
    (("motherboard", "mainboard", "bios", "pci express", "tpm connector"), ("英文主板手册",)),
    (("vacuum", "home base", "full bin", "extractor"), ("英文吸尘器手册",)),
    (("grill", "indirect cooking"), ("英文烧烤炉手册",)),
    (("outdoor antenna", "antenna jack", "300 ohm", "75 ohm"), ("英文电视机手册",)),
    (("coffee machine", "coffee maker"), ("英文咖啡机手册",)),
    (("earphones", "headphones", "earbuds"), ("英文耳机手册",)),
    (("air conditioner", "ac", "空调", "auto restart"), ("空调手册",)),
    (("blower", "leaf blower", "吹风机", "protective equipment", "ppe"), ("吹风机手册",)),
    (("air purifier", "purifier", "空气净化器"), ("空气净化器手册",)),
    (("ergonomic chair", "office chair", "chair", "人体工学椅", "椅子"), ("人体工学椅手册",)),
    (("dishwasher", "洗碗机", "spray arm"), ("洗碗机手册",)),
    (("steam cleaner", "蒸汽清洁机"), ("蒸汽清洁机手册",)),
)
SIDECAR_SUPPORTED_MANUALS = {"相机手册", "摩托艇手册"}
SIDECAR_TRIGGER_TERMS = {
    "battery",
    "charge",
    "memory card",
    "delete",
    "image",
    "max load",
    "load",
    "weight",
    "start",
    "start engine",
    "engine start",
    "start switch",
    "fuse",
    "jet wash",
    "maintenance setting",
    "factory reset",
    "shutter",
    "tv",
    "电池",
    "充电",
    "存储卡",
    "删除",
    "图像",
    "最大载重",
    "载重",
    "重量",
    "启动",
    "启动发动机",
    "发动机启动",
    "启动开关",
    "保险丝",
    "喷射清洗",
    "维护设置",
    "出厂重置",
    "快门按钮",
}
INSUFFICIENT_INDICATORS = (
    INSUFFICIENT_PHRASE,
    INSUFFICIENT_PHRASE.rstrip("。"),
    "当前检索到的说明书证据还不足以支持一个明确结论",
    "说明书证据还不足以支持一个明确结论",
    "说明书证据不足以支持明确结论",
)
QUERY_PHRASE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("battery conversion", "conversion feature", "电池转换", "转换功能"), "电池转换"),
    (("battery switches", "battery switch", "emerg parallel", "start switch", "house switch"), "电池开关"),
    (("over temperature", "temperature warning", "overheat", "overheating"), "过热警告"),
    (("swim platform", "rear platform hatch", "wet storage compartment"), "游泳平台"),
    (("charging with the travel case", "travel case", "charging case"), "旅行盒充电"),
    (("factory reset screen", "factory reset"), "出厂重置"),
    (("indirect cooking", "indirect heat"), "间接烹饪"),
    (("manual program", "memory/erase", "memorizing channels"), "手动记忆频道"),
    (("outdoor antenna", "antenna jack", "300 ohm", "75 ohm"), "室外天线"),
    (("anti-block shield", "anti block shield", "anti-block"), "防堵罩"),
    (("p mode", "p model", "program ae", "program mode"), "P模式"),
    (("unable to pump", "cannot pump", "does not pump", "无法抽水"), "无法抽水"),
    (("robot anatomy", "vacuum anatomy", "robot parts"), "扫地机器人部件"),
    (
        ("storage compartment", "storage compartments", "wet items", "储物舱", "储物格", "储物箱", "湿物"),
        "储物舱",
    ),
    (
        ("remove shutter button", "remove the camera shutter button", "拆下快门按钮", "移除快门按钮"),
        "拆下快门按钮",
    ),
    (("eyepiece cover", "目镜盖", "接目镜盖"), "目镜盖"),
)
PHRASE_ALIASES: dict[str, tuple[str, ...]] = {
    "电池转换": (
        "电池转换",
        "转换功能",
        "battery conversion",
        "conversion feature",
        "battery switches",
        "battery switch assembly",
        "start switch",
        "house switch",
        "emerg parallel",
    ),
    "电池开关": ("battery switches", "battery switch assembly", "start switch", "house switch", "emerg parallel"),
    "过热警告": ("over temperature warning", "over temperature", "temperature warning", "overheating", "cooling water pilot outlet"),
    "游泳平台": ("swim platform", "wet storage compartment", "rear platform hatch", "lock handle"),
    "旅行盒充电": ("charging with the travel case", "travel case", "charging travel case", "usb wall adapter", "battery indicator"),
    "出厂重置": ("factory reset screen", "factory reset", "reset button", "yes button", "factory default settings"),
    "间接烹饪": ("indirect cooking", "indirect heat", "lid close", "slow roasting", "baking", "flare-ups"),
    "手动记忆频道": ("manual program", "memorizing channels", "memory/erase", "channel number", "memorize or erase"),
    "室外天线": ("outdoor antenna", "antenna jack", "300 ohm", "75 ohm", "coaxial cable", "flat wire"),
    "防堵罩": ("anti-block shield", "anti block shield", "prevents food particles", "steam release pipe", "prongs"),
    "P模式": ("p program ae", "program ae", "creative zone", "mode dial", "shutter speed", "aperture value"),
    "无法抽水": ("无法抽水", "does not pump", "cannot pump", "unable to pump"),
    "扫地机器人部件": ("robot anatomy", "faceplate", "bin release", "clean button", "handle", "dust bin", "sensor"),
    "储物舱": ("储物舱", "储物格", "储物箱", "storage compartment", "storage compartments", "wet items"),
    "拆下快门按钮": ("拆下快门按钮", "移除快门按钮", "remove shutter button", "remove the camera shutter button"),
    "目镜盖": ("目镜盖", "接目镜盖", "eyepiece cover"),
}


@dataclass(slots=True)
class ChatReply:
    answer: str
    session_id: str
    timestamp: int
    references: list[dict]
    related_images: list[dict]


class ChatService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.repository = KnowledgeBaseRepository(self.settings)
        self.memory = SessionMemory(
            ttl_seconds=self.settings.session_ttl_seconds,
            max_turns=self.settings.max_session_turns,
        )
        self.multimodal = MultimodalUnderstandingService()
        self.embedding_client = EmbeddingClient(self.settings)
        self.llm_client = LLMClient(self.settings)
        self.reranker_client = RerankerClient(self.settings)
        self.query_rewriter = QueryRewriteService(self.llm_client)
        self.llm_query_planner = LLMQueryPlanner(self.llm_client)
        self.generator = EvidenceGroundedGenerator(self.llm_client)
        self.guardrail = GuardrailService()
        self._retriever: HybridRetriever | None = None
        self._sidecar_retriever: SidecarHybridRetriever | None = None
        self._dense_retriever: DenseEmbeddingRetriever | None = None
        self._retriever_version = 0.0
        self._agent_graph = None

    def _agent_graph_enabled(self) -> bool:
        return self.settings.agent_graph_enabled

    def _get_agent_graph(self):
        if self._agent_graph is None:
            from app.core.agent_graph import AgentGraph

            self._agent_graph = AgentGraph(self)
        return self._agent_graph

    def _chat_reply_from_agent_state(self, state) -> ChatReply:
        response = state.final_response or {}
        return ChatReply(
            answer=response.get("answer", state.answer),
            session_id=response.get("session_id", state.session_id or ""),
            timestamp=response.get("timestamp", int(time.time())),
            references=response.get("references", state.references),
            related_images=response.get("related_images", state.related_images),
        )

    def _diagnostic_from_agent_state(self, state, insight=None) -> dict:
        sub_question_results = [
            (
                state.resolved_question or state.question,
                state.raw_accepted_evidence or state.raw_reranked_candidates or state.raw_candidates,
            )
        ]
        query_trace = {
            "original_query": state.resolved_question or state.question,
            "query_en": state.query_variants[0] if state.query_variants else state.resolved_question or state.question,
            "query_zh": None,
            "query_mix": state.resolved_question or state.question,
            "search_keywords": [*state.must_terms, *state.should_terms],
            "product_hint": state.product,
            "intent": state.intent,
            "need_image": bool(state.image_ids or state.related_images),
            "planned_sub_questions": [],
        }
        refusal = self._diagnose_refusal(state.answer, sub_question_results)
        coverage_gap = self._diagnose_coverage_gap(refusal, state.product, [query_trace], sub_question_results)
        return {
            "question": state.resolved_question or state.question,
            "session_id": state.session_id,
            "english_dominant": looks_english_dominant(state.resolved_question or state.question),
            "multimodal_insight": {
                "manual_hints": getattr(insight, "manual_hints", []),
                "product_hint": state.product or getattr(insight, "product_hint", None),
                "image_warnings": getattr(insight, "warnings", []),
            },
            "query_rewrite": [query_trace],
            "retrieval": [
                {
                    "sub_question": state.resolved_question or state.question,
                    "query_variants": state.query_variants,
                    "boost_manuals": state.manual_scope,
                    "allowed_manuals": state.manual_scope,
                    "bm25_top_k": {},
                    "vector_top_k": {},
                    "hybrid_top_k": {},
                    "fusion_top_k": state.candidates,
                    "rerank_top_k": state.reranked_candidates,
                    "generation_top_k": state.accepted_evidence,
                }
            ],
            "generation": {
                "evidence_preview": [
                    {
                        "sub_question": state.resolved_question or state.question,
                        "evidence": state.accepted_evidence[:3],
                    }
                ],
                "llm_messages": [],
                "answer": state.answer,
                "references": state.references,
                "related_images": state.related_images,
            },
            "refusal": refusal,
            "coverage_gap": coverage_gap,
            "notes": {
                "agent_graph": "enabled",
                "verifier": state.verifier_result,
                "fallback_reason": state.fallback_reason,
                "trace_nodes": [item.get("node") for item in state.trace],
                "vector_retrieval": ("embedding_api" if self.settings.dense_enabled and self.embedding_client.is_enabled() and self._get_dense_retriever() is not None and self._get_dense_retriever().is_ready() else "not_configured"),
                "reranker": ("cross_encoder_api" if self.settings.rerank_enabled and self.reranker_client.is_enabled() else "not_configured"),
            },
        }

    def _get_retriever(self) -> HybridRetriever:
        self.repository.ensure_ready()
        if self._retriever is None or self._retriever_version != self.repository.version:
            self._retriever = HybridRetriever(self.repository.get_chunks())
            self._sidecar_retriever = SidecarHybridRetriever(
                self.repository.get_retrieval_corpus(),
                self.repository.get_chunk_lookup(),
            )
            self._dense_retriever = None
            if self.settings.dense_enabled and self.embedding_client.is_enabled():
                self._dense_retriever = DenseEmbeddingRetriever(
                    self.repository.get_dense_index(),
                    self.repository.get_chunk_lookup(),
                )
            self._retriever_version = self.repository.version
        return self._retriever

    def _get_sidecar_retriever(self) -> SidecarHybridRetriever:
        self._get_retriever()
        assert self._sidecar_retriever is not None
        return self._sidecar_retriever

    def _get_dense_retriever(self) -> DenseEmbeddingRetriever | None:
        self._get_retriever()
        return self._dense_retriever

    def _dense_route_results(
        self,
        dense_retriever: DenseEmbeddingRetriever | None,
        query_text: str,
        boost_manuals: list[str],
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        if not query_text or dense_retriever is None or not dense_retriever.is_ready():
            return []
        if not self.settings.dense_enabled or not self.embedding_client.is_enabled():
            return []
        vectors = self.embedding_client.embed([query_text])
        if not vectors:
            return []
        return dense_retriever.search_query_vector(
            vectors[0],
            top_k=max(self.settings.dense_top_k, self.settings.default_top_k),
            boost_manuals=boost_manuals,
            query_text=query_text,
            allowed_manuals=allowed_manuals,
        )

    def _fusion_candidate_limit(self) -> int:
        if self.settings.rerank_enabled and self.reranker_client.is_enabled():
            return max(self.settings.default_top_k * 8, self.settings.rerank_max_candidates, 24)
        return max(self.settings.default_top_k * 4, 12)

    def _model_rerank_candidates(
        self,
        query_text: str,
        candidates: list[SearchResult],
        boost_manuals: list[str],
        top_k: int,
    ) -> list[SearchResult] | None:
        if not self.settings.rerank_enabled or not self.reranker_client.is_enabled():
            return None
        if not query_text or not candidates:
            return None
        if not self._should_use_model_reranker(query_text, candidates, boost_manuals):
            return None

        limited = candidates[: max(self.settings.rerank_max_candidates, top_k)]
        target_n = min(max(top_k * 3, self.settings.rerank_top_n), len(limited))
        documents = [
            "\n".join(
                part
                for part in (
                    result.chunk.manual_name,
                    result.chunk.product_name,
                    result.chunk.section_title,
                    result.chunk.text[:1800],
                )
                if part
            )
            for result in limited
        ]
        reranked = self.reranker_client.rerank(
            query_text,
            documents,
            top_n=target_n,
        )
        if reranked is None:
            return None
        if not reranked:
            return []

        anchor_score = max(result.score for result in limited)
        rescored: list[SearchResult] = []
        for position, item in enumerate(reranked, start=1):
            if item.index < 0 or item.index >= len(limited):
                continue
            original = limited[item.index]
            score = max(0.0, anchor_score - 0.015 * (position - 1))
            if boost_manuals and original.chunk.manual_name in boost_manuals:
                score += 0.02
            score += max(0.0, min(0.08, item.relevance_score * 0.01))
            rescored.append(
                SearchResult(
                    chunk=original.chunk,
                    score=score,
                    bm25_score=original.bm25_score,
                    semantic_score=original.semantic_score,
                    matched_terms=original.matched_terms,
                )
            )
        rescored = self._slot_coverage_adjusted_results(rescored, {"query": query_text}, boost_manuals)
        return rescored[:target_n]

    def _action_terms_from_text(self, query_text: str) -> set[str]:
        lowered = query_text.lower()
        terms = {term for term in ACTION_HINT_TERMS if term in lowered}
        terms.update({phrase for phrase in CRITICAL_ACTION_PHRASES if phrase in lowered})
        return terms

    def _text_implies_delete_intent(self, query_text: str) -> bool:
        lowered = query_text.lower()
        return (
            ("delete" in lowered or "erase" in lowered or "删除" in lowered)
            and ("image" in lowered or "images" in lowered or "图像" in lowered)
        )

    def _should_use_model_reranker(
        self,
        query_text: str,
        candidates: list[SearchResult],
        boost_manuals: list[str],
    ) -> bool:
        if len(candidates) < 2:
            return False

        top = candidates[0]
        second = candidates[1]
        top_gap = top.score - second.score
        combined_top = f"{top.chunk.section_title} {top.chunk.text}"
        query_variants = {"query": query_text}
        action_terms = self._action_terms_from_text(query_text)
        force_rerank = os.getenv("RERANK_FORCE", "1").strip().lower() in {"1", "true", "yes", "on"}

        if self._text_implies_delete_intent(query_text):
            return False
        if self._looks_parameter_query(query_variants) and self._has_explicit_numeric_answer(combined_top):
            return False
        if force_rerank:
            return True
        if action_terms and self._result_matches_query_actions(top, action_terms) and top_gap >= 0.12:
            return False
        if boost_manuals and top.chunk.manual_name in boost_manuals and top_gap >= 0.18 and not self._is_generic_heading(top.chunk.section_title):
            return False
        if self._is_generic_gap_chunk(top) or self._is_generic_heading(top.chunk.section_title):
            return True

        top_manuals = {result.chunk.manual_name for result in candidates[:3]}
        if len(top_manuals) > 1:
            return True
        if top_gap <= 0.12:
            return True
        return False

    def _resolve_manual_hints_from_product(self, product_hint: str | None) -> list[str]:
        if not product_hint:
            return []
        lowered = product_hint.lower().strip()
        alias_lookup = self.repository.get_alias_lookup()
        matched: list[str] = []
        for aliases, manual_names in PRODUCT_MANUAL_HINTS:
            if any(alias in lowered for alias in aliases):
                for manual_name in manual_names:
                    if manual_name not in matched:
                        matched.append(manual_name)
        for alias, manual_name in alias_lookup.items():
            if alias == lowered or lowered in alias or alias in lowered:
                if manual_name not in matched:
                    matched.append(manual_name)
        return matched[:3]

    def _infer_product_hint_from_text(self, texts: list[str]) -> str | None:
        lowered_texts = [text.lower() for text in texts if text]
        if not lowered_texts:
            return None
        for aliases, product_hint in TEXT_PRODUCT_HINTS:
            for alias in aliases:
                if any(alias in text for text in lowered_texts):
                    return product_hint
        return None

    def _llm_query_plan_enabled(self) -> bool:
        return os.getenv("LLM_QUERY_PLAN_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

    def _llm_query_planner_enabled(self) -> bool:
        return os.getenv("LLM_QUERY_PLANNER_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

    def _question_language(self, question: str) -> str:
        return "en" if looks_english_dominant(question) else "zh"

    def _high_confidence_direct_intent(self, question: str) -> bool:
        try:
            plan = self.generator._build_evidence_plan(question)
            return plan.intent in self.generator._preplanned_direct_intents()
        except Exception:
            return False

    def _known_bad_intent_query(self, question: str) -> bool:
        lowered = question.lower()
        bad_patterns = (
            ("energy saving", "cooling"),
            ("节能", "制冷"),
            ("drill", "charge"),
            ("电钻", "充电"),
            ("precision", "equipment"),
            ("精密", "设备"),
            ("ear", "earbud"),
            ("耳塞", "耳机"),
            ("strap", "camera"),
            ("肩带", "相机"),
            ("t_sensor",),
            ("t-sensor",),
            ("snowmobile",),
            ("上坡", "雪地"),
            ("下坡", "雪地"),
            ("横坡", "雪地"),
        )
        return any(all(term in lowered for term in pattern) for pattern in bad_patterns)

    def _slot_coverage_low(self, results: list[SearchResult], variants: dict[str, str]) -> bool:
        if not results or not self._slot_aliases_from_query(variants):
            return False
        covered, total = self._slot_coverage_count(results[0].chunk, variants)
        return bool(total and covered < total)

    def _should_call_llm_query_planner(
        self,
        question: str,
        variants: dict[str, str],
        candidates: list[SearchResult],
        boost_manuals: list[str],
    ) -> bool:
        if not self._llm_query_planner_enabled() or not self.llm_client.is_enabled():
            return False
        if self._high_confidence_direct_intent(question):
            return False
        if self._known_bad_intent_query(question):
            return True
        if not candidates:
            return True
        if self._slot_coverage_low(candidates, variants):
            return True
        top = candidates[0]
        if self._is_generic_gap_chunk(top) or self._is_generic_heading(top.chunk.section_title):
            return True
        if len(candidates) > 1 and top.score - candidates[1].score <= 0.08:
            return True
        if boost_manuals and top.chunk.manual_name not in boost_manuals and self._query_has_product_signal(variants):
            return True
        return False

    def _manual_scope_from_plan(self, plan: LLMQueryPlan | None) -> list[str]:
        if plan is None:
            return []
        known_manuals = {chunk.manual_name for chunk in self.repository.get_chunks()}
        known_by_lower = {manual_name.lower(): manual_name for manual_name in known_manuals}
        scoped: list[str] = []

        def add(manual_name: str) -> None:
            if manual_name and manual_name not in scoped:
                scoped.append(manual_name)

        for raw in [plan.product or "", *plan.manual_scope]:
            candidate = raw.strip()
            if not candidate:
                continue
            exact = known_by_lower.get(candidate.lower())
            if exact:
                add(exact)
                continue
            for manual_name in self._resolve_manual_hints_from_product(candidate):
                add(manual_name)
            for manual_name in self._resolve_manual_hints_from_text([candidate]):
                add(manual_name)
        if len(scoped) > 4:
            return []
        return scoped[:4]

    def _term_matches_text(self, term: str, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (term or "").strip().lower())
        if len(normalized) < 2:
            return False
        text_lower = text.lower()
        if re.fullmatch(r"[a-z0-9_+-]+(?:\s+[a-z0-9_+-]+)*", normalized):
            pattern = rf"(?<![a-z0-9_+-]){re.escape(normalized)}(?![a-z0-9_+-])"
            return re.search(pattern, text_lower) is not None
        return normalized in text_lower

    def _planner_combined_text(self, result: SearchResult) -> str:
        chunk = result.chunk
        return f"{chunk.manual_name} {chunk.product_name} {chunk.section_title} {chunk.text}"

    def _planner_term_hits(self, terms: list[str], text: str) -> int:
        return sum(1 for term in terms if self._term_matches_text(term, text))

    def _planner_exclude_hit(self, plan: LLMQueryPlan, text: str) -> bool:
        ignored = {"manual", "product", "question", "answer", "步骤", "说明", "使用"}
        for term in plan.exclude_terms:
            if len(term.strip()) < 4 or term.strip().lower() in ignored:
                continue
            if self._term_matches_text(term, text):
                return True
        return False

    def _planner_rescore_results(
        self,
        results: list[SearchResult],
        plan: LLMQueryPlan | None,
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        if plan is None or not results:
            return results
        allowed = set(allowed_manuals or [])
        adjusted: list[SearchResult] = []
        for result in results:
            text = self._planner_combined_text(result)
            if allowed and result.chunk.manual_name not in allowed:
                continue
            must_hits = self._planner_term_hits(plan.must_terms, text)
            object_hits = self._planner_term_hits(plan.object_terms, text)
            should_hits = self._planner_term_hits(plan.should_terms, text)
            if self._planner_exclude_hit(plan, text) and must_hits == 0 and object_hits == 0:
                continue
            score = result.score
            score += 0.34 * must_hits
            score += 0.28 * object_hits
            score += 0.10 * should_hits
            if plan.must_terms and must_hits == 0:
                score -= 0.65
            if plan.object_terms and object_hits == 0:
                score -= 0.45
            adjusted.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    bm25_score=result.bm25_score,
                    semantic_score=result.semantic_score,
                    matched_terms=result.matched_terms,
                )
            )
        if not adjusted:
            return results
        adjusted.sort(key=lambda item: (-item.score, -item.bm25_score, item.chunk.order))
        return adjusted

    def _validate_evidence_with_plan(
        self,
        results: list[SearchResult],
        plan: LLMQueryPlan | None,
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        if plan is None or not results:
            return results
        allowed = set(allowed_manuals or [])
        validated: list[SearchResult] = []
        for result in results:
            text = self._planner_combined_text(result)
            if allowed and result.chunk.manual_name not in allowed:
                continue
            if self._planner_exclude_hit(plan, text):
                continue
            must_hits = self._planner_term_hits(plan.must_terms, text)
            object_hits = self._planner_term_hits(plan.object_terms, text)
            score = result.score
            if plan.must_terms:
                required = 1 if len(plan.must_terms) <= 2 else 2
                if must_hits < required:
                    score -= 0.85
            if plan.object_terms and object_hits == 0:
                score -= 0.45
            validated.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    bm25_score=result.bm25_score,
                    semantic_score=result.semantic_score,
                    matched_terms=result.matched_terms,
                )
            )
        if not validated:
            return results
        validated.sort(key=lambda item: (-item.score, -item.bm25_score, item.chunk.order))
        return validated

    def _planner_rerank_query(self, query: str, variants: dict[str, str], plan: LLMQueryPlan | None) -> str:
        if plan is None:
            return variants.get("query_en") or query
        parts = [
            query,
            " ".join(plan.action_terms[:5]),
            " ".join(plan.object_terms[:6]),
            " ".join(plan.must_terms[:6]),
        ]
        return " ".join(part for part in parts if part).strip()

    def _apply_llm_query_planner(
        self,
        retriever: HybridRetriever,
        query: str,
        rewritten,
        variants: dict[str, str],
        bm25_routes: dict[str, list[SearchResult]],
        hybrid_routes: dict[str, list[SearchResult]],
        vector_routes: dict[str, list[SearchResult]],
        fused: list[SearchResult],
        boost_manuals: list[str],
        allowed_manuals: list[str] | None,
    ) -> tuple[list[SearchResult], dict[str, str], list[str], list[str] | None, LLMQueryPlan | None]:
        if not self._should_call_llm_query_planner(query, variants, fused, boost_manuals):
            return fused, variants, boost_manuals, allowed_manuals, None

        product_hint = (
            getattr(rewritten, "product_hint", None)
            or self._infer_product_hint_from_text([query, " ".join(variants.values())])
        )
        plan = self.llm_query_planner.plan(
            question=query,
            product=product_hint,
            language=self._question_language(query),
            candidates=fused[:8],
        )
        if plan is None or plan.confidence < 0.35:
            return fused, variants, boost_manuals, allowed_manuals, None

        planner_manuals = self._manual_scope_from_plan(plan)
        planner_allowed = planner_manuals or allowed_manuals
        planner_boost = list(boost_manuals)
        for manual_name in planner_manuals:
            if manual_name not in planner_boost:
                planner_boost.append(manual_name)

        for route_name, route_query in plan.as_variants().items():
            if not route_query or route_query in variants.values():
                continue
            variants[route_name] = route_query
            bm25_routes[route_name] = retriever.search_bm25(
                route_query,
                top_k=max(self.settings.default_top_k, 5),
                boost_manuals=planner_boost,
                allowed_manuals=planner_allowed,
            )
            hybrid_routes[route_name] = retriever.score_query(
                route_query,
                boost_manuals=planner_boost,
                allowed_manuals=planner_allowed,
            )[: max(self.settings.default_top_k * 3, 8)]

        fused = self._fuse_route_results(
            {**hybrid_routes, **vector_routes},
            variants,
            planner_boost,
            self._fusion_candidate_limit(),
        )
        fused = self._planner_rescore_results(fused, plan, planner_allowed)
        fused = self._slot_coverage_adjusted_results(fused, variants, planner_boost)
        return fused, variants, planner_boost, planner_allowed, plan

    def _resolve_manual_hints_from_text(self, texts: list[str]) -> list[str]:
        alias_lookup = self.repository.get_alias_lookup()
        matches: list[tuple[int, str]] = []
        lowered_texts = [text.lower() for text in texts if text]
        manual_order: list[str] = []
        for aliases, manual_names in PRODUCT_MANUAL_HINTS:
            if any(alias in text for alias in aliases for text in lowered_texts):
                for manual_name in manual_names:
                    if manual_name not in manual_order:
                        manual_order.append(manual_name)
        for alias, manual_name in alias_lookup.items():
            if not alias:
                continue
            if any(alias in text for text in lowered_texts):
                matches.append((len(alias), manual_name))
        ordered: list[str] = list(manual_order)
        seen: set[str] = set()
        for manual_name in ordered:
            seen.add(manual_name)
        for _, manual_name in sorted(matches, key=lambda item: (-item[0], item[1])):
            if manual_name not in seen:
                seen.add(manual_name)
                ordered.append(manual_name)
        return ordered[:3]

    def _alias_matches_text(self, alias: str, lowered_text: str) -> bool:
        normalized = alias.strip().lower()
        if not normalized:
            return False
        if re.fullmatch(r"[a-z0-9_+-]+(?:\s+[a-z0-9_+-]+)*", normalized):
            pattern = rf"(?<![a-z0-9_+-]){re.escape(normalized)}(?![a-z0-9_+-])"
            return re.search(pattern, lowered_text) is not None
        return normalized in lowered_text

    def _scope_safe_alias(self, alias: str) -> bool:
        normalized = alias.strip().lower()
        if not normalized:
            return False
        if re.fullmatch(r"[a-z0-9_+-]+", normalized):
            return len(normalized) >= 4
        return len(normalized) >= 2

    def _resolve_allowed_manuals(
        self,
        texts: list[str],
        product_hint: str | None = None,
    ) -> list[str]:
        lowered_text = " ".join(text.lower() for text in texts if text)
        allowed: list[str] = []

        def add(manual_name: str) -> None:
            if manual_name and manual_name not in allowed:
                allowed.append(manual_name)

        if product_hint:
            for manual_name in self._resolve_manual_hints_from_product(product_hint):
                add(manual_name)

        for aliases, manual_names in PRODUCT_MANUAL_HINTS:
            if any(self._alias_matches_text(alias, lowered_text) for alias in aliases):
                for manual_name in manual_names:
                    add(manual_name)

        # More than four manuals means the signal is too broad; keep retrieval open.
        if len(allowed) > 4:
            return []
        return allowed[:4]

    def _apply_manual_preference(self, results: list, boost_manuals: list[str]) -> list:
        if not boost_manuals:
            return results
        preferred = [result for result in results if result.chunk.manual_name in boost_manuals]
        others = [result for result in results if result.chunk.manual_name not in boost_manuals]
        return (preferred + others)[: self.settings.default_top_k]

    def _query_variants(self, original_query: str, rewritten) -> dict[str, str]:
        candidates = [
            ("query_en", getattr(rewritten, "query_en", None) or original_query),
            ("query_zh", getattr(rewritten, "query_zh", None)),
            ("query_mix", getattr(rewritten, "query_mix", None) or getattr(rewritten, "retrieval_query", None)),
        ]
        variants: dict[str, str] = {}
        seen: set[str] = set()
        for route_name, query in candidates:
            if not query:
                continue
            normalized = query.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            variants[route_name] = normalized
        slot_query = self._slot_expanded_query(variants)
        if slot_query and slot_query not in seen:
            variants["query_slot"] = slot_query
        return variants

    def _slot_expanded_query(self, query_variants: dict[str, str]) -> str:
        if not query_variants:
            return ""
        base = query_variants.get("query_en") or next(iter(query_variants.values()), "")
        base_lower = base.lower()
        expansion_terms: list[str] = []
        seen: set[str] = set()
        for aliases in self._slot_aliases_from_query(query_variants):
            for alias in aliases[:4]:
                normalized = alias.strip().lower()
                if len(normalized) < 3 or normalized in base_lower or normalized in seen:
                    continue
                seen.add(normalized)
                expansion_terms.append(normalized)
                if len(expansion_terms) >= 10:
                    break
            if len(expansion_terms) >= 10:
                break
        if not expansion_terms:
            return ""
        return f"{base} {' '.join(expansion_terms)}"

    def _fuse_route_results(
        self,
        route_results: dict[str, list[SearchResult]],
        query_variants: dict[str, str],
        boost_manuals: list[str],
        top_k: int,
    ) -> list[SearchResult]:
        route_weights = {
            "query_en": 0.8,
            "query_en_sidecar": 0.92,
            "query_en_dense": 0.9,
            "query_zh": 1.05,
            "query_mix": 1.0,
            "query_slot": 1.15,
            "keyword_title": 1.3,
            "intent_anchor": 1.8,
        }
        merged: dict[str, dict] = {}
        for route_name, results in route_results.items():
            weight = route_weights.get(route_name, 1.0)
            for rank, result in enumerate(results[: max(top_k * 3, 8)], start=1):
                bucket = merged.setdefault(
                    result.chunk.chunk_id,
                    {
                        "chunk": result.chunk,
                        "score": 0.0,
                        "bm25_score": 0.0,
                        "semantic_score": 0.0,
                        "matched_terms": set(),
                        "hits": 0,
                    },
                )
                rank_bonus = max(0.0, 0.18 - 0.02 * (rank - 1))
                bucket["score"] += weight * result.score + rank_bonus
                bucket["bm25_score"] = max(bucket["bm25_score"], result.bm25_score)
                bucket["semantic_score"] = max(bucket["semantic_score"], result.semantic_score)
                bucket["matched_terms"].update(result.matched_terms)
                bucket["hits"] += 1

        parameter_query = self._looks_parameter_query(query_variants)
        slot_aliases = self._slot_aliases_from_query(query_variants)
        fused: list[SearchResult] = []
        for payload in merged.values():
            score = payload["score"] + max(0, payload["hits"] - 1) * 0.06
            score += self._informative_match_bonus(payload["matched_terms"], query_variants)
            score += self._slot_coverage_score(payload["chunk"], query_variants, boost_manuals, slot_aliases)
            if parameter_query:
                score += self._parameter_answer_bonus(
                    payload["chunk"],
                    payload["bm25_score"],
                    payload["matched_terms"],
                    query_variants,
                    boost_manuals,
                )
            fused.append(
                SearchResult(
                    chunk=payload["chunk"],
                    score=score,
                    bm25_score=payload["bm25_score"],
                    semantic_score=payload["semantic_score"],
                    matched_terms=sorted(payload["matched_terms"]),
                )
            )
        fused.sort(key=lambda item: (-item.score, -item.bm25_score, item.chunk.order))
        return fused[:top_k]

    def _slot_aliases_from_query(self, query_variants: dict[str, str]) -> list[tuple[str, ...]]:
        query_text = " ".join(query_variants.values()).lower()
        anti_block_query = any(term in query_text for term in ("anti-block shield", "anti block shield", "anti-block"))
        slots: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for aliases, canonical in QUERY_PHRASE_HINTS:
            if any(alias.lower() in query_text for alias in aliases):
                expanded = tuple(alias.lower() for alias in self._phrase_aliases(canonical))
                if expanded not in seen:
                    seen.add(expanded)
                    slots.append(expanded)
        extra_slots: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
            (("quick release", "qpr"), ("quick release", "quick release button", "vent position")),
            (("natural release", "npr"), ("natural release", "depressurizes naturally", "temperature drops")),
            (("float valve",), ("float valve", "silicone cap")),
            (("sealing ring",), ("sealing ring", "air-tight seal")),
            (("memory card",), ("memory card", "card slot", "insert")),
            (("camera battery",), ("battery", "charge", "usb")),
            (("base station",), ("base station", "dc input jack", "telephone socket")),
            (("fuse",), ("fuse", "fuse puller", "spare fuse")),
            (("over temperature",), ("over temperature", "cooling water pilot outlet")),
            (("swim platform",), ("swim platform", "wet storage", "rear platform hatch")),
            (("travel case",), ("travel case", "usb wall adapter", "battery indicator")),
            (("anti-block shield", "anti block shield", "anti-block"), ("anti-block shield", "prevents food particles", "steam release pipe", "prongs")),
            (("p mode", "p model", "program ae", "program mode"), ("p program ae", "program ae", "mode dial", "shutter speed", "aperture value")),
            (("unable to pump", "cannot pump", "does not pump", "无法抽水"), ("无法抽水", "软管接头", "卡箍松动", "o形圈", "机械密封")),
            (("robot anatomy", "vacuum anatomy", "robot parts"), ("faceplate", "bin release", "clean button", "handle", "dust bin", "sensor")),
            (("first use", "first time", "before first use"), ("before first use", "remove packaging", "clean", "wash")),
            (("steer", "steers", "steering", "turn"), ("steering", "jet thrust", "throttle", "steering wheel")),
            (("protective equipment", "ppe", "防护装备"), ("hearing protection", "eye protection", "face mask", "防滑鞋", "急救箱")),
            (("auto restart", "自动重启"), ("auto restart", "power failure", "on/off button", "自动重启")),
            (("components", "部件", "组成部件"), ("components", "部件介绍", "main components")),
            (("spray arm", "喷淋臂"), ("spray arm", "喷淋臂", "clogged", "clean")),
            (("chair functions", "椅子功能"), ("高度调节", "后仰", "扶手", "按摩功能")),
            (("air purifier", "空气净化器", "purifier"), ("常规运行", "自动", "睡眠", "空气质量", "风速")),
        )
        for triggers, aliases in extra_slots:
            if anti_block_query and triggers == ("float valve",):
                continue
            if any(trigger in query_text for trigger in triggers):
                expanded = tuple(alias.lower() for alias in aliases)
                if expanded not in seen:
                    seen.add(expanded)
                    slots.append(expanded)
        quoted_p_query = (('\\"p\\"' in query_text or '"p"' in query_text or "“p”" in query_text) and any(term in query_text for term in ("mode", "model")))
        if quoted_p_query or re.search(r'["“”]?\s*p\s*["“”]?\s*(?:mode|model)', query_text) or re.search(
            r'(?:mode|model)\s*["“”]?\s*p\s*["“”]?',
            query_text,
        ):
            expanded = ("p program ae", "program ae", "mode dial", "shutter speed", "aperture value")
            if expanded not in seen:
                seen.add(expanded)
                slots.append(expanded)
        return slots[:4]

    def _query_has_product_signal(self, query_variants: dict[str, str]) -> bool:
        query_text = " ".join(query_variants.values()).lower()
        return any(alias in query_text for aliases, _ in PRODUCT_MANUAL_HINTS for alias in aliases)

    def _slot_coverage_score(
        self,
        chunk,
        query_variants: dict[str, str],
        boost_manuals: list[str],
        slot_aliases: list[tuple[str, ...]],
    ) -> float:
        combined = f"{chunk.manual_name} {chunk.product_name} {chunk.section_title} {chunk.text}".lower()
        score = 0.0
        if boost_manuals:
            if chunk.manual_name in boost_manuals:
                score += 0.75
            elif self._query_has_product_signal(query_variants):
                score -= 0.90

        covered_slots = 0
        for aliases in slot_aliases:
            if any(alias in combined for alias in aliases):
                score += 0.75
                covered_slots += 1
            else:
                score -= 0.45

        if slot_aliases and covered_slots == 0:
            score -= 0.70

        if self._is_generic_heading(chunk.section_title) or chunk.chunk_type in {"toc", "title_only"}:
            score -= 0.70
        return score

    def _slot_coverage_count(self, chunk, query_variants: dict[str, str]) -> tuple[int, int]:
        slot_aliases = self._slot_aliases_from_query(query_variants)
        if not slot_aliases:
            return 0, 0
        combined = f"{chunk.manual_name} {chunk.product_name} {chunk.section_title} {chunk.text}".lower()
        covered = sum(1 for aliases in slot_aliases if any(alias in combined for alias in aliases))
        return covered, len(slot_aliases)

    def _slot_coverage_adjusted_results(
        self,
        results: list[SearchResult],
        query_variants: dict[str, str],
        boost_manuals: list[str],
    ) -> list[SearchResult]:
        if not results or not self._slot_aliases_from_query(query_variants):
            return results
        adjusted: list[SearchResult] = []
        for result in results:
            covered, total = self._slot_coverage_count(result.chunk, query_variants)
            score = result.score
            if total:
                if covered == 0:
                    score -= 1.2
                elif covered < total:
                    score -= 0.25 * (total - covered)
                else:
                    score += 0.25
            score += self._slot_coverage_score(result.chunk, query_variants, boost_manuals, self._slot_aliases_from_query(query_variants)) * 0.35
            adjusted.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                    bm25_score=result.bm25_score,
                    semantic_score=result.semantic_score,
                    matched_terms=result.matched_terms,
                )
            )
        adjusted.sort(key=lambda item: (-item.score, -item.bm25_score, item.chunk.order))
        return adjusted

    def _keyword_title_results(
        self,
        retriever: HybridRetriever,
        rewritten,
        boost_manuals: list[str],
        variants: dict[str, str],
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        phrases: list[str] = []
        if rewritten is not None:
            phrases.extend(getattr(rewritten, "search_keywords", []) or [])
        phrases.extend(variants.values())
        for aliases in self._slot_aliases_from_query(variants):
            phrases.extend(aliases)

        normalized: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            phrase = re.sub(r"\s+", " ", phrase or "").strip().lower()
            if len(phrase) < 4 or phrase in seen:
                continue
            seen.add(phrase)
            normalized.append(phrase)

        if not normalized:
            return []

        results: list[SearchResult] = []
        allowed = set(allowed_manuals or [])
        for chunk in retriever.chunks:
            if allowed and chunk.manual_name not in allowed:
                continue
            if boost_manuals and chunk.manual_name not in boost_manuals:
                continue
            title_lower = chunk.section_title.lower()
            combined_lower = f"{chunk.section_title} {chunk.text}".lower()
            title_hits = 0
            body_hits = 0
            for phrase in normalized:
                if phrase in title_lower:
                    title_hits += 1
                elif phrase in combined_lower:
                    body_hits += 1
            if title_hits == 0:
                continue
            score = 2.0 + 0.6 * title_hits + 0.12 * body_hits
            if boost_manuals and chunk.manual_name in boost_manuals:
                score += 0.2
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    bm25_score=score,
                    semantic_score=0.0,
                    matched_terms=sorted(set(tokenize(" ".join(normalized))) & set(tokenize(combined_lower))),
                )
            )

        results.sort(key=lambda item: (-item.score, item.chunk.order))
        return results[:12]

    def _intent_anchor_results(
        self,
        retriever: HybridRetriever,
        rewritten,
        boost_manuals: list[str],
        variants: dict[str, str],
        root_query: str = "",
        allowed_manuals: list[str] | None = None,
    ) -> list[SearchResult]:
        query_parts = [
            root_query,
            getattr(rewritten, "original_query", "") if rewritten is not None else "",
            *variants.values(),
            " ".join(getattr(rewritten, "search_keywords", []) or []) if rewritten is not None else "",
            getattr(rewritten, "product_hint", "") if rewritten is not None else "",
        ]
        query_text = " ".join(str(part) for part in query_parts if part).lower()
        anchors: list[SearchResult] = []
        allowed = set(allowed_manuals or [])

        def add_if(
            predicate,
            *,
            base_score: float,
            required_manual_terms: tuple[str, ...] = (),
        ) -> None:
            for chunk in retriever.chunks:
                if allowed and chunk.manual_name not in allowed:
                    continue
                manual_product = f"{chunk.manual_name} {chunk.product_name}".lower()
                if required_manual_terms and not any(term.lower() in manual_product for term in required_manual_terms):
                    continue
                combined = f"{chunk.section_title} {chunk.text}".lower()
                if not predicate(chunk, combined):
                    continue
                title_lower = chunk.section_title.lower()
                score = base_score
                if chunk.chunk_type == "step":
                    score += 0.3
                if any(token in title_lower for token in ("replacement", "connect", "quick release", "natural release", "float valve")):
                    score += 0.35
                anchors.append(
                    SearchResult(
                        chunk=chunk,
                        score=score,
                        bm25_score=score,
                        semantic_score=0.0,
                        matched_terms=sorted(set(tokenize(query_text)) & set(tokenize(combined))),
                )
            )

        if (
            any(term in query_text for term in ("air fryer", "airfryer", "空气炸锅"))
            and any(term in query_text for term in ("first use", "first time", "before first use", "首次", "第一次"))
        ):
            add_if(
                lambda chunk, combined: (
                    ("first use" in combined or "before first use" in combined or "首次" in combined)
                    and any(term in combined for term in ("remove", "packaging", "clean", "wash", "清洁", "包装"))
                    and not any(term in combined for term in ("wifi", "wi-fi", "nutriu", "pairing"))
                ),
                base_score=5.4,
                required_manual_terms=("空气炸锅", "air fryer"),
            )

        if (
            any(term in query_text for term in ("blower", "吹风机"))
            and any(term in query_text for term in ("protective equipment", "personal protective", "ppe", "防护装备", "佩戴"))
        ):
            add_if(
                lambda chunk, combined: (
                    ("个人防护装备" in combined or "protective equipment" in combined)
                    and any(term in combined for term in ("听力防护", "眼部防护", "面罩", "hearing protection", "eye protection"))
                ),
                base_score=5.8,
                required_manual_terms=("吹风机", "blower"),
            )

        if any(term in query_text for term in ("air conditioner", "空调")):
            if any(term in query_text for term in ("components", "parts", "组成部件", "部件")):
                add_if(
                    lambda chunk, combined: "部件介绍" in combined and ("室内机" in combined or "indoor unit" in combined),
                    base_score=5.8,
                    required_manual_terms=("空调", "air conditioner"),
                )
            if "auto restart" in query_text or "自动重启" in query_text:
                add_if(
                    lambda chunk, combined: ("自动重启" in combined or "auto restart" in combined) and ("6" in combined or "蜂鸣" in combined),
                    base_score=5.8,
                    required_manual_terms=("空调", "air conditioner"),
                )

        if any(term in query_text for term in ("ergonomic chair", "office chair", "人体工学椅", "椅子")):
            if any(term in query_text for term in ("function", "functions", "功能")):
                add_if(
                    lambda chunk, combined: "高度调节" in combined and ("椅背后仰" in combined or "按摩功能" in combined),
                    base_score=5.8,
                    required_manual_terms=("人体工学椅", "chair"),
                )
            if any(term in query_text for term in ("parts", "components", "assembly", "组装", "部件", "配件")):
                add_if(
                    lambda chunk, combined: "配件" in combined and ("安装脚轮" in combined or "扶手" in combined or "气杆" in combined),
                    base_score=5.8,
                    required_manual_terms=("人体工学椅", "chair"),
                )

        if any(term in query_text for term in ("dishwasher", "洗碗机")):
            if any(term in query_text for term in ("spray arm", "喷淋臂")):
                add_if(
                    lambda chunk, combined: "喷淋臂" in combined and ("孔是否堵塞" in combined or "clean" in combined or "清洁" in combined),
                    base_score=5.8,
                    required_manual_terms=("洗碗机", "dishwasher"),
                )
            if any(term in query_text for term in ("parts", "components", "部件")):
                add_if(
                    lambda chunk, combined: "程序选择" in combined and ("显示屏" in combined or "启动" in combined),
                    base_score=5.4,
                    required_manual_terms=("洗碗机", "dishwasher"),
                )

        if any(term in query_text for term in ("water pump", "水泵")) and any(term in query_text for term in ("无法抽水", "does not pump", "cannot pump", "unable to pump")):
            add_if(
                lambda chunk, combined: (
                    "无法抽水" in combined
                    and ("软管接头" in combined or "卡箍" in combined or "机械密封" in combined)
                ),
                base_score=6.2,
                required_manual_terms=("水泵", "pump"),
            )

        camera_p_mode_query = (
            any(term in query_text for term in ("p mode", "p model", "program ae", "program mode"))
            or (('\\"p\\"' in query_text or '"p"' in query_text or "“p”" in query_text) and any(term in query_text for term in ("mode", "model")))
        ) or bool(
            re.search(r'["“”]?\s*p\s*["“”]?\s*(?:mode|model)', query_text)
        )
        if any(term in query_text for term in ("camera", "相机")) and camera_p_mode_query:
            add_if(
                lambda chunk, combined: (
                    ("p program ae" in combined or "program ae" in combined)
                    and ("shutter speed" in combined or "aperture" in combined or "mode dial" in combined)
                ),
                base_score=6.0,
                required_manual_terms=("单反相机", "camera"),
            )

        if "anti-block shield" in query_text or "anti block shield" in query_text:
            add_if(
                lambda chunk, combined: (
                    "anti-block shield" in combined
                    and (
                        "prevents food particles" in combined
                        or "press down until it snaps" in combined
                        or "pops off the prongs" in combined
                    )
                ),
                base_score=6.4,
                required_manual_terms=("压力锅", "pressure cooker"),
            )

        if any(term in query_text for term in ("vacuum", "吸尘器")) and any(term in query_text for term in ("robot anatomy", "vacuum anatomy", "robot parts")):
            add_if(
                lambda chunk, combined: (
                    ("clean button" in combined or "bin release" in combined or "faceplate" in combined or "handle" in combined)
                    and not any(term in combined for term in ("virtual wall", "battery and charging", "standby mode"))
                ),
                base_score=5.8,
                required_manual_terms=("吸尘器", "vacuum"),
            )

        if (
            any(term in query_text for term in ("air purifier", "空气净化器"))
            and any(term in query_text for term in ("mode", "modes", "特点", "模式", "运行", "设置"))
        ):
            add_if(
                lambda chunk, combined: (
                    ("常规运行" in combined or "空气质量" in combined or "iaq" in combined)
                    and not any(term in combined for term in ("脚轮安装", "caster"))
                ),
                base_score=5.5,
                required_manual_terms=("空气净化器", "purifier"),
            )

        if (
            any(term in query_text for term in ("steam cleaner", "蒸汽清洁机"))
            and any(term in query_text for term in ("function", "features", "功能", "快速上手", "handheld", "hard floor"))
        ):
            add_if(
                lambda chunk, combined: (
                    ("手持蒸汽器" in combined or "handheld" in combined or "产品部件介绍" in combined)
                    and not any(term in combined for term in ("保修", "warranty"))
                ),
                base_score=5.6,
                required_manual_terms=("蒸汽清洁机", "steam"),
            )

        if "fuse" in query_text and ("boat" in query_text or "jet boat" in query_text or "喷射船" in query_text):
            add_if(
                lambda chunk, combined: (
                    "fuse" in combined
                    and (
                        "fuse replacement" in combined
                        or "to replace a fuse" in combined
                        or "fuse puller" in combined
                        or "spare fuse" in combined
                    )
                ),
                base_score=4.4,
                required_manual_terms=("喷射船", "boat"),
            )

        if (
            ("battery conversion" in query_text or "conversion feature" in query_text)
            and any(term in query_text for term in ("boat", "sailing", "jet boat", "喷射船"))
        ):
            add_if(
                lambda chunk, combined: (
                    "battery switches" in combined
                    and "battery switch assembly" in combined
                    and "emerg parallel" in combined
                    and "start" in combined
                    and "house" in combined
                ),
                base_score=4.8,
                required_manual_terms=("喷射船", "boat"),
            )

        if (
            ("over temperature" in query_text or "temperature warning" in query_text or "overheat" in query_text)
            and any(term in query_text for term in ("boat", "jet boat", "喷射船"))
        ):
            add_if(
                lambda chunk, combined: (
                    "over temperature warning" in combined
                    or (
                        "over temperature" in combined
                        and "cooling water pilot outlet" in combined
                    )
                ),
                base_score=4.7,
                required_manual_terms=("喷射船", "boat"),
            )

        if "swim platform" in query_text and any(term in query_text for term in ("open", "打开")):
            add_if(
                lambda chunk, combined: (
                    "wet storage compartment" in combined
                    and "swim platform" in combined
                    and "lock handle" in combined
                    and "rear platform hatch" in combined
                ),
                base_score=4.8,
                required_manual_terms=("喷射船", "boat"),
            )

        if (
            "toothbrush" in query_text
            and "travel case" in query_text
            and any(term in query_text for term in ("charge", "charging", "charges"))
        ):
            add_if(
                lambda chunk, combined: (
                    "charging with the travel case" in combined
                    or (
                        "travel case" in combined
                        and "usb wall adapter" in combined
                        and "battery indicator" in combined
                    )
                ),
                base_score=4.8,
                required_manual_terms=("电动牙刷", "toothbrush"),
            )

        if "factory reset" in query_text and any(term in query_text for term in ("boat", "steering", "喷射船")):
            add_if(
                lambda chunk, combined: (
                    "factory reset screen" in combined
                    and "reset" in combined
                    and "button" in combined
                    and "yes" in combined
                    and "factory default settings" in combined
                ),
                base_score=40.0,
                required_manual_terms=("喷射船", "boat"),
            )

        if "indirect cooking" in query_text and "grill" in query_text:
            add_if(
                lambda chunk, combined: (
                    "indirect cooking" in combined
                    and (
                        "indirect heat" in combined
                        or "lid close" in combined
                        or "slow roasting" in combined
                    )
                ),
                base_score=4.7,
                required_manual_terms=("烧烤炉", "grill"),
            )

        if (
            "manual program" in query_text
            and any(term in query_text for term in ("channel", "channels", "communication channels"))
        ):
            add_if(
                lambda chunk, combined: (
                    "manual program" in combined
                    and "memory/erase" in combined
                    and ("memorize" in combined or "erase" in combined)
                ),
                base_score=4.7,
                required_manual_terms=("电视", "tv", "television"),
            )

        if "outdoor antenna" in query_text or ("antenna" in query_text and "reception" in query_text):
            add_if(
                lambda chunk, combined: (
                    "antenna" in combined
                    and (
                        "outdoor antenna" in combined
                        or "75 ohm antenna jack" in combined
                        or "coaxial cable" in combined
                    )
                ),
                base_score=4.6,
                required_manual_terms=("电视", "tv", "television"),
            )

        if any(term in query_text for term in ("boat", "ship", "sailing", "jet boat", "喷射船")):
            boat_terms = ("喷射船", "boat")
            if "approval label" in query_text or "emission control certificate" in query_text:
                add_if(
                    lambda chunk, combined: "approval label" in combined and "emission control" in combined,
                    base_score=5.0,
                    required_manual_terms=boat_terms,
                )
            if "engine oil level" in query_text:
                add_if(
                    lambda chunk, combined: "check the engine oil level" in combined and ("dipstick" in combined or "oil tank filler cap" in combined),
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if "battery compartment" in query_text:
                add_if(
                    lambda chunk, combined: "battery compartment" in combined and "latch" in combined and "compartment lid" in combined,
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if "anchor light" in query_text:
                add_if(
                    lambda chunk, combined: "anchor light" in combined and ("set up the anchor light" in combined or "anchor light socket" in combined or "navigation and anchor lights switch" in combined),
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if "water supply" in query_text:
                add_if(
                    lambda chunk, combined: "water supply on or off" in combined or ("shut-off valve" in combined and "inspection cover" in combined),
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if "bilge pump" in query_text:
                add_if(
                    lambda chunk, combined: "bilge pump" in combined and ("bilge pump switch" in combined or "bilge pump outlet" in combined),
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if ("turn" in query_text or "steer" in query_text or "steers" in query_text or "steering" in query_text) and any(
                term in query_text for term in ("boat", "ship", "jet boat", "sailing")
            ):
                add_if(
                    lambda chunk, combined: ("steering your boat" in combined or "jet thrust turns" in combined or "you need throttle to steer" in combined),
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if "cross wakes" in query_text or "swells" in query_text:
                add_if(
                    lambda chunk, combined: "crossing wakes and swells" in combined and ("quarter" in combined or "least jolt" in combined),
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if "flush" in query_text or "cooling system" in query_text:
                add_if(
                    lambda chunk, combined: "flushing the cooling system" in combined and ("garden hose adapter" in combined or "cooling water pilot outlet" in combined),
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if "livewell" in query_text:
                add_if(
                    lambda chunk, combined: "livewell" in combined and ("livewell switch" in combined or "livewell pump" in combined),
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if "move forward" in query_text or "forward position" in query_text:
                add_if(
                    lambda chunk, combined: "remote control levers" in combined and "forward position" in combined and "jet thrust" in combined,
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )
            if "throttle" in query_text and "cable" in query_text:
                add_if(
                    lambda chunk, combined: "throttle cable" in combined and "inner wires" in combined,
                    base_score=5.1,
                    required_manual_terms=boat_terms,
                )

        if "摩托艇" in query_text or "水上摩托" in query_text:
            jetski_terms = ("摩托艇", "watercraft")
            if any(term in query_text for term in ("碰撞", "避让", "障碍")):
                add_if(
                    lambda chunk, combined: (
                        ("避免碰撞" in combined and "安全速度" in combined)
                        or ("转向需要油门" in combined and "障碍物" in combined)
                    ),
                    base_score=5.8,
                    required_manual_terms=jetski_terms,
                )
            if any(term in query_text for term in ("停止", "停车", "停稳", "制动", "减速")):
                add_if(
                    lambda chunk, combined: (
                        "独立制动" in combined
                        or ("停车距离" in combined and "ride" in combined)
                        or ("90" in combined and "300" in combined and "停" in combined)
                    ),
                    base_score=5.8,
                    required_manual_terms=jetski_terms,
                )
            if any(term in query_text for term in ("转向", "转弯", "油门", "半滑航")):
                add_if(
                    lambda chunk, combined: (
                        ("车把" in combined and "喷射推力" in combined and "油门" in combined)
                        or ("半滑航" in combined and "转向" in combined)
                        or ("释放油门" in combined and "转向能力" in combined)
                    ),
                    base_score=5.8,
                    required_manual_terms=jetski_terms,
                )

        if "发电机" in query_text and "启动" in query_text and "无法" not in query_text:
            add_if(
                lambda chunk, combined: (
                    ("启动发动机前" in combined and "反冲启动器" in combined)
                    or ("通气旋钮" in combined and "燃油开关" in combined and "阻风门" in combined)
                ),
                base_score=5.8,
                required_manual_terms=("发电机", "generator"),
            )

        if "吹风机" in query_text and any(term in query_text for term in ("启动", "冷机", "热机")):
            add_if(
                lambda chunk, combined: "冷机启动" in combined and "热机启动" in combined and "泵油膜片" in combined,
                base_score=5.8,
                required_manual_terms=("吹风机", "blower"),
            )

        if "空气净化器" in query_text:
            if "塑料包装" in query_text or ("滤网" in query_text and any(term in query_text for term in ("包装", "取下", "拆除"))):
                add_if(
                    lambda chunk, combined: "滤网塑料包装" in combined and "睡眠" in combined and "自动" in combined,
                    base_score=5.8,
                    required_manual_terms=("空气净化器", "purifier"),
                )
            if "更换滤网" in query_text or ("滤网" in query_text and any(term in query_text for term in ("更换", "指示灯", "红色"))):
                add_if(
                    lambda chunk, combined: "更换滤网" in combined and ("6-12" in combined or "睡眠" in combined),
                    base_score=5.6,
                    required_manual_terms=("空气净化器", "purifier"),
                )

        if "洗碗机" in query_text:
            if "洗涤剂" in query_text or "detergent" in query_text:
                add_if(
                    lambda chunk, combined: "添加洗涤剂" in combined and "洗涤剂盒" in combined,
                    base_score=5.8,
                    required_manual_terms=("洗碗机", "dishwasher"),
                )
            if "洗涤块" in query_text or "tablet" in query_text:
                add_if(
                    lambda chunk, combined: "洗涤块" in combined and ("半载" in combined or "指示灯" in combined),
                    base_score=5.6,
                    required_manual_terms=("洗碗机", "dishwasher"),
                )

        if "mouse" in query_text or "蓝牙激光鼠标" in query_text:
            if "battery" in query_text or "电池" in query_text or "电量" in query_text:
                if any(term in query_text for term in ("status", "low", "电量", "耗尽")):
                    add_if(
                        lambda chunk, combined: ("琥珀色" in combined or "amber" in combined) and ("电量" in combined or "battery" in combined),
                        base_score=5.6,
                        required_manual_terms=("蓝牙激光鼠标", "mouse"),
                    )
                elif any(term in query_text for term in ("install", "insert", "安装", "装入")):
                    add_if(
                        lambda chunk, combined: "安装电池" in combined and ("aa" in combined or "正负极" in combined),
                        base_score=5.8,
                        required_manual_terms=("蓝牙激光鼠标", "mouse"),
                    )
            if "hid" in query_text or "other device" in query_text or "其他" in query_text:
                add_if(
                    lambda chunk, combined: ("other hid" in combined or "discoverable" in combined) and "bluetooth" in combined,
                    base_score=5.6,
                    required_manual_terms=("蓝牙激光鼠标", "mouse"),
                )

        if "thermostat" in query_text or "温控器" in query_text:
            if any(term in query_text for term in ("temporary", "override", "hold", "临时", "保持")):
                add_if(
                    lambda chunk, combined: ("temporary override" in combined or "permanent hold" in combined) and ("cancel" in combined or "hold" in combined),
                    base_score=5.8,
                    required_manual_terms=("温控器", "thermostat"),
                )
            if any(term in query_text for term in ("date", "time", "schedule", "日期", "时间", "日程")):
                add_if(
                    lambda chunk, combined: (
                        "设置日期" in combined
                        or "设置时间" in combined
                        or "调整程序日程" in combined
                        or ("wake" in combined and "away" in combined and "home" in combined)
                    ),
                    base_score=5.6,
                    required_manual_terms=("温控器", "thermostat"),
                )

        if ("steam cleaner" in query_text or "蒸汽清洁机" in query_text) and any(term in query_text for term in ("assembly", "assemble", "组装")):
            add_if(
                lambda chunk, combined: ("快速组装" in combined or "quick assembly" in combined) and ("手柄杆" in combined or "handle rod" in combined),
                base_score=5.8,
                required_manual_terms=("蒸汽清洁机", "steam"),
            )

        if "microwave" in query_text:
            if "control" in query_text and ("setup" in query_text or "set up" in query_text):
                add_if(lambda chunk, combined: "control set-up" in combined and "defrost weight" in combined, base_score=5.7, required_manual_terms=("微波炉", "microwave"))
            if "light timer" in query_text:
                add_if(lambda chunk, combined: "light timer" in combined and "turn on" in combined and "turn off" in combined, base_score=5.4, required_manual_terms=("微波炉", "microwave"))
            if "favorite recipe" in query_text:
                add_if(lambda chunk, combined: "favorite recipe" in combined and ("recall" in combined or "custom recipe" in combined), base_score=5.4, required_manual_terms=("微波炉", "microwave"))
            if "reheat" in query_text:
                add_if(lambda chunk, combined: "reheat" in combined and ("casserole" in combined or "dinner plate" in combined), base_score=5.4, required_manual_terms=("微波炉", "microwave"))
            if "auto defrost" in query_text:
                add_if(lambda chunk, combined: "auto defrost" in combined and ("defrost sequences" in combined or "frozen foods" in combined), base_score=5.4, required_manual_terms=("微波炉", "microwave"))
            if "oven light" in query_text:
                add_if(lambda chunk, combined: "oven light replacement" in combined and "bulb" in combined, base_score=5.4, required_manual_terms=("微波炉", "microwave"))

        if "vacuum" in query_text:
            if "two primary modes" in query_text or "two main" in query_text or "dual mode" in query_text:
                add_if(lambda chunk, combined: "dual mode virtual wall barrier" in combined and ("virtual wall mode" in combined or "halo" in combined), base_score=5.0, required_manual_terms=("吸尘器", "vacuum"))
            if "empty" in query_text and "bin" in query_text:
                add_if(lambda chunk, combined: "emptying the bin" in combined and "bin release button" in combined, base_score=5.0, required_manual_terms=("吸尘器", "vacuum"))
            if "full bin sensor" in query_text:
                add_if(lambda chunk, combined: "cleaning the full bin sensors" in combined and "wipe the sensors" in combined, base_score=5.0, required_manual_terms=("吸尘器", "vacuum"))
            if "charging contacts" in query_text:
                add_if(lambda chunk, combined: "cleaning the sensor and charging contacts" in combined and "charging contacts" in combined, base_score=5.0, required_manual_terms=("吸尘器", "vacuum"))
            if "home base" in query_text or "positioning" in query_text:
                add_if(lambda chunk, combined: "positioning the vacuum" in combined and ("1.5 feet" in combined or "4 feet" in combined), base_score=5.0, required_manual_terms=("吸尘器", "vacuum"))

        if "lawn mower" in query_text or "mower" in query_text:
            if "roll bar" in query_text:
                add_if(
                    lambda chunk, combined: (
                        "roll bar" in combined
                        and (
                            "raised and locked" in combined
                            or "lowering the roll bar" in combined
                            or "raising the roll bar" in combined
                            or "hairpin cotter" in combined
                        )
                    ),
                    base_score=5.8,
                    required_manual_terms=("割草机", "mower"),
                )
            if "rear-shock" in query_text or "rear shock" in query_text:
                add_if(lambda chunk, combined: "rear-shock assemblies" in combined and "suspension system" in combined, base_score=5.6, required_manual_terms=("割草机", "mower"))
            if "height of cut" in query_text or "electric deck lift" in query_text:
                add_if(lambda chunk, combined: "height of cut" in combined and ("deck-lift switch" in combined or "height-of-cut bracket" in combined), base_score=5.6, required_manual_terms=("割草机", "mower"))
            if "filter" in query_text and "remove" in query_text:
                add_if(lambda chunk, combined: "removing the filters" in combined and "air-cleaner" in combined, base_score=5.8, required_manual_terms=("割草机", "mower"))
            if "mower belt" in query_text or ("replace" in query_text and "belt" in query_text):
                add_if(lambda chunk, combined: "replacing the mower belt" in combined and "idler arm" in combined, base_score=5.6, required_manual_terms=("割草机", "mower"))

        if "pressure cooker" in query_text or "air fryer" in query_text:
            if "pressure cooking lid" in query_text:
                add_if(lambda chunk, combined: "pressure cooking lid" in combined and ("removing the lid" in combined or "closing the lid" in combined), base_score=5.6, required_manual_terms=("压力锅", "pressure"))
            if "condensation collector" in query_text:
                add_if(lambda chunk, combined: "condensation collector" in combined and ("grooves" in combined or "tabs" in combined), base_score=5.6, required_manual_terms=("压力锅", "pressure"))
            if "sealing ring" in query_text:
                add_if(lambda chunk, combined: "sealing ring" in combined and ("air-tight seal" in combined or "installed before using" in combined or "only one sealing ring" in combined), base_score=5.8, required_manual_terms=("压力锅", "pressure"))

        if "ereader" in query_text or "e-reader" in query_text:
            if "button" in query_text or "interfaces" in query_text or "views" in query_text:
                add_if(lambda chunk, combined: "front view" in combined and "home/esc" in combined and "navigation" in combined, base_score=5.8, required_manual_terms=("电子阅读器", "ereader"))
            if "ebook mode" in query_text:
                add_if(lambda chunk, combined: "ebook mode" in combined and "page jump" in combined, base_score=5.8, required_manual_terms=("电子阅读器", "ereader"))
            if "music" in query_text:
                add_if(lambda chunk, combined: "music mode" in combined and "audio files list" in combined, base_score=5.8, required_manual_terms=("电子阅读器", "ereader"))
            if "record" in query_text or "voice" in query_text:
                add_if(lambda chunk, combined: ("voice recording" in combined or "record" in combined) and ("play/pause" in combined or "record mode" in combined), base_score=5.8, required_manual_terms=("电子阅读器", "ereader"))
            if "video" in query_text:
                add_if(lambda chunk, combined: "video mode" in combined and ("press" in combined or "subtitle" in combined or "full screen" in combined), base_score=5.8, required_manual_terms=("电子阅读器", "ereader"))
            if "main menu" in query_text or "browser history" in query_text:
                add_if(lambda chunk, combined: "main menu" in combined and "browser history" in combined, base_score=5.8, required_manual_terms=("电子阅读器", "ereader"))

        if "snowmobile" in query_text:
            if "uphill" in query_text:
                add_if(
                    lambda chunk, combined: "riding uphill" in combined and ("uphill side" in combined or "running boards" in combined),
                    base_score=6.2,
                    required_manual_terms=("雪地摩托", "snowmobile"),
                )
            if "downhill" in query_text:
                add_if(
                    lambda chunk, combined: "riding downhill" in combined and ("engine compression" in combined or "brake" in combined),
                    base_score=6.2,
                    required_manual_terms=("雪地摩托", "snowmobile"),
                )
            if "crossing a slope" in query_text or "cross slope" in query_text or "side hill" in query_text or "sidehill" in query_text:
                add_if(
                    lambda chunk, combined: "crossing a slope" in combined and ("uphill side" in combined or "downhill knee" in combined),
                    base_score=6.2,
                    required_manual_terms=("雪地摩托", "snowmobile"),
                )
            if "throttle cable" in query_text:
                add_if(
                    lambda chunk, combined: (
                        ("throttle cable" in combined and "throttle override system" in combined)
                        or ("brake/throttle cable ends" in combined and "low-temperature grease" in combined)
                    ),
                    base_score=5.8,
                    required_manual_terms=("雪地摩托", "snowmobile"),
                )
            if "steering system" in query_text:
                add_if(lambda chunk, combined: "steering system" in combined and "free play" in combined, base_score=5.8, required_manual_terms=("雪地摩托", "snowmobile"))
            if "turn" in query_text:
                add_if(lambda chunk, combined: "turning" in combined and "handlebars" in combined and "lean" in combined, base_score=5.8, required_manual_terms=("雪地摩托", "snowmobile"))
            if "spark plug" in query_text:
                add_if(lambda chunk, combined: "spark plug inspection" in combined and ("electrode gap" in combined or "spark plug torque" in combined), base_score=5.8, required_manual_terms=("雪地摩托", "snowmobile"))

        if "widcomm" in query_text or "蓝牙激光鼠标" in query_text:
            if "卸载" in query_text or "uninstall" in query_text:
                add_if(lambda chunk, combined: "卸载 widcomm" in combined or ("添加或删除程序" in combined and "widcomm" in combined), base_score=5.0, required_manual_terms=("蓝牙激光鼠标", "mouse"))
            elif "配对" in query_text or "pair" in query_text:
                add_if(lambda chunk, combined: "widcomm 蓝牙驱动程序配对" in combined or ("hid" in combined and "鼠标连接成功" in combined), base_score=5.0, required_manual_terms=("蓝牙激光鼠标", "mouse"))
            elif "首次" in query_text or "first" in query_text:
                add_if(lambda chunk, combined: "widcomm 蓝牙驱动程序使用" in combined and "首次使用" in combined, base_score=5.0, required_manual_terms=("蓝牙激光鼠标", "mouse"))
            else:
                add_if(lambda chunk, combined: "安装 widcomm" in combined or ("setup.exe" in combined and "widcomm" in combined), base_score=5.0, required_manual_terms=("蓝牙激光鼠标", "mouse"))

        if "toothbrush" in query_text and "intensity" in query_text:
            add_if(
                lambda chunk, combined: "intensity settings" in combined and "three different intensity settings" in combined,
                base_score=5.8,
                required_manual_terms=("电动牙刷", "toothbrush"),
            )

        if "motherboard" in query_text:
            if "pci express" in query_text and "x16" in query_text:
                add_if(
                    lambda chunk, combined: (
                        "pci express" in combined
                        and "slots" in combined
                        and ("graphic" in combined or "graphics card" in combined or "vga card" in combined)
                    ),
                    base_score=5.2,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "jumper" in query_text:
                add_if(
                    lambda chunk, combined: "jumper" in combined and ("clear rtc ram" in combined or "cpu over voltage" in combined),
                    base_score=4.8,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "rear panel connector" in query_text:
                add_if(
                    lambda chunk, combined: "rear panel connectors" in combined and ("ps/2" in combined or "lan" in combined or "usb" in combined),
                    base_score=4.8,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "onboard led" in query_text:
                add_if(
                    lambda chunk, combined: "onboard led" in combined and "standby power led" in combined,
                    base_score=5.2,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "bios" in query_text and ("update" in query_text or "file" in query_text):
                add_if(
                    lambda chunk, combined: (
                        ("update the bios" in combined or "updating the bios file" in combined or "ez flash" in combined)
                        and ("usb" in combined or "internet" in combined or "bios updater" in combined)
                    ),
                    base_score=4.8,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "raid" in query_text:
                add_if(
                    lambda chunk, combined: "raid" in combined and ("create a raid" in combined or "raid setup" in combined or "sata mode" in combined),
                    base_score=4.8,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "sata odd" in query_text or ("usb devices" in query_text and "operating system" in query_text):
                add_if(
                    lambda chunk, combined: (
                        ("sata odd" in combined and "usb" in combined and "install" in combined)
                        or ("installing an operating system" in combined and "usb" in combined)
                    ),
                    base_score=5.2,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "secure" in query_text and "chassis" in query_text:
                add_if(
                    lambda chunk, combined: "secure the motherboard to the chassis" in combined or ("screw holes" in combined and "chassis" in combined),
                    base_score=5.2,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "central processing unit" in query_text or re.search(r"\bcpu\b", query_text):
                add_if(
                    lambda chunk, combined: "central processing unit" in combined and ("lga1151" in combined or "cpu socket" in combined),
                    base_score=4.8,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "system memory" in query_text:
                add_if(
                    lambda chunk, combined: "recommended memory configurations" in combined or ("system memory" in combined and "channel" in combined),
                    base_score=5.2,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "serial port connector" in query_text:
                add_if(
                    lambda chunk, combined: "serial port connector" in combined and "10-1 pin" in combined,
                    base_score=4.8,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "tpm connector" in query_text:
                add_if(
                    lambda chunk, combined: "tpm connector" in combined and "14-1 pin" in combined,
                    base_score=5.2,
                    required_manual_terms=("主板", "motherboard"),
                )
            if "thermal sensor connector" in query_text or "t_sensor" in query_text:
                add_if(
                    lambda chunk, combined: ("thermal sensor connector" in combined or "t_sensor" in combined) and "thermistor cable" in combined,
                    base_score=6.0,
                    required_manual_terms=("主板", "motherboard"),
                )

        if "fax" in query_text and any(term in query_text for term in ("connect", "connecting", "procedure", "连接")):
            add_if(
                lambda chunk, combined: (
                    (
                        "telephone wall jack" in combined
                        or "telephone line cord" in combined
                        or "telecommunication line cord" in combined
                        or "standard modular jack" in combined
                        or "modular plug" in combined
                        or "usoc rj11" in combined
                    )
                    and ("ren" not in combined[:180])
                ),
                base_score=4.0,
                required_manual_terms=("传真", "fax"),
            )

        if "base station" in query_text and any(term in query_text for term in ("connect", "connecting", "连接")):
            add_if(
                lambda chunk, combined: (
                    "base station" in combined
                    and (
                        "connect each end" in combined
                        or ("dc input jack" in combined and "telephone socket" in combined)
                    )
                    and "charging contacts" not in combined
                    and "docking tone" not in combined
                ),
                base_score=4.6,
                required_manual_terms=("固定电话", "landline", "telephone"),
            )

        if ("quick release" in query_text or "qpr" in query_text) and "pressure cooker" in query_text:
            add_if(
                lambda chunk, combined: "quick release" in combined and ("quick release button" in combined or "vent position" in combined),
                base_score=4.1,
                required_manual_terms=("pressure cooker", "压力锅"),
            )

        if ("natural release" in query_text or "npr" in query_text or re.search(r"\bnr\b", query_text)) and "pressure cooker" in query_text:
            add_if(
                lambda chunk, combined: (
                    "natural release" in combined
                    and (
                        "depressurizes naturally" in combined
                        or "de pressurizes naturally" in combined
                        or "temperature within the cooker drops" in combined
                        or "cooker drops" in combined
                    )
                ),
                base_score=4.6,
                required_manual_terms=("pressure cooker", "压力锅"),
            )

        if "float valve" in query_text and "pressure cooker" in query_text:
            add_if(
                lambda chunk, combined: "float valve" in combined and ("silicone cap" in combined or "lid" in combined),
                base_score=4.1,
                required_manual_terms=("pressure cooker", "压力锅"),
            )

        anchors.sort(key=lambda item: (-item.score, item.chunk.order))
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for result in anchors:
            if result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            deduped.append(result)
        return deduped[:8]

    def _looks_parameter_query(self, query_variants: dict[str, str]) -> bool:
        combined = " ".join(query_variants.values()).lower()
        return any(term in combined for term in PARAMETER_TERMS)

    def _has_explicit_numeric_answer(self, text: str) -> bool:
        return bool(NUMERIC_ANSWER_RE.search(text))

    def _is_generic_heading(self, title: str) -> bool:
        compact = re.sub(r"\s+", "", title.strip())
        return bool(GENERIC_TITLE_RE.search(compact))

    def _parameter_alignment_score(self, matched_terms: set[str], query_variants: dict[str, str]) -> float:
        informative_terms: set[str] = set()
        for query in query_variants.values():
            for token in tokenize(query):
                if token in PARAMETER_TERMS or len(token) >= 2 and any(term in token for term in ("载重", "重量", "容量", "功率", "电压", "温度", "速度")):
                    informative_terms.add(token)
        if not informative_terms:
            return 0.0
        overlap = informative_terms & matched_terms
        if not overlap:
            return 0.0
        return min(0.22, 0.08 * len(overlap))

    def _informative_match_bonus(self, matched_terms: set[str], query_variants: dict[str, str]) -> float:
        if not matched_terms:
            return 0.0
        informative = {term for term in matched_terms if len(term) >= 2}
        bonus = min(0.24, 0.035 * len(informative))
        query_text = " ".join(query_variants.values()).lower()
        action_terms = {term.lower() for term in ACTION_HINT_TERMS}
        if any(term in query_text for term in action_terms):
            action_overlap = {term.lower() for term in informative} & action_terms
            if action_overlap:
                bonus += min(0.36, 0.12 * len(action_overlap))
            else:
                bonus -= 0.12
        return bonus

    def _parameter_answer_bonus(
        self,
        chunk,
        bm25_score: float,
        matched_terms: set[str],
        query_variants: dict[str, str],
        boost_manuals: list[str],
    ) -> float:
        combined_text = f"{chunk.section_title} {chunk.text}"
        bonus = 0.0
        explicit_answer = self._has_explicit_numeric_answer(combined_text)

        if explicit_answer:
            bonus += 0.34
            if bm25_score >= 40:
                bonus += 0.18
            if chunk.chunk_type == "title_only":
                bonus += 0.14
            if boost_manuals and chunk.manual_name in boost_manuals:
                bonus += 0.12
        else:
            if chunk.chunk_type == "title_only":
                bonus -= 0.18
            if self._is_generic_heading(chunk.section_title):
                bonus -= 0.20

        bonus += self._parameter_alignment_score(matched_terms, query_variants)
        return bonus

    def _supports_sidecar(self, boost_manuals: list[str]) -> bool:
        return any(manual_name in SIDECAR_SUPPORTED_MANUALS for manual_name in boost_manuals)

    def _has_sidecar_trigger(self, query: str, rewritten) -> bool:
        parts = [
            query,
            getattr(rewritten, "query_en", "") if rewritten is not None else "",
            getattr(rewritten, "query_zh", "") if rewritten is not None else "",
            getattr(rewritten, "query_mix", "") if rewritten is not None else "",
            " ".join(getattr(rewritten, "search_keywords", []) or []) if rewritten is not None else "",
        ]
        combined = " ".join(part.lower() for part in parts if part)
        if any(term in combined for term in SIDECAR_TRIGGER_TERMS):
            return True
        return any(term in combined for term in PARAMETER_TERMS)

    def _query_action_terms(self, query: str, rewritten) -> set[str]:
        parts = [
            query,
            getattr(rewritten, "query_en", "") if rewritten is not None else "",
            getattr(rewritten, "query_zh", "") if rewritten is not None else "",
            getattr(rewritten, "query_mix", "") if rewritten is not None else "",
            " ".join(getattr(rewritten, "search_keywords", []) or []) if rewritten is not None else "",
        ]
        combined = " ".join(part for part in parts if part).lower()
        terms = {term.lower() for term in ACTION_HINT_TERMS if term.lower() in combined}
        terms.update({phrase.lower() for phrase in CRITICAL_ACTION_PHRASES if phrase.lower() in combined})
        return terms

    def _result_matches_query_actions(self, result: SearchResult, action_terms: set[str]) -> bool:
        if not action_terms:
            return True
        combined = f"{result.chunk.section_title} {result.chunk.text}".lower()
        return any(term.lower() in combined for term in action_terms)

    def _query_delete_intent(self, query: str, rewritten) -> bool:
        parts = [
            query,
            getattr(rewritten, "query_en", "") if rewritten is not None else "",
            getattr(rewritten, "query_zh", "") if rewritten is not None else "",
            getattr(rewritten, "query_mix", "") if rewritten is not None else "",
            " ".join(getattr(rewritten, "search_keywords", []) or []) if rewritten is not None else "",
        ]
        combined = " ".join(part.lower() for part in parts if part)
        return (
            ("delete" in combined or "erase" in combined or "删除" in combined)
            and ("image" in combined or "images" in combined or "图像" in combined)
        )

    def _is_delete_answer_chunk(self, result: SearchResult) -> bool:
        combined = f"{result.chunk.section_title} {result.chunk.text}".lower()
        has_delete = any(term in combined for term in ("删除", "delete", "erase"))
        has_scope = any(term in combined for term in ("全部", "单张", "all", "single"))
        return has_delete and has_scope

    def _is_storage_card_note_chunk(self, result: SearchResult) -> bool:
        title = result.chunk.section_title
        combined = f"{result.chunk.section_title} {result.chunk.text}"
        if "在电脑上使用存储卡注意事项" in title:
            return True
        return "存储卡注意事项" in title and any(term in combined for term in ("电脑", "文件夹", "格式化"))

    def _prioritize_delete_evidence(
        self,
        results: list[SearchResult],
        query: str,
        rewritten,
    ) -> list[SearchResult]:
        if not results or not self._query_delete_intent(query, rewritten):
            return results

        answer_chunks = [result for result in results if self._is_delete_answer_chunk(result)]
        if not answer_chunks:
            return results

        neutral_chunks = [
            result
            for result in results
            if result not in answer_chunks and not self._is_storage_card_note_chunk(result)
        ]
        note_chunks = [result for result in results if self._is_storage_card_note_chunk(result)]

        ordered = answer_chunks + neutral_chunks + note_chunks
        anchor = max(result.score for result in ordered)
        rescored: list[SearchResult] = []
        for index, result in enumerate(ordered, start=1):
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=max(0.0, anchor - 0.03 * (index - 1)),
                    bm25_score=result.bm25_score,
                    semantic_score=result.semantic_score,
                    matched_terms=result.matched_terms,
                )
            )
        return rescored

    def _query_needs_generation_context(self, query: str, rewritten) -> bool:
        if self._query_delete_intent(query, rewritten):
            return True
        if self._looks_parameter_query(self._query_variants(query, rewritten) if rewritten is not None else {"query": query}):
            return True
        combined = " ".join(
            part.lower()
            for part in (
                query,
                getattr(rewritten, "query_en", "") if rewritten is not None else "",
                getattr(rewritten, "query_zh", "") if rewritten is not None else "",
                getattr(rewritten, "query_mix", "") if rewritten is not None else "",
            )
            if part
        )
        if any(term in combined for term in ("indicator", "light", "blink", "flashing", "meaning", "functions", "features", "mode", "modes", "指示灯", "闪烁", "含义", "功能", "模式")):
            return True
        return bool(self._query_action_terms(query, rewritten) or self._slot_aliases_from_query({"query": combined}))

    def _neighbor_is_useful_for_generation(
        self,
        base_result: SearchResult,
        neighbor,
        action_terms: set[str],
        slot_aliases: list[tuple[str, ...]],
    ) -> bool:
        if neighbor.chunk_id == base_result.chunk.chunk_id:
            return False
        if neighbor.manual_name != base_result.chunk.manual_name:
            return False
        if neighbor.chunk_type in {"toc"}:
            return False

        combined = f"{neighbor.section_title} {neighbor.text}".lower()
        base_title = re.sub(r"\s+", "", base_result.chunk.section_title)
        neighbor_title = re.sub(r"\s+", "", neighbor.section_title)
        if slot_aliases and any(any(alias in combined for alias in aliases) for aliases in slot_aliases):
            return True
        if action_terms and any(term.lower() in combined for term in action_terms):
            return True
        if neighbor.image_ids:
            return True
        if base_title and neighbor_title and (base_title in neighbor_title or neighbor_title in base_title):
            return True
        if neighbor.chunk_type in {"step", "menu", "list", "component"}:
            return True
        if len(neighbor.text) >= 40 and not self._is_generic_heading(neighbor.section_title):
            return True
        return False

    def _expand_results_for_generation(
        self,
        results: list[SearchResult],
        query: str,
        rewritten,
    ) -> list[SearchResult]:
        if not results or not self._query_needs_generation_context(query, rewritten):
            return results

        action_terms = self._query_action_terms(query, rewritten)
        query_variants = self._query_variants(query, rewritten) if rewritten is not None else {"query": query}
        slot_aliases = self._slot_aliases_from_query(query_variants)
        combined_query = " ".join(query_variants.values()).lower()
        broad_context = any(
            term in combined_query
            for term in ("indicator", "light", "blink", "flashing", "functions", "features", "mode", "modes", "指示灯", "闪烁", "功能", "模式")
        )
        expanded: list[SearchResult] = []
        seen: set[str] = set()

        def add_result(result: SearchResult) -> None:
            if result.chunk.chunk_id in seen:
                return
            seen.add(result.chunk.chunk_id)
            expanded.append(result)

        for rank, result in enumerate(results):
            add_result(result)
            if rank >= 4:
                continue
            after_window = 5 if broad_context else (3 if result.chunk.chunk_type in {"step", "general", "component", "title_only"} else 1)
            before_window = 2 if broad_context else 1
            neighbors = self.repository.get_adjacent_chunks(
                result.chunk,
                before=before_window,
                after=after_window,
            )
            for neighbor in neighbors:
                if not self._neighbor_is_useful_for_generation(result, neighbor, action_terms, slot_aliases):
                    continue
                distance = abs(neighbor.order - result.chunk.order)
                add_result(
                    SearchResult(
                        chunk=neighbor,
                        score=max(0.0, result.score - 0.04 - 0.02 * distance),
                        bm25_score=result.bm25_score,
                        semantic_score=result.semantic_score,
                        matched_terms=result.matched_terms,
                    )
                )

        return expanded[: max(self.settings.default_top_k + 5, 8)]

    def _should_backfill_with_sidecar(
        self,
        query: str,
        rewritten,
        boost_manuals: list[str],
        query_variants: dict[str, str],
        fused_results: list[SearchResult],
    ) -> bool:
        if not self._supports_sidecar(boost_manuals):
            return False
        if not self._has_sidecar_trigger(query, rewritten):
            return False
        if not fused_results:
            return True

        top_result = fused_results[0]
        if top_result.chunk.manual_name not in boost_manuals:
            return True
        if self._is_generic_gap_chunk(top_result):
            return True
        if self._is_generic_heading(top_result.chunk.section_title):
            return True
        if self._looks_parameter_query(query_variants):
            combined = f"{top_result.chunk.section_title} {top_result.chunk.text}"
            if not self._has_explicit_numeric_answer(combined):
                return True

        action_terms = self._query_action_terms(query, rewritten)
        if action_terms and not self._result_matches_query_actions(top_result, action_terms):
            return True
        if self._query_delete_intent(query, rewritten):
            combined = f"{top_result.chunk.section_title} {top_result.chunk.text}"
            if not any(term in combined for term in ("删除", "全部", "单张", "delete", "all", "single")):
                return True
        return False

    def _run_retrieval(
        self,
        retriever: HybridRetriever,
        sidecar_retriever: SidecarHybridRetriever | None,
        dense_retriever: DenseEmbeddingRetriever | None,
        query: str,
        rewritten,
        boost_manuals: list[str],
        allowed_manuals: list[str] | None = None,
        root_query: str = "",
    ) -> tuple[
        list[SearchResult],
        dict[str, str],
        dict[str, list[SearchResult]],
        dict[str, list[SearchResult]],
        dict[str, list[SearchResult]],
        list[SearchResult],
    ]:
        english_dominant = looks_english_dominant(query)
        if rewritten is None:
            candidate_results = retriever.score_query(
                query,
                boost_manuals=boost_manuals,
                allowed_manuals=allowed_manuals,
            )[: max(self.settings.default_top_k * 3, 8)]
            variants = {"query_en": query}
            slot_query = self._slot_expanded_query(variants)
            if slot_query and slot_query != query:
                variants["query_slot"] = slot_query
            bm25_routes = {
                "query_en": retriever.search_bm25(
                    query,
                    top_k=self.settings.default_top_k,
                    boost_manuals=boost_manuals,
                    allowed_manuals=allowed_manuals,
                )
            }
            if "query_slot" in variants:
                bm25_routes["query_slot"] = retriever.search_bm25(
                    variants["query_slot"],
                    top_k=max(self.settings.default_top_k, 5),
                    boost_manuals=boost_manuals,
                    allowed_manuals=allowed_manuals,
                )
            vector_routes: dict[str, list[SearchResult]] = {}
            hybrid_routes = {"query_en": candidate_results}
            if "query_slot" in variants:
                hybrid_routes["query_slot"] = retriever.score_query(
                    variants["query_slot"],
                    boost_manuals=boost_manuals,
                    allowed_manuals=allowed_manuals,
                )[: max(self.settings.default_top_k * 3, 8)]
            title_results = self._keyword_title_results(retriever, rewritten, boost_manuals, variants, allowed_manuals)
            if title_results:
                hybrid_routes["keyword_title"] = title_results
            anchor_results = self._intent_anchor_results(retriever, rewritten, boost_manuals, variants, root_query, allowed_manuals)
            if anchor_results:
                hybrid_routes["intent_anchor"] = anchor_results
            if english_dominant:
                dense_results = self._dense_route_results(dense_retriever, query, boost_manuals, allowed_manuals)
                if dense_results:
                    vector_routes["query_en_dense"] = dense_results
            fused = self._fuse_route_results(
                {**hybrid_routes, **vector_routes},
                variants,
                boost_manuals,
                self._fusion_candidate_limit(),
            )
            fused = self._slot_coverage_adjusted_results(fused, variants, boost_manuals)
            if (
                sidecar_retriever is not None
                and english_dominant
                and self._should_backfill_with_sidecar(
                    query,
                    rewritten,
                    boost_manuals,
                    variants,
                    fused,
                )
            ):
                bm25_routes["query_en_sidecar"] = sidecar_retriever.search_bm25(
                    query,
                    top_k=max(self.settings.default_top_k, 5),
                    boost_manuals=boost_manuals,
                    allowed_manuals=allowed_manuals,
                )
                hybrid_routes["query_en_sidecar"] = sidecar_retriever.score_query(
                    query,
                    boost_manuals=boost_manuals,
                    allowed_manuals=allowed_manuals,
                )[: max(self.settings.default_top_k * 3, 8)]
                fused = self._fuse_route_results(
                    {**hybrid_routes, **vector_routes},
                    variants,
                    boost_manuals,
                    self._fusion_candidate_limit(),
                )
                fused = self._slot_coverage_adjusted_results(fused, variants, boost_manuals)
            fused, variants, rerank_boost_manuals, planner_allowed_manuals, planner_plan = self._apply_llm_query_planner(
                retriever,
                query,
                rewritten,
                variants,
                bm25_routes,
                hybrid_routes,
                vector_routes,
                fused,
                boost_manuals,
                allowed_manuals,
            )
            rerank_query = self._planner_rerank_query(query, variants, planner_plan)
            model_reranked = self._model_rerank_candidates(
                rerank_query,
                fused,
                rerank_boost_manuals,
                self.settings.default_top_k,
            )
            reranked = retriever.rerank_candidates(
                model_reranked if model_reranked else fused,
                query_variants=variants,
                boost_manuals=rerank_boost_manuals,
                search_keywords=[],
                top_k=self.settings.default_top_k,
            )
            reranked = self._slot_coverage_adjusted_results(reranked, variants, rerank_boost_manuals)
            reranked = self._validate_evidence_with_plan(reranked, planner_plan, planner_allowed_manuals)
            reranked = self._prioritize_delete_evidence(reranked, query, rewritten)
            return reranked, variants, bm25_routes, vector_routes, hybrid_routes, fused

        variants = self._query_variants(query, rewritten)
        bm25_routes: dict[str, list[SearchResult]] = {}
        vector_routes: dict[str, list[SearchResult]] = {}
        hybrid_routes: dict[str, list[SearchResult]] = {}
        for route_name, route_query in variants.items():
            bm25_routes[route_name] = retriever.search_bm25(
                route_query,
                top_k=max(self.settings.default_top_k, 5),
                boost_manuals=boost_manuals,
                allowed_manuals=allowed_manuals,
            )
            hybrid_routes[route_name] = retriever.score_query(
                route_query,
                boost_manuals=boost_manuals,
                allowed_manuals=allowed_manuals,
            )[: max(self.settings.default_top_k * 3, 8)]
        title_results = self._keyword_title_results(retriever, rewritten, boost_manuals, variants, allowed_manuals)
        if title_results:
            hybrid_routes["keyword_title"] = title_results
        anchor_results = self._intent_anchor_results(retriever, rewritten, boost_manuals, variants, root_query, allowed_manuals)
        if anchor_results:
            hybrid_routes["intent_anchor"] = anchor_results
        if english_dominant and "query_en" in variants:
            dense_results = self._dense_route_results(dense_retriever, variants["query_en"], boost_manuals, allowed_manuals)
            if dense_results:
                vector_routes["query_en_dense"] = dense_results
        fused = self._fuse_route_results(
            {**hybrid_routes, **vector_routes},
            variants,
            boost_manuals,
            self._fusion_candidate_limit(),
        )
        fused = self._slot_coverage_adjusted_results(fused, variants, boost_manuals)
        if (
            sidecar_retriever is not None
            and english_dominant
            and "query_en" in variants
            and self._should_backfill_with_sidecar(
                query,
                rewritten,
                boost_manuals,
                variants,
                fused,
            )
        ):
            bm25_routes["query_en_sidecar"] = sidecar_retriever.search_bm25(
                variants["query_en"],
                top_k=max(self.settings.default_top_k, 5),
                boost_manuals=boost_manuals,
                allowed_manuals=allowed_manuals,
            )
            hybrid_routes["query_en_sidecar"] = sidecar_retriever.score_query(
                variants["query_en"],
                boost_manuals=boost_manuals,
                allowed_manuals=allowed_manuals,
            )[: max(self.settings.default_top_k * 3, 8)]
            fused = self._fuse_route_results(
                {**hybrid_routes, **vector_routes},
                variants,
                boost_manuals,
                self._fusion_candidate_limit(),
            )
            fused = self._slot_coverage_adjusted_results(fused, variants, boost_manuals)
        fused, variants, rerank_boost_manuals, planner_allowed_manuals, planner_plan = self._apply_llm_query_planner(
            retriever,
            query,
            rewritten,
            variants,
            bm25_routes,
            hybrid_routes,
            vector_routes,
            fused,
            boost_manuals,
            allowed_manuals,
        )
        rerank_query = self._planner_rerank_query(variants.get("query_en") or query, variants, planner_plan)
        model_reranked = self._model_rerank_candidates(
            rerank_query,
            fused,
            rerank_boost_manuals,
            self.settings.default_top_k,
        )
        reranked = retriever.rerank_candidates(
            model_reranked if model_reranked else fused,
            query_variants=variants,
            boost_manuals=rerank_boost_manuals,
            search_keywords=getattr(rewritten, "search_keywords", []),
            top_k=self.settings.default_top_k,
        )
        reranked = self._slot_coverage_adjusted_results(reranked, variants, rerank_boost_manuals)
        reranked = self._validate_evidence_with_plan(reranked, planner_plan, planner_allowed_manuals)
        reranked = self._prioritize_delete_evidence(reranked, query, rewritten)
        return reranked, variants, bm25_routes, vector_routes, hybrid_routes, fused

    def _build_query_trace(
        self,
        original_query: str,
        rewritten,
    ) -> dict:
        return {
            "original_query": original_query,
            "query_en": getattr(rewritten, "query_en", original_query),
            "query_zh": getattr(rewritten, "query_zh", None),
            "query_mix": getattr(rewritten, "query_mix", original_query),
            "search_keywords": getattr(rewritten, "search_keywords", []),
            "product_hint": getattr(rewritten, "product_hint", None),
            "intent": getattr(rewritten, "intent", None),
            "need_image": getattr(rewritten, "need_image", False),
            "planned_sub_questions": getattr(rewritten, "planned_sub_questions", []),
        }

    def _safe_planned_sub_questions(self, original_question: str, question_plan) -> list[str]:
        planned = getattr(question_plan, "planned_sub_questions", []) or []
        if not planned:
            return []
        required_terms = [
            term
            for term in CRITICAL_ACTION_PHRASES
            if term.lower() in original_question.lower() or term in original_question
        ]
        if not required_terms:
            return planned
        planned_text = " ".join(planned).lower()
        if any(term.lower() in planned_text for term in required_terms):
            return planned
        return []

    def _summarize_result(self, result) -> dict:
        chunk = result.chunk
        text = chunk.text.strip().replace("\n", " ")
        if len(text) > 180:
            text = text[:177].rstrip("，,；; ") + "..."
        return {
            "chunk_id": chunk.chunk_id,
            "manual_name": chunk.manual_name,
            "section_title": chunk.section_title,
            "chunk_type": chunk.chunk_type,
            "score": round(result.score, 4),
            "bm25_score": round(result.bm25_score, 4),
            "semantic_score": round(result.semantic_score, 4),
            "matched_terms": result.matched_terms,
            "text_summary": text,
        }

    def _diagnose_refusal(self, answer: str, sub_question_results: list[tuple[str, list]]) -> dict:
        if (
            "订单、售后或平台政策咨询" in answer
            or "人工客服核实订单" in answer
            or "需要结合订单状态和平台售后规则确认" in answer
            or ("订单号" in answer and "人工客服" in answer)
        ):
            return {"is_refusal": True, "reason": "support_fallback"}
        normalized_answer = re.sub(r"\s+", " ", answer).strip()
        if not any(indicator in normalized_answer for indicator in INSUFFICIENT_INDICATORS):
            return {"is_refusal": False, "reason": "none"}
        if not any(results for _, results in sub_question_results):
            return {"is_refusal": True, "reason": "no_results"}
        top_scores = [results[0].score for _, results in sub_question_results if results]
        best_score = max(top_scores) if top_scores else 0.0
        if best_score < 0.12:
            return {"is_refusal": True, "reason": "weak_retrieval"}
        return {"is_refusal": True, "reason": "insufficient_after_generation"}

    def _is_generic_gap_chunk(self, result: SearchResult) -> bool:
        chunk = result.chunk
        combined_text = f"{chunk.section_title} {chunk.text}"
        if self._has_explicit_numeric_answer(combined_text):
            return False
        if any(phrase in combined_text for phrase in CRITICAL_ACTION_PHRASES):
            return False
        if chunk.chunk_type in {"toc", "warning", "note"}:
            return True
        if chunk.chunk_type == "title_only":
            return True
        title = re.sub(r"\s+", "", chunk.section_title.strip())
        if GENERIC_GAP_SECTION_RE.search(title):
            return True
        if chunk.chunk_type in {"general", "list"} and len(result.matched_terms) <= 2:
            return True
        return False

    def _diagnose_coverage_gap(
        self,
        refusal: dict,
        effective_product_hint: str | None,
        query_traces: list[dict],
        sub_question_results: list[tuple[str, list[SearchResult]]],
    ) -> dict:
        if not refusal.get("is_refusal"):
            return {"status": "none", "reason": "answered"}
        if refusal.get("reason") == "support_fallback":
            return {"status": "support_fallback", "reason": "support_or_policy_question"}

        expected_manuals: list[str] = []
        seen_manuals: set[str] = set()
        for product_hint in [effective_product_hint, *[trace.get("product_hint") for trace in query_traces]]:
            if not product_hint:
                continue
            for manual_name in self._resolve_manual_hints_from_product(product_hint):
                if manual_name not in seen_manuals:
                    seen_manuals.add(manual_name)
                    expected_manuals.append(manual_name)

        if effective_product_hint and not expected_manuals:
            return {
                "status": "product_not_in_kb",
                "reason": f"no_manual_matches_product_hint:{effective_product_hint}",
            }

        critical_phrases = self._critical_action_phrases(query_traces)
        if expected_manuals and critical_phrases:
            for phrase in critical_phrases:
                phrase_in_manual = self._manuals_contain_phrase(expected_manuals, phrase)
                phrase_in_results = self._results_contain_phrase(
                    [result for _, results in sub_question_results for result in results[:3]],
                    phrase,
                )
                if not phrase_in_manual:
                    return {
                        "status": "knowledge_coverage_gap",
                        "reason": f"expected_manual_missing_action_phrase:{phrase}",
                    }
                if not phrase_in_results:
                    return {
                        "status": "retrieval_failure",
                        "reason": f"expected_manual_has_action_phrase_but_not_retrieved:{phrase}",
                    }

        all_results = [result for _, results in sub_question_results for result in results[:3]]
        if not all_results:
            if expected_manuals:
                return {"status": "retrieval_failure", "reason": "no_results_for_expected_manual"}
            return {"status": "knowledge_coverage_gap", "reason": "no_results_and_no_product_anchor"}

        if expected_manuals:
            expected_hits = [result for result in all_results if result.chunk.manual_name in expected_manuals]
            if not expected_hits:
                return {"status": "retrieval_failure", "reason": "expected_manual_not_retrieved"}
            if all(self._is_generic_gap_chunk(result) for result in expected_hits[:2]):
                return {
                    "status": "knowledge_coverage_gap",
                    "reason": "expected_manual_retrieved_but_only_generic_chunks",
                }
            if max(result.score for result in expected_hits) < 0.18:
                return {"status": "retrieval_failure", "reason": "expected_manual_scores_too_low"}

        top_manuals = {result.chunk.manual_name for result in all_results[:3]}
        if len(top_manuals) > 1:
            return {"status": "retrieval_failure", "reason": "mixed_manuals_without_clear_answer"}

        if all(self._is_generic_gap_chunk(result) for result in all_results[:2]):
            return {"status": "knowledge_coverage_gap", "reason": "only_generic_chunks_available"}

        return {"status": "retrieval_failure", "reason": "retrieved_chunks_do_not_ground_answer"}

    def _critical_action_phrases(self, query_traces: list[dict]) -> list[str]:
        combined_parts: list[str] = []
        for trace in query_traces:
            combined_parts.extend(
                [
                    trace.get("query_en", "") or "",
                    trace.get("query_zh", "") or "",
                    trace.get("query_mix", "") or "",
                    " ".join(trace.get("search_keywords", []) or []),
                ]
            )
        combined = " ".join(combined_parts)
        combined_lower = combined.lower()
        phrases: list[str] = [phrase for phrase in CRITICAL_ACTION_PHRASES if phrase in combined]
        for aliases, canonical in QUERY_PHRASE_HINTS:
            if any(alias.lower() in combined_lower for alias in aliases):
                phrases.append(canonical)
        return list(dict.fromkeys(phrases))

    def _phrase_aliases(self, phrase: str) -> tuple[str, ...]:
        return PHRASE_ALIASES.get(phrase, (phrase,))

    def _manuals_contain_phrase(self, manual_names: list[str], phrase: str) -> bool:
        aliases = tuple(alias.lower() for alias in self._phrase_aliases(phrase))
        for chunk in self.repository.get_chunks():
            if chunk.manual_name not in manual_names:
                continue
            combined = f"{chunk.section_title} {chunk.text}".lower()
            if any(alias in combined for alias in aliases):
                return True
        return False

    def _results_contain_phrase(self, results: list[SearchResult], phrase: str) -> bool:
        aliases = tuple(alias.lower() for alias in self._phrase_aliases(phrase))
        for result in results:
            combined = f"{result.chunk.section_title} {result.chunk.text}".lower()
            if any(alias in combined for alias in aliases):
                return True
        return False

    def chat(
        self,
        question: str,
        images: list[str] | None = None,
        session_id: str | None = None,
    ) -> ChatReply:
        normalized_question = normalize_question(question)
        if not normalized_question:
            raise ValueError("question is required")

        payload_images = images or []
        if len(payload_images) > 3:
            raise ValueError("images supports up to 3 items")

        session_id = session_id or f"kf_session_{uuid4().hex[:12]}"
        if self._agent_graph_enabled():
            state = self._get_agent_graph().run(
                question=normalized_question,
                session_id=session_id,
                images=payload_images,
            )
            return self._chat_reply_from_agent_state(state)

        context = self.memory.get(session_id)
        insight = self.multimodal.analyze(
            normalized_question,
            payload_images,
            context,
            self.repository.get_alias_lookup(),
        )
        if is_general_support_question(normalized_question) or is_manual_access_question(normalized_question):
            generated = self.guardrail.review(self.generator.build_support_fallback(normalized_question))
            timestamp = int(time.time())
            self.memory.append_turn(
                session_id=session_id,
                question=normalized_question,
                answer=generated.answer,
                manuals=[],
                product_name=insight.product_hint,
                section_titles=[],
            )
            return ChatReply(
                answer=generated.answer,
                session_id=session_id,
                timestamp=timestamp,
                references=[],
                related_images=[],
            )

        retriever = self._get_retriever()
        sidecar_retriever = self._get_sidecar_retriever()
        question_plan = self.query_rewriter.rewrite(normalized_question, insight.product_hint) if self._llm_query_plan_enabled() else None
        planned_sub_questions = self._safe_planned_sub_questions(normalized_question, question_plan)
        sub_questions = planned_sub_questions or split_sub_questions(normalized_question) or [normalized_question]
        sub_question_results: list[tuple[str, list]] = []
        boost_manuals = insight.manual_hints or context.last_manuals

        effective_product_hint = insight.product_hint or getattr(question_plan, "product_hint", None)
        if getattr(question_plan, "product_hint", None):
            for manual_name in self._resolve_manual_hints_from_product(question_plan.product_hint):
                if manual_name not in boost_manuals:
                    boost_manuals.append(manual_name)

        for sub_question in sub_questions:
            query = rewrite_with_context(sub_question, effective_product_hint, context.last_sections)
            rewritten = (
                self.query_rewriter.rewrite(query, effective_product_hint)
                if (looks_english_dominant(query) or self._llm_query_plan_enabled())
                else None
            )
            local_boost_manuals = list(boost_manuals)
            inferred_product_hint = self._infer_product_hint_from_text(
                [
                    query,
                    getattr(rewritten, "query_en", ""),
                    getattr(rewritten, "query_zh", ""),
                    getattr(rewritten, "query_mix", ""),
                ]
            )
            if rewritten and rewritten.product_hint:
                if effective_product_hint is None:
                    effective_product_hint = rewritten.product_hint
                for manual_name in self._resolve_manual_hints_from_product(rewritten.product_hint):
                    if manual_name not in local_boost_manuals:
                        local_boost_manuals.append(manual_name)
            if inferred_product_hint:
                if effective_product_hint is None:
                    effective_product_hint = inferred_product_hint
                for manual_name in self._resolve_manual_hints_from_product(inferred_product_hint):
                    if manual_name not in local_boost_manuals:
                        local_boost_manuals.append(manual_name)
            for manual_name in self._resolve_manual_hints_from_text(
                [
                    query,
                    getattr(rewritten, "query_en", ""),
                    getattr(rewritten, "query_zh", ""),
                    getattr(rewritten, "query_mix", ""),
                ]
            ):
                if manual_name not in local_boost_manuals:
                    local_boost_manuals.append(manual_name)
            allowed_manuals = self._resolve_allowed_manuals(
                [
                    query,
                    getattr(rewritten, "query_en", ""),
                    getattr(rewritten, "query_zh", ""),
                    getattr(rewritten, "query_mix", ""),
                    " ".join(getattr(rewritten, "search_keywords", []) or []) if rewritten else "",
                ],
                (getattr(rewritten, "product_hint", None) if rewritten else None) or inferred_product_hint,
            )
            for manual_name in allowed_manuals:
                if manual_name not in local_boost_manuals:
                    local_boost_manuals.append(manual_name)
            results, _, _, _, _, _ = self._run_retrieval(
                retriever,
                sidecar_retriever,
                self._get_dense_retriever(),
                query,
                rewritten,
                local_boost_manuals,
                allowed_manuals=allowed_manuals,
                root_query=normalized_question,
            )
            sub_question_results.append((sub_question, self._expand_results_for_generation(results, query, rewritten)))

        generated = self.generator.generate(
            normalized_question,
            sub_question_results,
            self.repository.image_index(),
            product_hint=effective_product_hint,
        )
        generated = self.guardrail.review(generated)

        timestamp = int(time.time())
        self.memory.append_turn(
            session_id=session_id,
            question=normalized_question,
            answer=generated.answer,
            manuals=generated.used_manuals,
            product_name=effective_product_hint,
            section_titles=generated.used_sections,
        )

        return ChatReply(
            answer=generated.answer,
            session_id=session_id,
            timestamp=timestamp,
            references=generated.references,
            related_images=generated.related_images,
        )

    def diagnose(
        self,
        question: str,
        images: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict:
        normalized_question = normalize_question(question)
        if not normalized_question:
            raise ValueError("question is required")

        payload_images = images or []
        if len(payload_images) > 3:
            raise ValueError("images supports up to 3 items")

        session_id = session_id or f"diag_session_{uuid4().hex[:12]}"
        if self._agent_graph_enabled():
            state = self._get_agent_graph().run(
                question=normalized_question,
                session_id=session_id,
                images=payload_images,
            )
            return self._diagnostic_from_agent_state(state)

        context = self.memory.get(session_id)
        insight = self.multimodal.analyze(
            normalized_question,
            payload_images,
            context,
            self.repository.get_alias_lookup(),
        )
        if is_general_support_question(normalized_question) or is_manual_access_question(normalized_question):
            generated = self.guardrail.review(self.generator.build_support_fallback(normalized_question))
            refusal = self._diagnose_refusal(generated.answer, [])
            coverage_gap = self._diagnose_coverage_gap(refusal, insight.product_hint, [], [])
            return {
                "question": normalized_question,
                "session_id": session_id,
                "english_dominant": looks_english_dominant(normalized_question),
                "multimodal_insight": {
                    "manual_hints": insight.manual_hints,
                    "product_hint": insight.product_hint,
                    "image_warnings": insight.warnings,
                },
                "query_rewrite": [],
                "retrieval": [],
                "generation": {
                    "evidence_preview": [],
                    "llm_messages": [],
                    "answer": generated.answer,
                    "references": generated.references,
                    "related_images": generated.related_images,
                },
                "refusal": refusal,
                "coverage_gap": coverage_gap,
                "notes": {
                    "vector_retrieval": "not_configured",
                    "reranker": "not_configured",
                },
            }
        retriever = self._get_retriever()
        sidecar_retriever = self._get_sidecar_retriever()
        question_plan = self.query_rewriter.rewrite(normalized_question, insight.product_hint) if self._llm_query_plan_enabled() else None
        planned_sub_questions = self._safe_planned_sub_questions(normalized_question, question_plan)
        sub_questions = planned_sub_questions or split_sub_questions(normalized_question) or [normalized_question]
        sub_question_results: list[tuple[str, list]] = []
        query_traces: list[dict] = []
        retrieval_traces: list[dict] = []
        boost_manuals = insight.manual_hints or context.last_manuals
        effective_product_hint = insight.product_hint or getattr(question_plan, "product_hint", None)
        if getattr(question_plan, "product_hint", None):
            for manual_name in self._resolve_manual_hints_from_product(question_plan.product_hint):
                if manual_name not in boost_manuals:
                    boost_manuals.append(manual_name)

        for sub_question in sub_questions:
            query = rewrite_with_context(sub_question, effective_product_hint, context.last_sections)
            rewritten = (
                self.query_rewriter.rewrite(query, effective_product_hint)
                if (looks_english_dominant(query) or self._llm_query_plan_enabled())
                else None
            )
            local_boost_manuals = list(boost_manuals)
            inferred_product_hint = self._infer_product_hint_from_text(
                [
                    query,
                    getattr(rewritten, "query_en", ""),
                    getattr(rewritten, "query_zh", ""),
                    getattr(rewritten, "query_mix", ""),
                ]
            )
            if rewritten and rewritten.product_hint:
                if effective_product_hint is None:
                    effective_product_hint = rewritten.product_hint
                for manual_name in self._resolve_manual_hints_from_product(rewritten.product_hint):
                    if manual_name not in local_boost_manuals:
                        local_boost_manuals.append(manual_name)
            if inferred_product_hint:
                if effective_product_hint is None:
                    effective_product_hint = inferred_product_hint
                for manual_name in self._resolve_manual_hints_from_product(inferred_product_hint):
                    if manual_name not in local_boost_manuals:
                        local_boost_manuals.append(manual_name)
            for manual_name in self._resolve_manual_hints_from_text(
                [
                    query,
                    getattr(rewritten, "query_en", ""),
                    getattr(rewritten, "query_zh", ""),
                    getattr(rewritten, "query_mix", ""),
                ]
            ):
                if manual_name not in local_boost_manuals:
                    local_boost_manuals.append(manual_name)
            allowed_manuals = self._resolve_allowed_manuals(
                [
                    query,
                    getattr(rewritten, "query_en", ""),
                    getattr(rewritten, "query_zh", ""),
                    getattr(rewritten, "query_mix", ""),
                    " ".join(getattr(rewritten, "search_keywords", []) or []) if rewritten else "",
                ],
                (getattr(rewritten, "product_hint", None) if rewritten else None) or inferred_product_hint,
            )
            for manual_name in allowed_manuals:
                if manual_name not in local_boost_manuals:
                    local_boost_manuals.append(manual_name)
            rerank_results, variants, bm25_routes, vector_routes, hybrid_routes, fusion_results = self._run_retrieval(
                retriever,
                sidecar_retriever,
                self._get_dense_retriever(),
                query,
                rewritten,
                local_boost_manuals,
                allowed_manuals=allowed_manuals,
                root_query=normalized_question,
            )
            generation_results = self._expand_results_for_generation(rerank_results, query, rewritten)
            sub_question_results.append((sub_question, generation_results))
            query_traces.append(self._build_query_trace(query, rewritten))
            retrieval_traces.append(
                {
                    "sub_question": sub_question,
                    "query_variants": variants,
                    "boost_manuals": local_boost_manuals,
                    "allowed_manuals": allowed_manuals,
                    "bm25_top_k": {
                        route_name: [self._summarize_result(result) for result in results]
                        for route_name, results in bm25_routes.items()
                    },
                    "vector_top_k": {
                        route_name: [self._summarize_result(result) for result in results[: self.settings.default_top_k]]
                        for route_name, results in vector_routes.items()
                    },
                    "hybrid_top_k": {
                        route_name: [self._summarize_result(result) for result in results[: self.settings.default_top_k]]
                        for route_name, results in hybrid_routes.items()
                    },
                    "fusion_top_k": [self._summarize_result(result) for result in fusion_results],
                    "rerank_top_k": [self._summarize_result(result) for result in rerank_results],
                    "generation_top_k": [self._summarize_result(result) for result in generation_results],
                }
            )

        is_support_fallback = (
            (is_general_support_question(normalized_question) or is_manual_access_question(normalized_question))
            and not (insight.manual_hints or context.last_manuals)
        )
        if is_support_fallback:
            generated = self.guardrail.review(self.generator.build_support_fallback(normalized_question))
            llm_messages: list[dict] = []
            evidence_preview: list[dict] = []
        else:
            answer_context = self.generator._collect_answer_context(sub_question_results, self.repository.image_index())
            llm_messages = []
            if self.llm_client.is_enabled():
                messages = self.generator._build_llm_messages(
                    question=normalized_question,
                    sub_question_results=sub_question_results,
                    context=answer_context,
                    product_hint=effective_product_hint,
                )
                llm_messages = [{"role": message.role, "content": message.content} for message in messages]
            evidence_preview = []
            max_results_per_question = 2 if len(sub_question_results) <= 2 else 1
            for sub_question, results in sub_question_results:
                selected_results = self.generator._select_primary_evidence_results(
                    sub_question,
                    results[: max_results_per_question + 4],
                )
                evidence_preview.append(
                    {
                        "sub_question": sub_question,
                        "evidence": [
                            self._summarize_result(result)
                            for result in selected_results[:max_results_per_question]
                        ],
                    }
                )
            generated = self.guardrail.review(
                self.generator.generate(
                    normalized_question,
                    sub_question_results,
                    self.repository.image_index(),
                    product_hint=effective_product_hint,
                )
            )

        refusal = self._diagnose_refusal(generated.answer, sub_question_results)
        coverage_gap = self._diagnose_coverage_gap(
            refusal,
            effective_product_hint,
            query_traces,
            sub_question_results,
        )
        return {
            "question": normalized_question,
            "session_id": session_id,
            "english_dominant": looks_english_dominant(normalized_question),
            "multimodal_insight": {
                "manual_hints": insight.manual_hints,
                "product_hint": insight.product_hint,
                "image_warnings": insight.warnings,
            },
            "query_rewrite": query_traces,
            "retrieval": retrieval_traces,
            "generation": {
                "evidence_preview": evidence_preview,
                "llm_messages": llm_messages,
                "answer": generated.answer,
                "references": generated.references,
                "related_images": generated.related_images,
            },
            "refusal": refusal,
            "coverage_gap": coverage_gap,
            "notes": {
                "vector_retrieval": ("embedding_api" if self.settings.dense_enabled and self.embedding_client.is_enabled() and self._get_dense_retriever() is not None and self._get_dense_retriever().is_ready() else "not_configured"),
                "reranker": ("cross_encoder_api" if self.settings.rerank_enabled and self.reranker_client.is_enabled() else "not_configured"),
            },
        }
