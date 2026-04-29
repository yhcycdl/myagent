from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from app.services.llm_client import LLMClient, LLMMessage
from app.services.preprocess import looks_english_dominant


QUERY_REWRITE_PROMPT = (
    "你负责把英文或中英混合的用户问题改写成适合中文说明书检索的查询。"
    "不要回答问题本身。"
    "请保留关键英文术语、型号、按钮名、功能名，不要误翻。"
    "如果能判断出产品类别，请给出简短中文产品提示，例如：空调、摩托艇、耳机、冰箱、电话基站。"
    "输出严格使用以下三行格式：\n"
    "中文检索问题: ...\n"
    "检索关键词: ...\n"
    "产品提示: ...\n"
    "如果没有明确产品提示，写“无”。"
)
QUERY_PLAN_PROMPT = (
    "你是客服RAG系统的检索规划器，不回答用户问题。"
    "你的任务是把用户问题解析成稳定、可检索的结构化计划。"
    "必须保留型号、按钮名、英文专有名词、错误码、功能名。"
    "如果问题是订单、售后、物流、发票、投诉、退换货，也要标成 support_policy，不要强行映射到产品说明书。"
    "只输出严格JSON，不要Markdown，不要解释。格式："
    "{"
    "\"intent\":\"manual_qa|support_policy|troubleshooting|install_setup|parameter|image_status|unknown\","
    "\"product_hint\":\"产品名或无\","
    "\"need_image\":true,"
    "\"sub_questions\":[\"需要逐项回答的问题\"],"
    "\"queries\":[\"适合检索的query，包含产品/型号/动作/英文关键词\"],"
    "\"keywords\":[\"关键检索词\"]"
    "}"
)

DOMAIN_TERM_GLOSSARY: tuple[tuple[str, dict[str, object]], ...] = (
    (
        r"\bjetski\b|\bjet ski\b|\bwatercraft\b|\bpwc\b",
        {"product_hint": "摩托艇", "keywords": ["摩托艇"]},
    ),
    (
        r"\bboat\b|jet boat|210fsh|bimini|bilge|livewell|aerator",
        {"product_hint": "boat", "keywords": ["boat", "jet boat"]},
    ),
    (
        r"max load|maximum load",
        {"translated": "最大载重", "keywords": ["最大载重", "载重", "重量"]},
    ),
    (
        r"start .*jetski|start .*boat|turn on .*engine",
        {"translated": "启动发动机", "keywords": ["启动发动机", "发动机启动", "启动开关", "熄火绳"]},
    ),
    (
        r"jet wash",
        {
            "translated": "jet wash",
            "keywords": ["jet wash", "To use the jet wash", "coil hose", "hose fitting", "jet wash switch", "fresh water"],
            "product_hint": "boat",
        },
    ),
    (
        r"storage compartments?|compartments?",
        {"translated": "储物舱", "keywords": ["储物", "储物舱", "放置物品"]},
    ),
    (
        r"\bfuse\b",
        {"translated": "fuse replacement", "keywords": ["fuse replacement", "replace fuse", "fuse box", "spare fuse", "blown fuse"]},
    ),
    (
        r"maintenance setting screen",
        {
            "translated": "维护设置画面",
            "keywords": ["Maintenance setting screen", "maintenance", "Reset", "hours of operation"],
            "product_hint": "boat",
        },
    ),
    (
        r"factory reset screen",
        {
            "translated": "出厂重置画面",
            "keywords": ["Factory reset screen", "factory reset", "steering position", "reset"],
            "product_hint": "boat",
        },
    ),
    (
        r"steering position",
        {"translated": "转向位置", "keywords": ["steering position", "steering", "position"], "product_hint": "boat"},
    ),
    (
        r"\bcamera\b",
        {"product_hint": "相机", "keywords": ["相机"]},
    ),
    (
        r"camera battery|charge .*camera battery|battery .*camera",
        {"translated": "相机电池充电", "keywords": ["电池", "充电", "可充电电池"]},
    ),
    (
        r"install .*card|card .*camera|memory card",
        {"translated": "安装存储卡", "keywords": ["存储卡", "装入", "插入"]},
    ),
    (
        r"shutter button",
        {"translated": "快门按钮", "keywords": ["快门按钮", "快门"]},
    ),
    (
        r"date/time battery|date and time battery",
        {"translated": "日期和时间设置电池", "keywords": ["日期", "时间", "电池", "设置"]},
    ),
    (
        r"eyepiece cover",
        {"translated": "目镜盖", "keywords": ["目镜盖", "目镜", "盖"]},
    ),
    (
        r"view .* on tv|image on tv",
        {"translated": "在电视上查看图像", "keywords": ["电视", "图像", "回放"]},
    ),
    (
        r"erase all images|delete all images",
        {"translated": "全部删除图像", "keywords": ["全部删除图像", "删除", "图像", "回放"]},
    ),
    (
        r"\bereader\b|\be-reader\b|\bebook reader\b",
        {"product_hint": "ereader", "keywords": ["ereader", "ebook reader", "main menu", "browser history"]},
    ),
    (
        r"main menu|browser history",
        {
            "translated": "电子阅读器 主菜单 浏览历史 功能",
            "keywords": ["Main Menu", "Browser History", "eBook", "Explorer"],
            "product_hint": "ereader",
        },
    ),
    (
        r"\bfax\b|fax machine",
        {"product_hint": "fax", "keywords": ["fax", "telephone line cord", "LINE jack", "EXT jack"]},
    ),
    (
        r"connect(?:ing)? .*fax|fax .*connect",
        {
            "translated": "传真机 连接 电话线 LINE EXT 步骤",
            "keywords": [
                "telephone wall jack",
                "AC power outlet",
                "telephone line cord",
                "fully inserted",
                "fax function",
            ],
            "product_hint": "fax",
        },
    ),
    (
        r"landline|base station|handset|caller id",
        {"product_hint": "landline", "keywords": ["landline", "base station", "handset", "telephone"]},
    ),
    (
        r"connect .*base station|base station .*connect",
        {
            "translated": "固定电话 连接基站 电源 电话线",
            "keywords": ["Connect the base station", "telephone socket", "power socket", "base station"],
            "product_hint": "landline",
        },
    ),
    (
        r"lawn mower|\bmower\b",
        {"product_hint": "lawn mower", "keywords": ["lawn mower", "mower", "engine oil"]},
    ),
    (
        r"change .*engine oil|replace .*engine oil|engine oil",
        {
            "translated": "割草机 更换发动机机油 放油 加注机油",
            "keywords": ["Changing the Engine Oil", "engine oil", "change oil", "drain oil", "oil fill", "lawn mower"],
            "product_hint": "lawn mower",
        },
    ),
    (
        r"riding uphill|uphill",
        {
            "translated": "snowmobile riding uphill climb uphill side running boards",
            "keywords": ["snowmobile", "riding uphill", "uphill side", "running boards", "lean forward"],
            "product_hint": "snowmobile",
        },
    ),
    (
        r"riding downhill|downhill",
        {
            "translated": "snowmobile riding downhill engine compression brake",
            "keywords": ["snowmobile", "riding downhill", "engine compression", "brake", "minimum speed"],
            "product_hint": "snowmobile",
        },
    ),
    (
        r"crossing a slope|cross slope|side hill|sidehill",
        {
            "translated": "snowmobile crossing a slope side hill uphill side running board",
            "keywords": ["snowmobile", "crossing a slope", "uphill side", "running board", "sideways slipping"],
            "product_hint": "snowmobile",
        },
    ),
    (
        r"snowmobile",
        {"product_hint": "snowmobile", "keywords": ["snowmobile"]},
    ),
    (
        r"spark plug",
        {
            "translated": "火花塞 检查 拆卸 安装 间隙",
            "keywords": ["spark plug", "inspect", "remove", "gap", "install"],
        },
    ),
    (
        r"multi-use pressure cooker|pressure cooker|air fryer",
        {
            "product_hint": "multi-use pressure cooker",
            "keywords": ["pressure cooker", "air fryer", "float valve", "steam release valve"],
        },
    ),
    (
        r"quick release|\bqpr\b|steam release|vent(?:ing)? method",
        {
            "translated": "多功能压力锅 快速泄压 Quick Release 蒸汽释放阀 Vent",
            "keywords": ["Quick Release", "QPR", "quick release button", "steam release valve", "Vent position"],
            "product_hint": "multi-use pressure cooker",
        },
    ),
    (
        r"natural release|\bnpr\b|\bnr\b",
        {
            "translated": "多功能压力锅 自然泄压 Natural Release",
            "keywords": ["Natural Release", "NR", "NPR", "depressurizes naturally", "float valve"],
            "product_hint": "multi-use pressure cooker",
        },
    ),
    (
        r"float valve",
        {
            "translated": "多功能压力锅 浮阀 float valve 落下 压力释放",
            "keywords": ["float valve", "drops", "pressure", "lid", "release pressure"],
            "product_hint": "multi-use pressure cooker",
        },
    ),
    (
        r"microwave",
        {"product_hint": "microwave", "keywords": ["microwave", "over-the-range microwave"]},
    ),
    (
        r"motherboard|mainboard|bios|uefi",
        {"product_hint": "motherboard", "keywords": ["motherboard", "BIOS", "UEFI", "setup"]},
    ),
)


@dataclass(slots=True)
class QueryRewriteResult:
    retrieval_query: str
    query_en: str
    query_zh: str | None = None
    query_mix: str | None = None
    translated_question: str | None = None
    search_keywords: list[str] = field(default_factory=list)
    product_hint: str | None = None
    intent: str | None = None
    need_image: bool = False
    planned_sub_questions: list[str] = field(default_factory=list)


class QueryRewriteService:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def rewrite(
        self,
        question: str,
        product_hint: str | None = None,
    ) -> QueryRewriteResult | None:
        use_query_plan = self._query_plan_enabled()
        if not looks_english_dominant(question) and not use_query_plan:
            return None

        rule_rewrite = self._rule_rewrite(question, product_hint)
        if self.llm_client is None or not self.llm_client.is_enabled():
            return rule_rewrite

        messages = self._build_query_plan_messages(question, product_hint) if use_query_plan else [
            LLMMessage(role="system", content=QUERY_REWRITE_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"原问题: {question}\n"
                    f"已有产品提示: {product_hint or '无'}\n"
                    "请输出检索改写。"
                ),
            ),
        ]
        raw = self.llm_client.chat(messages, temperature=0.0, max_tokens=220 if use_query_plan else 160)
        if not raw:
            return rule_rewrite
        if use_query_plan:
            return self._parse_query_plan(raw, question, product_hint) or rule_rewrite
        return self._parse_rewrite(raw, question, product_hint) or rule_rewrite

    def _query_plan_enabled(self) -> bool:
        return os.getenv("LLM_QUERY_PLAN_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

    def _build_query_plan_messages(self, question: str, product_hint: str | None) -> list[LLMMessage]:
        return [
            LLMMessage(role="system", content=QUERY_PLAN_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"用户问题：{question}\n"
                    f"已有产品提示：{product_hint or '无'}\n"
                    "输出检索规划JSON。"
                ),
            ),
        ]

    def _rule_rewrite(
        self,
        question: str,
        product_hint: str | None,
    ) -> QueryRewriteResult | None:
        translated, keywords, normalized_product_hint = self._apply_glossary_overrides(
            question,
            None,
            [],
            product_hint,
        )
        if not translated and not keywords and not normalized_product_hint:
            return None

        original_terms = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{1,}", question)
        query_parts = []
        if translated:
            query_parts.append(translated)
        if keywords:
            query_parts.append(" ".join(keywords[:12]))
        if original_terms:
            query_parts.append(" ".join(original_terms[:10]))
        retrieval_query = " ".join(part.strip() for part in query_parts if part.strip())
        if not retrieval_query:
            return None
        return QueryRewriteResult(
            retrieval_query=retrieval_query,
            query_en=question,
            query_zh=None,
            query_mix=retrieval_query,
            translated_question=translated,
            search_keywords=keywords,
            product_hint=normalized_product_hint,
        )

    def _parse_rewrite(
        self,
        raw: str,
        original_question: str,
        original_product_hint: str | None,
    ) -> QueryRewriteResult | None:
        translated = self._extract_field(raw, "中文检索问题")
        keywords_line = self._extract_field(raw, "检索关键词")
        product_hint = self._extract_field(raw, "产品提示")

        keywords = self._split_keywords(keywords_line)
        if not translated and not keywords:
            return None

        normalized_product_hint = product_hint if product_hint and product_hint != "无" else original_product_hint
        translated, keywords, normalized_product_hint = self._apply_glossary_overrides(
            original_question,
            translated,
            keywords,
            normalized_product_hint,
        )

        query_parts: list[str] = []
        if translated:
            query_parts.append(translated)
        if keywords:
            query_parts.append(" ".join(keywords[:8]))

        original_terms = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{1,}", original_question)
        if original_terms:
            query_parts.append(" ".join(original_terms[:6]))

        retrieval_query = " ".join(part.strip() for part in query_parts if part.strip())
        if not retrieval_query:
            return None

        return QueryRewriteResult(
            retrieval_query=retrieval_query,
            query_en=original_question,
            query_zh=translated,
            query_mix=retrieval_query,
            translated_question=translated,
            search_keywords=keywords,
            product_hint=normalized_product_hint,
        )

    def _parse_query_plan(
        self,
        raw: str,
        original_question: str,
        original_product_hint: str | None,
    ) -> QueryRewriteResult | None:
        payload = self._extract_json_object(raw)
        if not payload:
            return None

        intent = self._safe_string(payload.get("intent"))
        product_hint = self._safe_string(payload.get("product_hint"))
        normalized_product_hint = product_hint if product_hint and product_hint != "无" else original_product_hint
        planned_sub_questions = self._safe_string_list(payload.get("sub_questions"))[:5]
        queries = self._safe_string_list(payload.get("queries"))[:6]
        keywords = self._safe_string_list(payload.get("keywords"))[:12]
        need_image = bool(payload.get("need_image")) if isinstance(payload.get("need_image"), bool) else False

        rule_rewrite = self._rule_rewrite(original_question, original_product_hint)
        translated = " ".join(planned_sub_questions[:2]) or None
        translated, keywords, normalized_product_hint = self._apply_glossary_overrides(
            original_question,
            translated,
            keywords,
            normalized_product_hint,
        )
        if rule_rewrite and rule_rewrite.product_hint:
            normalized_product_hint = rule_rewrite.product_hint

        query_parts = []
        if rule_rewrite:
            query_parts.append(rule_rewrite.retrieval_query)
        query_parts.extend(queries)
        if translated:
            query_parts.append(translated)
        if keywords:
            query_parts.append(" ".join(keywords[:12]))
        original_terms = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{1,}", original_question)
        if original_terms:
            query_parts.append(" ".join(original_terms[:10]))

        retrieval_query = " ".join(part.strip() for part in query_parts if part and part.strip())
        retrieval_query = re.sub(r"\s+", " ", retrieval_query).strip()
        if not retrieval_query:
            return None

        return QueryRewriteResult(
            retrieval_query=retrieval_query,
            query_en=original_question,
            query_zh=translated,
            query_mix=retrieval_query,
            translated_question=translated,
            search_keywords=list(dict.fromkeys((rule_rewrite.search_keywords if rule_rewrite else []) + keywords)),
            product_hint=normalized_product_hint,
            intent=intent,
            need_image=need_image,
            planned_sub_questions=planned_sub_questions,
        )

    def _extract_json_object(self, raw: str) -> dict | None:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None

    def _safe_string(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _safe_string_list(self, value: object) -> list[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
        return items

    def _extract_field(self, text: str, field_name: str) -> str | None:
        match = re.search(rf"{re.escape(field_name)}\s*[:：]\s*(.+)", text)
        if not match:
            return None
        value = match.group(1).strip()
        return value or None

    def _split_keywords(self, keywords_line: str | None) -> list[str]:
        if not keywords_line:
            return []
        if keywords_line == "无":
            return []
        parts = re.split(r"[、,，;/|]\s*|\s{2,}", keywords_line)
        return [part.strip() for part in parts if part.strip()]

    def _apply_glossary_overrides(
        self,
        original_question: str,
        translated: str | None,
        keywords: list[str],
        product_hint: str | None,
    ) -> tuple[str | None, list[str], str | None]:
        lowered = original_question.lower()
        translated_parts: list[str] = [translated] if translated else []
        merged_keywords = list(keywords)
        merged_product_hint = product_hint

        for pattern, payload in DOMAIN_TERM_GLOSSARY:
            if not re.search(pattern, lowered):
                continue
            translated_phrase = payload.get("translated")
            if isinstance(translated_phrase, str) and translated_phrase not in translated_parts:
                translated_parts.append(translated_phrase)
            for keyword in payload.get("keywords", []):
                if keyword not in merged_keywords:
                    merged_keywords.append(keyword)
            if isinstance(payload.get("product_hint"), str):
                merged_product_hint = payload["product_hint"]

        merged_translated = " ".join(part.strip() for part in translated_parts if part and part.strip()) or None
        return merged_translated, merged_keywords[:12], merged_product_hint
