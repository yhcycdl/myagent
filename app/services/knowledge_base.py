from __future__ import annotations

import ast
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from app.core.config import Settings


LOGGER = logging.getLogger(__name__)
SECTION_MARK_RE = re.compile(r"#\s*")
IMAGE_MARK_RE = re.compile(r"\[\[IMG:([^\]]+)\]\]")
KEYWORD_RE = re.compile(r"[A-Za-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}")
STEP_LINE_RE = re.compile(r"^\s*(?:\d+|[A-Z])(?:[.)、]|\s)")
BULLET_LINE_RE = re.compile(r"^\s*[•●\-]")
ALPHA_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+./-]{1,}")
MENU_TITLE_HINT_RE = re.compile(
    r"(菜单|设置|删除|复制|插入|装入|更换|选择|模式|回放|拍摄|启动|关闭|调节|单位|亮度|时间|日期|语言|维护|重置)"
)
MENU_TEXT_HINT_RE = re.compile(
    r"^(选项[:：]|设置生效|出现菜单画面|显示.*画面|按下MENU/OK|按下启动开关|按下.*按钮|选择.*然后按下)"
)
NUMERIC_SNIPPET_RE = re.compile(
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
ENGLISH_SUMMARY_MANUAL_SPECS: tuple[tuple[str, str, set[str]], ...] = (
    (
        "英文单反相机手册",
        "英文单反相机",
        {"dslr", "cf card", "eyepiece cover", "date/time battery", "canon camera", "eos camera"},
    ),
    (
        "英文咖啡机手册",
        "英文咖啡机",
        {"coffee machine", "coffee maker", "espresso machine", "nespresso"},
    ),
    (
        "英文空气炸锅手册",
        "英文空气炸锅",
        {"air fryer", "airfryer", "nutriu", "airfrying", "keep warm mode"},
    ),
    (
        "英文喷射船手册",
        "英文喷射船",
        {"boat", "jet boat", "210fsh", "bimini top", "bilge pump", "livewell", "aerator", "yamaha boat"},
    ),
    (
        "英文水上摩托手册",
        "英文水上摩托",
        {"waverunner", "personal watercraft", "pwc", "watercraft", "jet ski", "jetski"},
    ),
    (
        "英文传真机手册",
        "英文传真机",
        {"fax", "fax machine", "telephone line cord", "line jack", "brother fax"},
    ),
    (
        "英文耳机手册",
        "英文耳机",
        {"earphones", "headphones", "earbuds", "charging case", "noise canceling"},
    ),
    (
        "英文电子阅读器手册",
        "英文电子阅读器",
        {"ereader", "e-reader", "ebook reader", "browser history", "main menu", "ebook"},
    ),
    (
        "英文烧烤炉手册",
        "英文烧烤炉",
        {"grill", "outdoor grill", "barbecue", "bbq"},
    ),
    (
        "英文雪地摩托手册",
        "英文雪地摩托",
        {"snowmobile", "spark plug", "track", "ski stance"},
    ),
    (
        "英文电视机手册",
        "英文电视机",
        {"television", "tv", "color television"},
    ),
    (
        "英文吸尘器手册",
        "英文吸尘器",
        {"vacuum", "robot vacuum", "dock charger"},
    ),
    (
        "英文网络摄像头手册",
        "英文网络摄像头",
        {"network camera", "security camera", "camera installation", "cloud camera"},
    ),
    (
        "英文电动牙刷手册",
        "英文电动牙刷",
        {"toothbrush", "electric toothbrush", "brush head"},
    ),
    (
        "英文洗衣机手册",
        "英文洗衣机",
        {"washing machine", "washer", "washing program"},
    ),
    (
        "英文多功能压力锅空气炸锅手册",
        "英文多功能压力锅空气炸锅",
        {
            "multi-use pressure cooker",
            "pressure cooker",
            "air fryer lid",
            "quick release",
            "natural release",
            "float valve",
            "steam release valve",
        },
    ),
    (
        "英文微波炉手册",
        "英文微波炉",
        {"over-the-range microwave", "microwave", "microwave oven"},
    ),
    (
        "英文主板手册",
        "英文主板",
        {"motherboard", "mainboard", "bios", "uefi", "asus motherboard"},
    ),
    (
        "英文固定电话手册",
        "英文固定电话",
        {"landline", "telephone", "base station", "handset", "caller id", "phonebook", "xl490", "xl495"},
    ),
    (
        "英文割草机手册",
        "英文割草机",
        {"lawn mower", "mower", "engine oil", "cutting height", "grass catcher"},
    ),
)

MANUAL_ENGLISH_ALIASES: dict[str, set[str]] = {
    "VR头显手册": {"vr", "vr headset", "headset", "virtual reality headset"},
    "人体工学椅手册": {"ergonomic chair", "office chair", "chair"},
    "健身单车手册": {"exercise bike", "spin bike", "bike"},
    "健身追踪器手册": {"fitness tracker", "tracker", "smartwatch", "watch"},
    "儿童电动摩托车手册": {"kids motorcycle", "electric motorcycle", "ride-on motorcycle"},
    "冰箱手册": {"refrigerator", "fridge"},
    "功能键盘手册": {"keyboard", "gaming keyboard"},
    "发电机手册": {"generator"},
    "可编程温控器手册": {"programmable thermostat", "thermostat"},
    "吹风机手册": {"hair dryer", "blow dryer", "dryer"},
    "摩托艇手册": {"jetski", "jet ski", "watercraft", "pwc"},
    "水泵手册": {"water pump", "pump"},
    "洗碗机手册": {"dishwasher"},
    "烤箱手册": {"oven"},
    "电钻手册": {"drill", "power drill"},
    "相机手册": {"camera", "dslr", "camcorder"},
    "空气净化器手册": {"air purifier", "purifier"},
    "空调手册": {"air conditioner", "ac", "remote controller", "remote"},
    "蒸汽清洁机手册": {"steam cleaner", "steam cleaning machine"},
    "蓝牙激光鼠标手册": {"bluetooth laser mouse", "mouse"},
}
MANUAL_ENGLISH_ALIASES.update({manual_name: aliases for manual_name, _, aliases in ENGLISH_SUMMARY_MANUAL_SPECS})
RETRIEVAL_TRANSLATIONS: dict[str, str] = {
    "摩托艇": "jetski watercraft",
    "相机": "camera",
    "空调": "air conditioner",
    "水泵": "water pump",
    "发电机": "generator",
    "冰箱": "refrigerator",
    "洗碗机": "dishwasher",
    "吹风机": "hair dryer",
    "空气净化器": "air purifier",
    "电钻": "drill",
    "蒸汽清洁机": "steam cleaner",
    "可编程温控器": "programmable thermostat",
    "健身追踪器": "fitness tracker",
    "健身单车": "exercise bike",
    "功能键盘": "keyboard",
    "蓝牙激光鼠标": "bluetooth laser mouse",
    "VR头显": "vr headset",
    "人体工学椅": "ergonomic chair",
    "儿童电动摩托车": "kids electric motorcycle",
    "最大载重": "maximum load",
    "载重": "load",
    "重量": "weight",
    "容量": "capacity",
    "功率": "power",
    "电压": "voltage",
    "电流": "current",
    "压力": "pressure",
    "温度": "temperature",
    "速度": "speed",
    "转速": "rpm",
    "启动发动机": "start engine",
    "发动机启动": "engine start",
    "启动开关": "start switch",
    "熄火绳": "engine stop lanyard",
    "喷射清洗": "jet wash",
    "清洗": "clean",
    "使用后": "after use",
    "维护设置": "maintenance setting",
    "出厂重置": "factory reset",
    "保险丝": "fuse",
    "存储卡": "memory card",
    "装入": "insert",
    "插入": "insert",
    "删除图像": "delete images",
    "全部删除图像": "delete all images",
    "删除": "delete",
    "图像": "images",
    "回放": "playback",
    "电池": "battery",
    "充电": "charge",
    "可充电电池": "rechargeable battery",
    "快门按钮": "shutter button",
    "快门": "shutter",
    "目镜盖": "eyepiece cover",
    "目镜": "eyepiece",
    "日期": "date",
    "时间": "time",
    "设置": "setup",
    "设置画面": "settings screen",
    "画面": "screen",
    "屏幕": "screen",
    "转向": "steering",
    "位置": "position",
    "储物": "storage",
    "储物舱": "storage compartment",
    "操作方法": "operation",
    "警告": "warning",
    "重要信息": "important information",
    "目录": "contents",
    "概述": "overview",
    "前言": "introduction",
    "目标": "goal",
    "训练": "training",
    "练习": "practice",
    "登艇": "boarding",
    "启动": "start",
    "关闭": "close",
    "设置通知": "set notifications",
    "部件": "components",
    "组件": "components",
    "按钮": "button",
    "接口": "port",
    "更换": "replace",
    "安装": "install",
    "拆卸": "remove",
    "转向位置": "steering position",
}


@dataclass(slots=True)
class ParsedManual:
    manual_name: str
    product_name: str
    source_file: str
    text: str
    image_ids: list[str]
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict) -> "ParsedManual":
        return cls(**payload)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    manual_name: str
    product_name: str
    section_title: str
    page: int | None
    text: str
    image_ids: list[str]
    keywords: list[str]
    source_file: str
    order: int
    chunk_type: str = "general"

    @classmethod
    def from_dict(cls, payload: dict) -> "ChunkRecord":
        return cls(**payload)


@dataclass(slots=True)
class ImageRecord:
    image_id: str
    manual_name: str
    product_name: str
    caption: str
    page: int | None
    related_chunk_ids: list[str]
    image_path: str | None

    @classmethod
    def from_dict(cls, payload: dict) -> "ImageRecord":
        return cls(**payload)


@dataclass(slots=True)
class RetrievalCorpusRecord:
    chunk_id: str
    manual_name: str
    product_name: str
    chunk_type: str
    title_zh: str
    text_zh: str
    keywords_zh: list[str]
    title_en: str
    text_en_summary: str
    keywords_en: list[str]
    product_aliases_en: list[str]
    retrieval_text_en: str
    retrieval_text_mix: str

    @classmethod
    def from_dict(cls, payload: dict) -> "RetrievalCorpusRecord":
        return cls(**payload)


@dataclass(slots=True)
class DenseIndexRecord:
    chunk_id: str
    vector: list[float]

    @classmethod
    def from_dict(cls, payload: dict) -> "DenseIndexRecord":
        return cls(**payload)


def _product_name_from_manual(manual_name: str) -> str:
    return manual_name.removesuffix("手册").strip()


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _extract_english_terms(text: str) -> list[str]:
    if not text:
        return []
    ordered_terms: list[str] = []
    for source, translated in sorted(RETRIEVAL_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        if source in text:
            ordered_terms.append(translated)
    ordered_terms.extend(token.lower() for token in ALPHA_TOKEN_RE.findall(text))
    ordered_terms.extend(match.group(0).lower() for match in NUMERIC_SNIPPET_RE.finditer(text))
    return _dedupe_keep_order(ordered_terms)


def _render_title_en(chunk: ChunkRecord, product_aliases_en: list[str]) -> str:
    title_terms = _extract_english_terms(chunk.section_title)
    if title_terms:
        return " | ".join(title_terms[:4])
    fallback = list(product_aliases_en[:2]) + [chunk.chunk_type.replace("_", " ")]
    return " | ".join(_dedupe_keep_order(fallback))


def _render_text_en_summary(
    chunk: ChunkRecord,
    title_en: str,
    product_aliases_en: list[str],
) -> tuple[str, list[str]]:
    keyword_terms: list[str] = []
    for keyword in chunk.keywords:
        keyword_terms.extend(_extract_english_terms(keyword))

    body_terms = _extract_english_terms(chunk.text)
    salient_terms: list[str] = []
    for fragment in _salient_fragments(chunk.text):
        salient_terms.extend(_extract_english_terms(fragment))
    summary_terms = _dedupe_keep_order(
        [title_en]
        + keyword_terms
        + salient_terms
        + body_terms
        + product_aliases_en
        + [chunk.chunk_type.replace("_", " ")]
    )
    keywords_en = summary_terms[1:13] if summary_terms and summary_terms[0] == title_en else summary_terms[:12]
    return ", ".join(summary_terms[:18]), keywords_en


def build_retrieval_corpus_records(chunks: list[ChunkRecord]) -> list[RetrievalCorpusRecord]:
    records: list[RetrievalCorpusRecord] = []
    for chunk in chunks:
        product_aliases_en = sorted(MANUAL_ENGLISH_ALIASES.get(chunk.manual_name, set()))
        if not product_aliases_en:
            product_aliases_en = _extract_english_terms(chunk.product_name)
        title_en = _render_title_en(chunk, product_aliases_en)
        text_en_summary, keywords_en = _render_text_en_summary(chunk, title_en, product_aliases_en)
        retrieval_text_en = " ".join(
            _dedupe_keep_order([title_en, text_en_summary] + keywords_en + product_aliases_en)
        )
        retrieval_text_mix = " ".join(
            _dedupe_keep_order(
                [
                    chunk.product_name,
                    chunk.section_title,
                    " ".join(chunk.keywords[:6]),
                    title_en,
                    " ".join(keywords_en[:8]),
                ]
            )
        )
        records.append(
            RetrievalCorpusRecord(
                chunk_id=chunk.chunk_id,
                manual_name=chunk.manual_name,
                product_name=chunk.product_name,
                chunk_type=chunk.chunk_type,
                title_zh=chunk.section_title,
                text_zh=chunk.text,
                keywords_zh=list(chunk.keywords),
                title_en=title_en,
                text_en_summary=text_en_summary,
                keywords_en=keywords_en,
                product_aliases_en=product_aliases_en,
                retrieval_text_en=retrieval_text_en,
                retrieval_text_mix=retrieval_text_mix,
            )
        )
    return records


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_concatenated_json_payloads(stripped: str) -> list[list]:
    decoder = json.JSONDecoder()
    payloads: list[list] = []
    position = 0
    while position < len(stripped):
        while position < len(stripped) and stripped[position].isspace():
            position += 1
        if position >= len(stripped):
            break
        try:
            payload, end = decoder.raw_decode(stripped, position)
        except Exception:
            return []
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            return []
        payloads.append([payload[0], list(payload[1])])
        position = end
    return payloads


def _parse_manual_payloads(raw: str) -> list[list]:
    stripped = raw.strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            payload = parser(stripped)
            if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                return [[payload[0], list(payload[1])]]
        except Exception:
            continue

    concatenated_payloads = _parse_concatenated_json_payloads(stripped)
    if concatenated_payloads:
        return concatenated_payloads

    delimiter = '", ['
    split_at = stripped.rfind(delimiter)
    if stripped.startswith('["') and split_at > 1 and stripped.endswith("]"):
        text_payload = stripped[2:split_at]
        images_payload = stripped[split_at + 3 : -1].strip()
        images = ast.literal_eval(images_payload)
        text = bytes(text_payload, "utf-8").decode("unicode_escape", errors="ignore")
        return [[text, list(images)]]

    raise ValueError("Unable to parse manual payload")


def _manual_identity_for_payload(path: Path, index: int, payload_count: int) -> tuple[str, str, str]:
    if path.name == "汇总英文手册.txt" and payload_count > 1:
        if index < len(ENGLISH_SUMMARY_MANUAL_SPECS):
            manual_name, product_name, _ = ENGLISH_SUMMARY_MANUAL_SPECS[index]
        else:
            manual_name = f"英文汇总子手册{index + 1:02d}"
            product_name = manual_name.removesuffix("手册")
        source_file = f"{path.name}#{index + 1:02d}"
        return manual_name, product_name, source_file
    return path.stem, _product_name_from_manual(path.stem), path.name


def parse_manual_files(path: Path) -> list[ParsedManual]:
    raw = path.read_text(encoding="utf-8")
    try:
        payloads = _parse_manual_payloads(raw)
    except Exception as exc:
        LOGGER.warning("Skip malformed manual %s: %s", path.name, exc)
        return []

    manuals: list[ParsedManual] = []
    for index, payload in enumerate(payloads):
        text, image_ids = payload[0], payload[1]
        warnings: list[str] = []
        pic_count = str(text).count("<PIC>")
        if pic_count != len(image_ids):
            warnings.append(
                f"<PIC> count {pic_count} does not match image count {len(image_ids)}"
            )
        manual_name, product_name, source_file = _manual_identity_for_payload(path, index, len(payloads))
        manuals.append(
            ParsedManual(
                manual_name=manual_name,
                product_name=product_name,
                source_file=source_file,
                text=_normalize_text(str(text)),
                image_ids=[str(image_id) for image_id in image_ids],
                warnings=warnings,
            )
        )
    return manuals


def parse_manual_file(path: Path) -> ParsedManual | None:
    manuals = parse_manual_files(path)
    return manuals[0] if manuals else None


def _inject_image_markers(text: str, image_ids: list[str]) -> str:
    parts = text.split("<PIC>")
    if len(parts) == 1:
        return text

    rebuilt: list[str] = []
    for index, part in enumerate(parts[:-1]):
        rebuilt.append(part)
        if index < len(image_ids):
            rebuilt.append(f"\n[[IMG:{image_ids[index]}]]\n")
        else:
            rebuilt.append("\n[[IMG:MISSING]]\n")
    rebuilt.append(parts[-1])

    if len(image_ids) > len(parts) - 1:
        extras = " ".join(f"[[IMG:{image_id}]]" for image_id in image_ids[len(parts) - 1 :])
        rebuilt.append(f"\n{extras}\n")

    return "".join(rebuilt)


def _split_heading_and_body(block: str) -> tuple[str, str]:
    body = block.lstrip("#").strip()
    if not body:
        return "未命名章节", ""

    for separator in ("\n", "  ", "： ", ": "):
        idx = body.find(separator)
        if 0 < idx <= 48:
            title = body[:idx].strip(" ：:.")
            remainder = body[idx + len(separator) :].strip()
            return title or "未命名章节", remainder

    compact = body.replace("\n", " ").strip()
    if len(compact) <= 48:
        return compact, ""

    title = compact[:48].rsplit(" ", 1)[0].strip(" ：:.") or compact[:48].strip(" ：:.")
    remainder = compact[len(title) :].strip()
    return title, remainder


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    matches = list(SECTION_MARK_RE.finditer(text))
    if not matches:
        cleaned = text.strip()
        return [("概述", cleaned)] if cleaned else []

    prelude = text[: matches[0].start()].strip()
    if prelude:
        sections.append(("概述", prelude))

    for index, match in enumerate(matches):
        block_start = match.end()
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[block_start:block_end].strip()
        if not block:
            continue
        title, body = _split_heading_and_body(block)
        if title or body:
            sections.append((title, body))
    return sections


def _split_large_unit(unit: str, target_chars: int) -> list[str]:
    if len(unit) <= target_chars:
        return [unit]

    parts = re.split(r"(?<=[。！？.!?])\s+|(?=(?:\d+[.)、]|[A-Za-z][.)]|[-•●]))", unit)
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) <= 1:
        return [unit[i : i + target_chars] for i in range(0, len(unit), target_chars)]
    return parts


def _starts_new_structured_item(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(STEP_LINE_RE.match(stripped) or BULLET_LINE_RE.match(stripped))


def _build_structured_units(body: str) -> list[str]:
    lines = [line.rstrip() for line in body.splitlines()]
    if not lines:
        return []

    units: list[str] = []
    current: list[str] = []
    saw_structured_item = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current and current[-1] != "":
                current.append("")
            continue

        if _starts_new_structured_item(stripped):
            saw_structured_item = True
            if current:
                units.append("\n".join(part for part in current if part is not None).strip())
            current = [stripped]
            continue

        if current:
            current.append(stripped)
        else:
            current = [stripped]

    if current:
        units.append("\n".join(part for part in current if part is not None).strip())

    if saw_structured_item and len(units) >= 2:
        return [unit for unit in units if unit]
    return []


def _merge_units_with_limit(units: list[str], target_chars: int, overlap_chars: int) -> list[str]:
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for unit in units:
        projected = current_length + len(unit) + (1 if current else 0)
        if current and projected > target_chars:
            chunks.append("\n".join(current).strip())
            overlap: list[str] = []
            overlap_length = 0
            for previous in reversed(current):
                overlap.insert(0, previous)
                overlap_length += len(previous)
                if overlap_length >= overlap_chars:
                    break
            current = overlap + [unit]
            current_length = sum(len(item) for item in current)
        else:
            current.append(unit)
            current_length = projected

    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def _build_chunk_bodies(body: str, target_chars: int = 820, overlap_chars: int = 120) -> list[str]:
    structured_units = _build_structured_units(body)
    if structured_units:
        return _merge_units_with_limit(
            structured_units,
            target_chars=max(260, target_chars // 2),
            overlap_chars=0,
        )

    raw_units = [unit.strip() for unit in re.split(r"\n{2,}", body) if unit.strip()]
    units: list[str] = []
    for unit in raw_units:
        units.extend(_split_large_unit(unit, target_chars))
    return _merge_units_with_limit(units, target_chars, overlap_chars)


def _clean_chunk_text(text: str) -> str:
    text = IMAGE_MARK_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_chunk_text(section_title: str, text: str, chunk_type: str) -> str:
    if not text:
        return text
    if chunk_type == "title_only":
        return text
    if section_title in text:
        return text

    short_body = len(text) <= 120
    title_has_action = bool(MENU_TITLE_HINT_RE.search(section_title))
    body_has_menu_signal = bool(MENU_TEXT_HINT_RE.search(text))
    if short_body and (title_has_action or body_has_menu_signal):
        return f"{section_title}\n{text}".strip()
    return text


def _extract_keywords(text: str, section_title: str, product_name: str) -> list[str]:
    candidates = KEYWORD_RE.findall(f"{product_name} {section_title} {text}")
    scored: dict[str, float] = {}
    for keyword in candidates:
        normalized = keyword.lower()
        bonus = 2.0 if any(char.isdigit() for char in keyword) else 1.0
        scored[normalized] = max(scored.get(normalized, 0.0), math.log(len(keyword) + 1) + bonus)
    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return [keyword for keyword, _ in ranked[:12]]


def _salient_fragments(text: str, limit: int = 3) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    structured = [line for line in lines if _starts_new_structured_item(line)]
    if structured:
        return structured[:limit]

    sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", text) if part.strip()]
    return sentences[:limit]


def _classify_chunk(section_title: str, text: str) -> str:
    title = section_title.strip()
    lowered_title = title.lower()
    lowered_text = text.strip().lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    step_lines = sum(1 for line in lines if STEP_LINE_RE.match(line))
    bullet_lines = max(sum(1 for line in lines if BULLET_LINE_RE.match(line)), text.count("•"), text.count("●"))

    if "目录" in title or "contents" in lowered_title:
        return "toc"
    if lowered_text == lowered_title or text.strip() == title.strip():
        return "title_only"
    if title in {"注", "注：", "备注"} or title.startswith("注 ") or lowered_text.startswith(("注：", "注 ", "备注")):
        if step_lines >= 2:
            return "step"
        return "note"
    if any(keyword in title for keyword in ("保修", "质保", "免费服务", "除外责任", "免责")):
        return "warranty"
    if any(keyword in title for keyword in ("故障", "无法", "不发电", "异常", "报修前检查", "自诊断", "错误", "不运行", "有异味", "温度过高", "结霜", "冰晶")):
        return "troubleshoot"
    if any(keyword in title for keyword in ("部件", "组件", "接口", "按键", "按钮", "处理器单元")):
        return "component"
    if MENU_TITLE_HINT_RE.search(title) and (len(text) <= 180 or MENU_TEXT_HINT_RE.search(text)):
        return "menu"
    if step_lines >= 2:
        return "step"
    if title.startswith(("警告", "危险", "小心", "注意")) or lowered_text.startswith(
        ("警告", "危险", "小心", "注意")
    ):
        if bullet_lines >= 3:
            return "list"
        return "warning"
    if any(keyword in title for keyword in ("安装", "拆卸", "设置", "启动", "关闭", "清洁", "调节", "更换", "使用")):
        if step_lines >= 1 or bullet_lines >= 2:
            return "step"
    if bullet_lines >= 3:
        return "list"
    return "general"


def build_chunks(parsed_manual: ParsedManual) -> list[ChunkRecord]:
    marked_text = _inject_image_markers(parsed_manual.text, parsed_manual.image_ids)
    chunks: list[ChunkRecord] = []
    order = 0
    for section_title, body in _split_sections(marked_text):
        chunk_bodies = _build_chunk_bodies(body) if body else []
        if not chunk_bodies and section_title:
            chunk_bodies = [section_title]
        for chunk_body in chunk_bodies:
            image_ids = [image_id for image_id in IMAGE_MARK_RE.findall(chunk_body) if image_id != "MISSING"]
            cleaned_text = _clean_chunk_text(chunk_body)
            if not cleaned_text:
                continue
            order += 1
            chunk_type = _classify_chunk(section_title, cleaned_text)
            normalized_text = _normalize_chunk_text(section_title, cleaned_text, chunk_type)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{parsed_manual.product_name}_{order:04d}",
                    manual_name=parsed_manual.manual_name,
                    product_name=parsed_manual.product_name,
                    section_title=section_title,
                    page=None,
                    text=normalized_text,
                    image_ids=image_ids,
                    keywords=_extract_keywords(normalized_text, section_title, parsed_manual.product_name),
                    chunk_type=chunk_type,
                    source_file=parsed_manual.source_file,
                    order=order,
                )
            )
    return chunks


def build_image_records(
    parsed_manual: ParsedManual,
    chunks: list[ChunkRecord],
    image_dir: Path,
) -> list[ImageRecord]:
    image_to_chunks: dict[str, list[str]] = {}
    image_to_caption: dict[str, str] = {}
    for chunk in chunks:
        for image_id in chunk.image_ids:
            image_to_chunks.setdefault(image_id, []).append(chunk.chunk_id)
            image_to_caption.setdefault(image_id, chunk.section_title)

    records: list[ImageRecord] = []
    for image_id in parsed_manual.image_ids:
        image_path = next(iter(sorted(image_dir.glob(f"{image_id}.*"))), None)
        records.append(
            ImageRecord(
                image_id=image_id,
                manual_name=parsed_manual.manual_name,
                product_name=parsed_manual.product_name,
                caption=image_to_caption.get(image_id, parsed_manual.manual_name),
                page=None,
                related_chunk_ids=image_to_chunks.get(image_id, []),
                image_path=str(image_path) if image_path else None,
            )
        )
    return records


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_knowledge_base(settings: Settings) -> dict[str, int | list[str]]:
    settings.ensure_data_dirs()

    manuals: list[ParsedManual] = []
    chunks: list[ChunkRecord] = []
    images: list[ImageRecord] = []
    warnings: list[str] = []

    for manual_path in sorted(settings.raw_manual_dir.glob("*.txt")):
        parsed_manuals = parse_manual_files(manual_path)
        if not parsed_manuals:
            warnings.append(f"Skipped malformed manual: {manual_path.name}")
            continue

        for parsed_manual in parsed_manuals:
            pic_count = parsed_manual.text.count("<PIC>")
            image_count = len(parsed_manual.image_ids)
            smaller = max(1, min(pic_count, image_count))
            mismatch_ratio = max(pic_count, image_count) / smaller
            if mismatch_ratio > 5:
                warnings.append(
                    "Skipped likely corrupted manual: "
                    f"{parsed_manual.source_file} (pic_count={pic_count}, image_count={image_count})"
                )
                continue

            warnings.extend(f"{parsed_manual.source_file}: {warning}" for warning in parsed_manual.warnings)

            manuals.append(parsed_manual)
            manual_chunks = build_chunks(parsed_manual)
            chunks.extend(manual_chunks)
            images.extend(build_image_records(parsed_manual, manual_chunks, settings.image_dir))

    write_jsonl(settings.parsed_manual_path, (asdict(item) for item in manuals))
    write_jsonl(settings.chunk_path, (asdict(item) for item in chunks))
    retrieval_records = build_retrieval_corpus_records(chunks)
    write_jsonl(settings.retrieval_corpus_path, (asdict(item) for item in retrieval_records))
    write_jsonl(settings.image_path, (asdict(item) for item in images))

    metadata = {
        "manual_count": len(manuals),
        "chunk_count": len(chunks),
        "retrieval_record_count": len(retrieval_records),
        "image_count": len(images),
        "warnings": warnings,
    }
    settings.metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def build_retrieval_corpus(settings: Settings) -> dict[str, int]:
    settings.ensure_data_dirs()
    if not settings.chunk_path.exists():
        build_knowledge_base(settings)

    chunks = [ChunkRecord.from_dict(row) for row in load_jsonl(settings.chunk_path)]
    records = build_retrieval_corpus_records(chunks)
    write_jsonl(settings.retrieval_corpus_path, (asdict(item) for item in records))
    return {
        "chunk_count": len(chunks),
        "retrieval_record_count": len(records),
    }


class KnowledgeBaseRepository:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._chunks: list[ChunkRecord] = []
        self._images: list[ImageRecord] = []
        self._retrieval_corpus: list[RetrievalCorpusRecord] = []
        self._dense_index: list[DenseIndexRecord] = []
        self._parsed_manuals: list[ParsedManual] = []
        self._chunk_lookup: dict[str, ChunkRecord] = {}
        self._chunks_by_manual: dict[str, list[ChunkRecord]] = {}
        self._alias_lookup: dict[str, str] = {}
        self._version: float = 0.0

    @property
    def version(self) -> float:
        return self._version

    def ensure_ready(self) -> None:
        if not self.settings.chunk_path.exists():
            build_knowledge_base(self.settings)
        if not self.settings.retrieval_corpus_path.exists():
            build_retrieval_corpus(self.settings)
        self._refresh()

    def _refresh(self) -> None:
        latest_mtime = max(
            path.stat().st_mtime
            for path in (
                self.settings.parsed_manual_path,
                self.settings.chunk_path,
                self.settings.retrieval_corpus_path,
                self.settings.dense_index_path,
                self.settings.image_path,
            )
            if path.exists()
        )
        if latest_mtime <= self._version:
            return

        self._parsed_manuals = [
            ParsedManual.from_dict(row) for row in load_jsonl(self.settings.parsed_manual_path)
        ]
        self._chunks = [ChunkRecord.from_dict(row) for row in load_jsonl(self.settings.chunk_path)]
        self._retrieval_corpus = [
            RetrievalCorpusRecord.from_dict(row) for row in load_jsonl(self.settings.retrieval_corpus_path)
        ]
        self._dense_index = [
            DenseIndexRecord.from_dict(row) for row in load_jsonl(self.settings.dense_index_path)
        ]
        self._images = [ImageRecord.from_dict(row) for row in load_jsonl(self.settings.image_path)]
        self._chunk_lookup = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._chunks_by_manual = {}
        for chunk in self._chunks:
            self._chunks_by_manual.setdefault(chunk.manual_name, []).append(chunk)
        for manual_chunks in self._chunks_by_manual.values():
            manual_chunks.sort(key=lambda chunk: chunk.order)
        self._alias_lookup = self._build_alias_lookup()
        self._version = latest_mtime

    def _build_alias_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for manual in self._parsed_manuals:
            aliases = {
                manual.manual_name.lower(),
                manual.product_name.lower(),
                manual.manual_name.removesuffix("手册").lower(),
            }
            aliases.update(MANUAL_ENGLISH_ALIASES.get(manual.manual_name, set()))
            for alias in aliases:
                if alias:
                    lookup[alias] = manual.manual_name
        return lookup

    def get_chunks(self) -> list[ChunkRecord]:
        self.ensure_ready()
        return self._chunks

    def get_images(self) -> list[ImageRecord]:
        self.ensure_ready()
        return self._images

    def get_retrieval_corpus(self) -> list[RetrievalCorpusRecord]:
        self.ensure_ready()
        return self._retrieval_corpus

    def get_dense_index(self) -> list[DenseIndexRecord]:
        self.ensure_ready()
        return self._dense_index

    def get_chunk_lookup(self) -> dict[str, ChunkRecord]:
        self.ensure_ready()
        return self._chunk_lookup

    def get_adjacent_chunks(
        self,
        chunk: ChunkRecord,
        *,
        before: int = 1,
        after: int = 1,
    ) -> list[ChunkRecord]:
        self.ensure_ready()
        manual_chunks = self._chunks_by_manual.get(chunk.manual_name, [])
        if not manual_chunks:
            return []
        neighbors: list[ChunkRecord] = []
        lower = chunk.order - max(before, 0)
        upper = chunk.order + max(after, 0)
        for candidate in manual_chunks:
            if lower <= candidate.order <= upper:
                neighbors.append(candidate)
        return neighbors

    def get_alias_lookup(self) -> dict[str, str]:
        self.ensure_ready()
        return self._alias_lookup

    def image_index(self) -> dict[str, ImageRecord]:
        self.ensure_ready()
        return {image.image_id: image for image in self._images}
