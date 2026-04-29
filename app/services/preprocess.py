from __future__ import annotations

import re
from typing import Iterable


FOLLOW_UP_HINTS = (
    "这个",
    "这个灯",
    "这个按钮",
    "它",
    "那",
    "上面",
    "刚才",
    "上一轮",
    "前面",
    "if so",
    "this feature",
    "this function",
    "that feature",
    "that function",
    "it ",
    "them",
)
CONSTRAINT_PREFIXES = ("只需", "只要", "只告诉", "仅需", "仅告诉", "麻烦只", "其中", "并且只")
CONSTRAINT_HINTS = ("前五条", "前5条", "前两个步骤", "前六个步骤", "最后三个步骤", "最后三步")
SUPPORT_KEYWORDS = (
    "退款",
    "退货",
    "换货",
    "更换",
    "发票",
    "投诉",
    "赔偿",
    "物流",
    "快递",
    "售后",
    "维修",
    "检修",
    "补发",
    "订单",
    "运费",
    "签收",
    "到货",
    "发货",
    "优惠券",
    "上门安装",
    "上门检修",
    "仓库维修",
    "安装服务",
    "智能客服",
    "客服",
    "包装破损",
    "少发",
    "漏发",
    "丢件",
    "丢失",
    "尺寸差价",
    "试用",
    "保质期",
    "临期",
    "过期",
    "以旧换新",
    "旧换新",
    "生产日期",
    "出厂日期",
    "制造日期",
    "生产批号",
    "虚假宣传",
    "宣传功能",
    "实物不符",
    "颜色偏差",
    "少件",
    "缺件",
)
FORCE_SUPPORT_HINTS = (
    "优惠券",
    "上门安装",
    "上门检修",
    "安装服务",
    "仓库维修",
    "检修人员",
    "智能客服",
    "客服解答",
    "人工客服",
    "尺寸差价",
    "更大的尺寸",
    "更换成更大",
    "试用期间",
    "保质期",
    "临期",
    "过期",
    "以旧换新",
    "旧换新",
    "生产日期",
    "出厂日期",
    "制造日期",
    "生产批号",
    "虚假宣传",
    "宣传功能",
    "实物不符",
    "颜色偏差",
    "少件",
    "缺件",
    "快递丢失",
    "物流丢失",
    "包装破损",
    "售后保障",
    "保修卡",
    "保障卡",
)
MANUAL_STYLE_HINTS = (
    "手册",
    "说明书",
    "如何",
    "步骤",
    "按钮",
    "模式",
    "设置",
    "安装",
    "拆卸",
    "清洁",
    "保养",
    "启动",
    "关闭",
    "滤网",
    "电池",
    "遥控器",
    "型号",
)


def normalize_question(question: str) -> str:
    question = question.replace("\r\n", "\n").replace("\r", "\n")
    question = re.sub(r"[ \t]+", " ", question)
    question = re.sub(r"\n{3,}", "\n\n", question)
    return question.strip().strip('"')


def split_sub_questions(question: str) -> list[str]:
    question = normalize_question(question)
    if not question:
        return []

    segments = re.split(r"[\n]+|(?<=[？?])", question)
    parts: list[str] = []
    for segment in segments:
        candidate = segment.strip(" ，,。；;！!？?\"'")
        if candidate:
            parts.append(candidate)

    unique_parts: list[str] = []
    seen: set[str] = set()
    for part in parts or [question]:
        if unique_parts and _should_attach_to_previous(part):
            merged = f"{unique_parts[-1]}，{part}"
            unique_parts[-1] = merged
            seen.discard(part)
            seen.add(merged)
            continue
        if part not in seen:
            seen.add(part)
            unique_parts.append(part)
    return unique_parts


def looks_like_follow_up(question: str) -> bool:
    return any(hint in question for hint in FOLLOW_UP_HINTS) or len(question) <= 14


def is_general_support_question(question: str) -> bool:
    if any(keyword in question for keyword in FORCE_SUPPORT_HINTS):
        return True
    has_support_keyword = any(keyword in question for keyword in SUPPORT_KEYWORDS)
    has_manual_hint = any(keyword in question for keyword in MANUAL_STYLE_HINTS)
    return has_support_keyword and not has_manual_hint


def is_manual_access_question(question: str) -> bool:
    access_terms = ("纸质版", "电子版", "说明书下载", "在哪里可以找到", "在哪下载", "哪里下载")
    return any(term in question for term in access_terms)


def looks_english_dominant(question: str) -> bool:
    ascii_letters = sum(char.isascii() and char.isalpha() for char in question)
    chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in question)
    english_terms = re.findall(r"[A-Za-z]{3,}(?:[- ][A-Za-z0-9]{2,})*", question)
    return ascii_letters > chinese_chars or len(english_terms) >= 3


def rewrite_with_context(
    question: str,
    product_hint: str | None,
    section_hints: Iterable[str] | None = None,
) -> str:
    if not product_hint:
        return question

    hint_blob = " ".join(section_hints or [])
    if looks_like_follow_up(question):
        return f"关于{product_hint}，{hint_blob} {question}".strip()
    return question


def _should_attach_to_previous(segment: str) -> bool:
    lowered = segment.lower()
    english_follow_up_prefixes = (
        "if so",
        "if not",
        "if no",
        "if yes",
        "then how",
        "how do i operate",
        "how do i use this",
        "how can i operate",
        "how can i use this",
        "what about this",
    )
    english_follow_up_hints = (
        "this feature",
        "this function",
        "that feature",
        "that function",
        "operate it",
        "use it",
        "turn it",
        "set it",
    )
    return (
        segment.startswith(CONSTRAINT_PREFIXES)
        or any(hint in segment for hint in CONSTRAINT_HINTS)
        or lowered.startswith(english_follow_up_prefixes)
        or any(hint in lowered for hint in english_follow_up_hints)
    )
