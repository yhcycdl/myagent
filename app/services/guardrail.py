from __future__ import annotations

import re

from app.services.generator import GeneratedAnswer, _looks_english_dominant_text


class GuardrailService:
    def review(self, generated: GeneratedAnswer) -> GeneratedAnswer:
        answer = generated.answer.strip()
        english_answer = _looks_english_dominant_text(answer) and not answer.startswith("您好")
        if generated.references and not answer.startswith("您好") and not english_answer:
            if re.match(r"^\d+[\.\)、]", answer):
                answer = f"您好，\n{answer}"
            else:
                answer = f"您好，{answer}"
        support_terms = (
            "订单号",
            "发票",
            "退款",
            "退货",
            "换货",
            "补发",
            "投诉",
            "物流",
            "快递",
            "售后",
            "保修",
            "运费",
            "优惠券",
            "试用",
            "商品名称",
        )
        lower_answer = answer.lower()
        has_suggestion = (
            "建议" in answer
            or "请提供" in answer
            or "可以提供" in answer
            or "please provide" in lower_answer
            or "please share" in lower_answer
        )
        high_confidence_direct = generated.confidence >= 0.95 and not generated.references
        if (
            not generated.references
            and not generated.related_images
            and not high_confidence_direct
            and not has_suggestion
            and not any(term in answer for term in support_terms)
        ):
            answer = f"{answer}\n建议补充更具体的信息后再确认。"
        generated.answer = answer
        return generated
