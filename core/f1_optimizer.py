import copy
import json
import re


class F1PostProcessor:
    """SkillOpt-style execution policy for schema repair and confidence-gated correction.

    This keeps the base model frozen. It improves F1 by using the current skill router
    as a lightweight critic before evaluation.
    """

    VALID_INTENTS = ["退款纠纷", "物流投诉", "账号封禁", "系统Bug", "虚假宣传"]
    HIGH_URGENCY_INTENTS = {"物流投诉", "系统Bug", "账号封禁"}
    URGENT_KEYWORDS = ["马上", "立刻", "赶紧", "投诉", "封禁", "崩溃", "黑屏", "超时", "态度差"]
    ENTITY_PATTERNS = [
        re.compile(r"(?:订单号|单号|编号|id|ID)[:：]?\s*([A-Za-z0-9-]{5,})"),
        re.compile(r"\b\d{6,}\b"),
        re.compile(r"\d+(?:\.\d+)?\s*(?:元|块|人民币)"),
    ]

    def __init__(self, memory_bank, route_override_threshold=1):
        self.memory_bank = memory_bank
        self.route_override_threshold = route_override_threshold

    def optimize(self, input_text, prediction):
        raw_prediction = self._coerce_dict(prediction)
        optimized = copy.deepcopy(raw_prediction)
        route_scores = self.memory_bank.score_categories(input_text)
        routed_intent = route_scores[0][0] if route_scores else None
        routed_score = route_scores[0][1] if route_scores else 0

        predicted_intent = str(optimized.get("core_intent", "")).strip()
        if routed_intent and (predicted_intent not in self.VALID_INTENTS or routed_score >= self.route_override_threshold):
            optimized["core_intent"] = routed_intent
        elif predicted_intent not in self.VALID_INTENTS:
            optimized["core_intent"] = "退款纠纷"

        optimized["urgency_level"] = self._optimize_urgency(input_text, optimized.get("core_intent"))
        optimized["entities"] = self._merge_entities(input_text, optimized.get("entities"))
        optimized["summary"] = self._optimize_summary(optimized.get("summary"))
        return optimized

    def optimize_to_json(self, input_text, prediction):
        optimized = self.optimize(input_text, prediction)
        return json.dumps(optimized, ensure_ascii=False, separators=(",", ":"))

    def _coerce_dict(self, prediction):
        if isinstance(prediction, dict):
            return prediction
        if isinstance(prediction, str):
            try:
                parsed = json.loads(prediction)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    def _optimize_urgency(self, input_text, intent):
        if intent in self.HIGH_URGENCY_INTENTS:
            return "高"
        if any(keyword in input_text for keyword in self.URGENT_KEYWORDS):
            return "高"
        return "中"

    def _merge_entities(self, input_text, entities):
        merged = []
        if isinstance(entities, list):
            merged.extend(str(item).strip() for item in entities if str(item).strip())
        for pattern in self.ENTITY_PATTERNS:
            for match in pattern.findall(input_text):
                value = match if isinstance(match, str) else match[0]
                value = str(value).strip()
                if value and value not in merged:
                    merged.append(value)
        return merged

    def _optimize_summary(self, summary):
        summary_text = str(summary or "").strip()
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", summary_text)
        if 4 <= len(summary_text) <= 20 and len(cjk_chars) >= 2:
            return summary_text
        return "用户负面体验客诉处理"
