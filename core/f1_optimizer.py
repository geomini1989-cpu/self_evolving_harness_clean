import copy
import json
import re


class F1PostProcessor:
    """SkillOpt-style execution policy for schema repair and confidence-gated correction."""

    REFUND = "\u9000\u6b3e\u7ea0\u7eb7"
    LOGISTICS = "\u7269\u6d41\u6295\u8bc9"
    ACCOUNT = "\u8d26\u53f7\u5c01\u7981"
    SYSTEM_BUG = "\u7cfb\u7edfBug"
    FALSE_AD = "\u865a\u5047\u5ba3\u4f20"
    HIGH = "\u9ad8"
    MEDIUM = "\u4e2d"

    VALID_INTENTS = [REFUND, LOGISTICS, ACCOUNT, SYSTEM_BUG, FALSE_AD]
    HIGH_URGENCY_INTENTS = {LOGISTICS, SYSTEM_BUG, ACCOUNT}
    URGENT_KEYWORDS = ["\u9a6c\u4e0a", "\u7acb\u523b", "\u8d76\u7d27", "\u6295\u8bc9", "\u5c01\u7981", "\u5d29\u6e83", "\u9ed1\u5c4f", "\u8d85\u65f6", "\u6001\u5ea6\u5dee"]
    ENTITY_PATTERNS = [
        re.compile(r"(?:\u8ba2\u5355\u53f7|\u5355\u53f7|\u7f16\u53f7|id|ID)[:\uff1a]?\s*([A-Za-z0-9-]{5,})"),
        re.compile(r"\b\d{6,}\b"),
        re.compile(r"\d+(?:\.\d+)?\s*(?:\u5143|\u5757|\u4eba\u6c11\u5e01)"),
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
            optimized["core_intent"] = self.REFUND

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
            return self.HIGH
        if any(keyword in input_text for keyword in self.URGENT_KEYWORDS):
            return self.HIGH
        return self.MEDIUM

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
        return "\u7528\u6237\u8d1f\u9762\u4f53\u9a8c\u5ba2\u8bc9\u5904\u7406"
