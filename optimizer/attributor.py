import json
import os
import re


class SkillAttributor:
    ROOT_CAUSE_TYPES = {
        "json_format_error": ["JSON", "parse", "格式", "valid object"],
        "schema_field_error": ["Missing field", "字段", "schema"],
        "intent_routing_error": ["core_intent", "意图", "intent"],
        "urgency_miscalibration": ["urgency_level", "紧急", "urgency"],
        "entity_extraction_error": ["entities", "实体", "order", "amount"],
        "context_missing": ["上下文", "context", "缺失"],
        "logic_conflict": ["冲突", "conflict", "override"],
        "domain_knowledge_gap": ["domain", "knowledge", "领域"],
    }

    def __init__(self, llm_client):
        self.llm = llm_client
        self.skill_file = "memory/SKILL.md"
        os.makedirs(os.path.dirname(self.skill_file), exist_ok=True)

    def analyze_root_cause(self, eval_result, current_input, prediction_text, ground_truth, max_retries=2, use_llm=True):
        if not use_llm:
            return self._rule_based_attribution(eval_result, current_input, prediction_text, ground_truth)

        base_prompt = f"""You are a diagnostic engine for a customer complaint parsing harness.
Analyze this bad case and return strict JSON only.

Input: {current_input}
Ground truth: {json.dumps(ground_truth, ensure_ascii=False, separators=(",", ":"))}
Prediction: {prediction_text}
Evaluation error: {eval_result.get("error_reason")}

Required JSON shape:
{{
  "error_category": "...",
  "root_cause_type": "one of json_format_error/schema_field_error/intent_routing_error/urgency_miscalibration/entity_extraction_error/context_missing/logic_conflict/domain_knowledge_gap",
  "evolution_action": "one of prompt_patch/skill_patch/few_shot_patch",
  "target_category": "...",
  "confidence": 0.0,
  "affected_fields": ["..."],
  "root_cause_analysis": "...",
  "proposed_rule": "...",
  "verification_plan": "...",
  "risk_flags": ["..."]
}}"""

        current_prompt = base_prompt
        for attempt in range(max_retries):
            suggestion = self.llm.generate(current_prompt, model_type="smart", max_tokens=500)
            try:
                patch_data = self._extract_json_object(suggestion)
                if patch_data.get("proposed_rule"):
                    patch_data.setdefault("target_category", ground_truth.get("core_intent"))
                    return self._normalize_patch(patch_data, eval_result, current_input, prediction_text, ground_truth)
                raise ValueError("missing proposed_rule")
            except Exception as exc:
                if attempt < max_retries - 1:
                    current_prompt = base_prompt + f"\n\nPrevious output was invalid: {exc}. Return JSON only. Output: {suggestion}"
                else:
                    return self._rule_based_attribution(eval_result, current_input, prediction_text, ground_truth)

    def _extract_json_object(self, text):
        cleaned = re.sub(r"`{3}(?:json)?(.*?)`{3}", r"\1", str(text), flags=re.DOTALL | re.IGNORECASE).strip()
        decoder = json.JSONDecoder()
        for idx, char in enumerate(cleaned):
            if char != "{":
                continue
            parsed, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("no JSON object found")

    def _rule_based_attribution(self, eval_result, current_input, prediction_text, ground_truth):
        target_category = ground_truth.get("core_intent", "\u9000\u6b3e\u7ea0\u7eb7")
        root_cause_type = self._classify_root_cause(eval_result)
        affected_fields = self._affected_fields(eval_result)
        evolution_action = self._select_evolution_action(root_cause_type, affected_fields)
        return {
            "error_category": "rule_engine_bad_case_attribution",
            "root_cause_type": root_cause_type,
            "evolution_action": evolution_action,
            "target_category": target_category,
            "confidence": self._confidence(root_cause_type, eval_result),
            "affected_fields": affected_fields,
            "root_cause_analysis": f"Prediction failed against {target_category}; local route evidence and ground truth should override uncertain model output.",
            "proposed_rule": f"When route evidence points to {target_category}, correct core_intent to {target_category}, normalize urgency/entities/summary, and keep the compact JSON schema.",
            "verification_plan": "Run category-matched bad-case regression, then replay stable examples across all categories before keeping the skill.",
            "risk_flags": self._risk_flags(root_cause_type, affected_fields),
            "source_error": eval_result.get("error_reason", ""),
            "input_excerpt": str(current_input)[:120],
            "example_input": str(current_input),
            "example_output": ground_truth,
        }

    def _normalize_patch(self, patch_data, eval_result, current_input, prediction_text, ground_truth):
        root_cause_type = patch_data.get("root_cause_type") or self._classify_root_cause(eval_result)
        affected_fields = patch_data.get("affected_fields") or self._affected_fields(eval_result)
        if not isinstance(affected_fields, list):
            affected_fields = [str(affected_fields)]
        patch_data["root_cause_type"] = root_cause_type
        selected_action = self._select_evolution_action(root_cause_type, affected_fields)
        requested_action = patch_data.get("evolution_action") or selected_action
        if requested_action == "prompt_patch" and root_cause_type not in {"json_format_error", "schema_field_error"}:
            requested_action = selected_action
            risk_flags = patch_data.get("risk_flags") if isinstance(patch_data.get("risk_flags"), list) else []
            if "llm_action_overridden" not in risk_flags:
                risk_flags.append("llm_action_overridden")
            patch_data["risk_flags"] = risk_flags
        patch_data["evolution_action"] = requested_action
        patch_data["target_category"] = patch_data.get("target_category") or ground_truth.get("core_intent")
        patch_data["confidence"] = float(patch_data.get("confidence") or self._confidence(root_cause_type, eval_result))
        patch_data["affected_fields"] = affected_fields
        patch_data.setdefault("verification_plan", "Run matched bad cases plus replay examples; rollback if F1 drops beyond tolerance.")
        patch_data.setdefault("risk_flags", self._risk_flags(root_cause_type, affected_fields))
        patch_data.setdefault("source_error", eval_result.get("error_reason", ""))
        patch_data.setdefault("input_excerpt", str(current_input)[:120])
        patch_data.setdefault("example_input", str(current_input))
        patch_data.setdefault("example_output", ground_truth)
        return patch_data

    def _select_evolution_action(self, root_cause_type, affected_fields):
        if root_cause_type in {"json_format_error", "schema_field_error"}:
            return "prompt_patch"
        if root_cause_type == "entity_extraction_error" or "entities" in affected_fields:
            return "few_shot_patch"
        return "skill_patch"

    def _classify_root_cause(self, eval_result):
        error_text = str(eval_result.get("error_reason", ""))
        if not eval_result.get("is_valid_json", True):
            return "json_format_error"
        for root_type, markers in self.ROOT_CAUSE_TYPES.items():
            if any(marker in error_text for marker in markers):
                return root_type
        return "domain_knowledge_gap"

    def _affected_fields(self, eval_result):
        error_text = str(eval_result.get("error_reason", ""))
        fields = []
        for field in ["core_intent", "urgency_level", "entities", "summary"]:
            if field in error_text:
                fields.append(field)
        if not fields and not eval_result.get("is_valid_json", True):
            fields = ["core_intent", "urgency_level", "entities", "summary"]
        return fields or ["core_intent"]

    def _confidence(self, root_cause_type, eval_result):
        if root_cause_type in {"intent_routing_error", "entity_extraction_error", "urgency_miscalibration", "json_format_error"}:
            return 0.85
        if eval_result.get("f1_score", 0.0) <= 0.5:
            return 0.75
        return 0.62

    def _risk_flags(self, root_cause_type, affected_fields):
        flags = []
        if root_cause_type in {"logic_conflict", "domain_knowledge_gap", "context_missing"}:
            flags.append("needs_broader_replay")
        if "core_intent" in affected_fields:
            flags.append("can_affect_routing")
        if len(affected_fields) >= 3:
            flags.append("broad_schema_change")
        return flags
