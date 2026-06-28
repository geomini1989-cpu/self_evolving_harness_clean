import json
import os
import re


class SkillAttributor:
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
  "target_category": "...",
  "root_cause_analysis": "...",
  "proposed_rule": "..."
}}"""

        current_prompt = base_prompt
        for attempt in range(max_retries):
            suggestion = self.llm.generate(current_prompt, model_type="smart", max_tokens=500)
            try:
                patch_data = self._extract_json_object(suggestion)
                if patch_data.get("proposed_rule"):
                    patch_data.setdefault("target_category", ground_truth.get("core_intent"))
                    return patch_data
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
        return {
            "error_category": "rule_engine_bad_case_attribution",
            "target_category": target_category,
            "root_cause_analysis": f"Prediction failed against {target_category}; local route evidence and ground truth should override uncertain model output.",
            "proposed_rule": f"When route evidence points to {target_category}, correct core_intent to {target_category}, normalize urgency/entities/summary, and keep the compact JSON schema.",
            "source_error": eval_result.get("error_reason", ""),
            "input_excerpt": str(current_input)[:120],
        }
