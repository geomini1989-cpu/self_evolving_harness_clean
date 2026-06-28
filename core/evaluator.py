import json
import re

import yaml


class TaskEvaluator:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.schema_keys = self.config["schema"].keys()

    def _extract_json_from_text(self, text):
        if not text:
            return None
        text = str(text).strip()
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
                return parsed
            except json.JSONDecodeError:
                continue
        return None

    def _calculate_list_f1(self, pred_list, gt_list):
        if not isinstance(pred_list, list) or not isinstance(gt_list, list):
            return 0.0
        set_pred = set(str(x).strip() for x in pred_list if str(x).strip())
        set_gt = set(str(x).strip() for x in gt_list if str(x).strip())
        if not set_pred and not set_gt:
            return 1.0
        if not set_pred or not set_gt:
            return 0.0

        intersection = len(set_pred & set_gt)
        precision = intersection / len(set_pred)
        recall = intersection / len(set_gt)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate(self, prediction_text, ground_truth_dict):
        result = {
            "is_valid_json": False,
            "exact_match": False,
            "f1_score": 0.0,
            "error_reason": "",
        }

        pred_dict = self._extract_json_from_text(prediction_text)
        if not pred_dict or not isinstance(pred_dict, dict):
            result["error_reason"] = "JSON parse failed or no valid object found."
            return result

        result["is_valid_json"] = True
        total_score = 0.0
        total_fields = len(self.schema_keys)
        errors = []

        for key in self.schema_keys:
            if key not in pred_dict:
                errors.append(f"Missing field: {key}")
                continue

            pred_val = pred_dict.get(key)
            gt_val = ground_truth_dict.get(key)

            if isinstance(gt_val, list):
                score = self._calculate_list_f1(pred_val, gt_val)
                total_score += score
                if score < 1.0:
                    errors.append(f"Low list F1 for {key}: {score:.2f}")
            elif key == "summary":
                set_p = set(str(pred_val))
                set_g = set(str(gt_val))
                overlap = len(set_p & set_g) / max(len(set_g), 1)
                if overlap > 0.5:
                    total_score += 1.0
                else:
                    errors.append(f"Summary mismatch: {pred_val} vs {gt_val}")
            else:
                if str(pred_val).strip() == str(gt_val).strip():
                    total_score += 1.0
                else:
                    errors.append(f"Field mismatch {key}: {pred_val} vs {gt_val}")

        result["f1_score"] = total_score / total_fields if total_fields > 0 else 0.0
        result["exact_match"] = len(errors) == 0
        if not result["exact_match"]:
            result["error_reason"] = " | ".join(errors)
        return result
