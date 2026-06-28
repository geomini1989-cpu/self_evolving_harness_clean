# core/evaluator.py
import json
import re

class TaskEvaluator:
    def __init__(self, config_path):
        """
        初始化评估器，加载任务配置以了解需要评估哪些字段。
        """
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.schema_keys = self.config['schema'].keys()

    def _extract_json_from_text(self, text):
        """从 LLM 的输出中提取 JSON 字符串（兼容带有 markdown 标记的输出）"""
        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    def _calculate_list_f1(self, pred_list, gt_list):
        """新增：计算实体列表的 F1 分数，避免顺序不同导致误判"""
        if not isinstance(pred_list, list) or not isinstance(gt_list, list):
            return 0.0
        set_pred, set_gt = set(str(x).strip() for x in pred_list), set(str(x).strip() for x in gt_list)
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
        """
        核心评估逻辑：计算 F1 Score 与完全匹配度 (Exact Match)
        """
        result = {
            "is_valid_json": False,
            "exact_match": False,
            "f1_score": 0.0,
            "error_reason": ""
        }

        # 1. 基础格式校验
        pred_dict = self._extract_json_from_text(prediction_text)
        if not pred_dict:
            result["error_reason"] = "JSON 格式解析失败或未找到有效 JSON。"
            return result
        
        result["is_valid_json"] = True

        # 2. 字段比对与 F1 计算
        total_score = 0.0
        total_fields = len(self.schema_keys)
        errors = []

        for key in self.schema_keys:
            if key not in pred_dict:
                errors.append(f"缺失关键字段: {key}")
                continue
            
            pred_val = pred_dict.get(key)
            gt_val = ground_truth_dict.get(key)

            if isinstance(gt_val, list):
                # 实体列表采用 F1 评估
                score = self._calculate_list_f1(pred_val, gt_val)
                total_score += score
                if score < 1.0:
                    errors.append(f"列表字段 '{key}' 匹配度低 (F1: {score:.2f})")
            elif key == 'summary':
                # 摘要文本采用词级/字级重合度近似评估
                set_p, set_g = set(str(pred_val)), set(str(gt_val))
                overlap = len(set_p & set_g) / max(len(set_g), 1)
                if overlap > 0.5:  
                    total_score += 1.0
                else:
                    errors.append(f"字段 '{key}' 语义偏差较大")
            else:
                # 核心意图和紧急程度必须严格匹配
                if str(pred_val).strip() == str(gt_val).strip():
                    total_score += 1.0
                else:
                    errors.append(f"字段 '{key}' 取值错误 (预测: {pred_val} vs 真实: {gt_val})")

        # 3. 计算最终指标
        result["f1_score"] = total_score / total_fields if total_fields > 0 else 0.0
        result["exact_match"] = (len(errors) == 0)

        if not result["exact_match"]:
            result["error_reason"] = " | ".join(errors)

        return result