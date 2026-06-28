import concurrent.futures
import json
import os
import re
import shutil


class SkillEvolver:
    def __init__(self, evaluator, llm_client):
        self.evaluator = evaluator
        self.llm = llm_client
        self.skill_file = "memory/SKILL.md"
        self.backup_file = "memory/SKILL_backup.md"
        self.valid_categories = ["\u9000\u6b3e\u7ea0\u7eb7", "\u7269\u6d41\u6295\u8bc9", "\u8d26\u53f7\u5c01\u7981", "\u7cfb\u7edfBug", "\u865a\u5047\u5ba3\u4f20"]

    def _build_evolve_prompt(self, root_cause_analysis):
        analysis_text = json.dumps(root_cause_analysis, ensure_ascii=False, separators=(",", ":")) if isinstance(root_cause_analysis, dict) else str(root_cause_analysis)
        return f"""You write compact business rules for a customer complaint extraction system.
Return only one Markdown rule block in this exact shape:
## [one category] short title
Trigger: ...
Action: ...

Allowed categories: {self.valid_categories}
Root cause JSON: {analysis_text}
"""

    def _safe_write_skill_to_md(self, new_skill_text):
        clean_text = new_skill_text.strip()
        pattern = re.compile(r"^##\s*\[?([^\s\]]+)\]?\s+(.*?)\n(.*)", re.DOTALL)
        match = pattern.match(clean_text)
        if not match:
            print(f"[Evolver] Rejected invalid skill format: {clean_text[:80]}...")
            return False
        category = match.group(1).strip()
        if category not in self.valid_categories:
            print(f"[Evolver] Rejected unknown category: {category}")
            return False
        if os.path.exists(self.skill_file):
            shutil.copy(self.skill_file, self.backup_file)
        os.makedirs(os.path.dirname(self.skill_file), exist_ok=True)
        with open(self.skill_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{clean_text}\n")
        print(f"[Evolver] Added new skill for {category}.")
        return True

    def _chunk_dataset(self, dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]

    def _score_dataset(self, dataset, config, build_prompt_func, skills, batch_size=8, max_workers=6, use_cache=True):
        if not dataset:
            return 0.0
        total_f1 = 0.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {}
            for batch_data in self._chunk_dataset(dataset, batch_size):
                batch_inputs = [data["input"] for data in batch_data]
                prompt = build_prompt_func(config, skills, few_shots="", batch_texts=batch_inputs, meta_intervention=False)
                future = executor.submit(self.llm.generate, prompt, temperature=0.0, model_type="default", use_cache=use_cache, max_tokens=max(500, 220 * len(batch_inputs)))
                future_to_batch[future] = batch_data
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_data = future_to_batch[future]
                try:
                    prediction_text = future.result()
                    parsed_array = self.evaluator._extract_json_from_text(prediction_text)
                    if not isinstance(parsed_array, list) or len(parsed_array) != len(batch_data):
                        continue
                    for idx, data in enumerate(batch_data):
                        prediction = json.dumps(parsed_array[idx], ensure_ascii=False, separators=(",", ":"))
                        eval_result = self.evaluator.evaluate(prediction, data["ground_truth"])
                        total_f1 += eval_result["f1_score"]
                except Exception as exc:
                    print(f"[Evolver] Regression batch failed: {exc}")
        return total_f1 / len(dataset)

    def _sample_regression_set(self, patch_data, golden_set, limit=10):
        text = json.dumps(patch_data, ensure_ascii=False)
        categories = [category for category in self.valid_categories if category in text]
        matched = []
        for item in golden_set:
            gt_intent = item.get("ground_truth", {}).get("core_intent")
            if gt_intent in categories:
                matched.append(item)
        if not matched:
            matched = golden_set[:]
        return matched[:limit]

    def apply_patch_with_rollback(self, attributor, patch_data, config, golden_set, build_prompt_func, baseline_f1):
        if not patch_data or not patch_data.get("proposed_rule"):
            print("[Evolver] Invalid patch data; skipping this evolution step.")
            return baseline_f1, False
        print("[Evolver] Generating a compact skill patch...")
        evolve_prompt = self._build_evolve_prompt(patch_data)
        new_skill_text = self.llm.generate(evolve_prompt, temperature=0.2, model_type="smart", max_tokens=500)
        if not self._safe_write_skill_to_md(new_skill_text):
            return baseline_f1, False
        with open(self.skill_file, "r", encoding="utf-8") as f:
            updated_skills = f.read()
        evolution_cfg = config.get("evolution", {})
        runtime_cfg = config.get("runtime", {})
        regression_mode = evolution_cfg.get("regression_mode", "sample_then_full")
        threshold = float(evolution_cfg.get("f1_tolerance", 0.02))
        batch_size = int(runtime_cfg.get("batch_size", 8))
        max_workers = int(runtime_cfg.get("max_workers", 6))
        sample_set = self._sample_regression_set(patch_data, golden_set, limit=int(evolution_cfg.get("sample_size", 10)))
        sample_f1 = self._score_dataset(sample_set, config, build_prompt_func, updated_skills, batch_size=batch_size, max_workers=max_workers, use_cache=True)
        print(f"[Evolver] Sample regression F1: {sample_f1:.2f}; baseline: {baseline_f1:.2f}")
        if sample_f1 + threshold < baseline_f1:
            if os.path.exists(self.backup_file):
                shutil.copy(self.backup_file, self.skill_file)
            return baseline_f1, False
        if regression_mode == "full" or abs(sample_f1 - baseline_f1) <= threshold:
            new_avg_f1 = self._score_dataset(golden_set, config, build_prompt_func, updated_skills, batch_size=batch_size, max_workers=max_workers, use_cache=True)
            print(f"[Evolver] Full regression F1: {new_avg_f1:.2f}; baseline: {baseline_f1:.2f}")
            if new_avg_f1 + threshold < baseline_f1:
                if os.path.exists(self.backup_file):
                    shutil.copy(self.backup_file, self.skill_file)
                return baseline_f1, False
            return new_avg_f1, True
        return sample_f1, True
