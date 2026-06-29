import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import time

from optimizer.tip_memory import TipMemory


class SkillEvolver:
    def __init__(self, evaluator, llm_client):
        self.evaluator = evaluator
        self.llm = llm_client
        self.skill_file = "memory/SKILL.md"
        self.backup_file = "memory/SKILL_backup.md"
        self.prompt_policy_file = "memory/PROMPT_POLICY.md"
        self.prompt_policy_backup_file = "memory/PROMPT_POLICY_backup.md"
        self.examples_file = "memory/examples.json"
        self.examples_backup_file = "memory/examples_backup.json"
        self.version_log_file = "memory/skill_versions.jsonl"
        self.rejected_file = "memory/rejected_skills.jsonl"
        self.tip_memory = TipMemory()
        self.valid_categories = ["\u9000\u6b3e\u7ea0\u7eb7", "\u7269\u6d41\u6295\u8bc9", "\u8d26\u53f7\u5c01\u7981", "\u7cfb\u7edfBug", "\u865a\u5047\u5ba3\u4f20"]
        self.valid_actions = {"skill_patch", "prompt_patch", "few_shot_patch"}

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

    def _build_rule_skill(self, patch_data):
        category = patch_data.get("target_category") or self._infer_category(patch_data)
        if category not in self.valid_categories:
            category = self.valid_categories[0]
        title = "auto evolved bad-case correction"
        trigger = patch_data.get("input_excerpt") or patch_data.get("root_cause_analysis") or "matched local route evidence"
        action = patch_data.get("proposed_rule") or f"Correct core_intent to {category} and keep compact JSON output."
        verification = patch_data.get("verification_plan") or "Replay matched bad cases before accepting this skill."
        return f"""## [{category}] {title}
Trigger: {trigger}
Action: {action}
Verification: {verification}"""

    def _build_prompt_policy(self, patch_data):
        rule = patch_data.get("proposed_rule") or "Keep output compact, valid, and schema-compliant."
        root_type = patch_data.get("root_cause_type") or "unknown"
        return f"""## Prompt policy patch: {root_type}
Trigger: {patch_data.get("source_error") or patch_data.get("input_excerpt") or "schema/output instability"}
Instruction: {rule}
Output contract: Return compact JSON only and preserve the required schema exactly."""

    def _build_few_shot_example(self, patch_data):
        example_input = patch_data.get("example_input") or patch_data.get("input_excerpt") or ""
        example_output = patch_data.get("example_output") or {
            "core_intent": patch_data.get("target_category") or self.valid_categories[0],
            "urgency_level": "\u4e2d",
            "entities": [],
            "summary": "\u7528\u6237\u8d1f\u9762\u4f53\u9a8c\u5ba2\u8bc9\u5904\u7406",
        }
        return {"input": example_input, "output": example_output, "source": "evolution_action_selector"}

    def _bounded_skill_text(self, skill_text, config):
        max_chars = int(config.get("evolution", {}).get("max_new_skill_chars", 900))
        clean_text = skill_text.strip()
        if len(clean_text) <= max_chars:
            return clean_text
        lines = clean_text.splitlines()
        kept = []
        total = 0
        for line in lines:
            if total + len(line) + 1 > max_chars:
                break
            kept.append(line)
            total += len(line) + 1
        return "\n".join(kept).rstrip() + "\nNote: truncated by textual learning-rate budget."

    def _infer_category(self, patch_data):
        text = json.dumps(patch_data, ensure_ascii=False)
        for category in self.valid_categories:
            if category in text:
                return category
        return None

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

    def _safe_write_prompt_policy(self, policy_text):
        clean_text = policy_text.strip()
        if not clean_text.startswith("## Prompt policy patch:"):
            print(f"[Evolver] Rejected invalid prompt policy format: {clean_text[:80]}...")
            return False
        if os.path.exists(self.prompt_policy_file):
            shutil.copy(self.prompt_policy_file, self.prompt_policy_backup_file)
        os.makedirs(os.path.dirname(self.prompt_policy_file), exist_ok=True)
        with open(self.prompt_policy_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{clean_text}\n")
        print("[Evolver] Added prompt policy patch.")
        return True

    def _safe_write_few_shot(self, example):
        os.makedirs(os.path.dirname(self.examples_file), exist_ok=True)
        examples = []
        if os.path.exists(self.examples_file):
            shutil.copy(self.examples_file, self.examples_backup_file)
            try:
                with open(self.examples_file, "r", encoding="utf-8") as f:
                    examples = json.load(f)
            except Exception:
                examples = []
        if any(item.get("input") == example.get("input") for item in examples):
            print("[Evolver] Few-shot example already exists; keeping current memory.")
            return True
        examples.append(example)
        with open(self.examples_file, "w", encoding="utf-8") as f:
            json.dump(examples[-100:], f, ensure_ascii=False, indent=2)
        print("[Evolver] Added few-shot example.")
        return True

    def _skill_hash(self, text):
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _is_rejected_before(self, new_skill_text):
        if not os.path.exists(self.rejected_file):
            return False
        target_hash = self._skill_hash(new_skill_text)
        try:
            with open(self.rejected_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("skill_hash") == target_hash:
                        return True
        except Exception:
            return False
        return False

    def _append_rejected_skill(self, reason, skill_text, patch_data, metrics=None):
        os.makedirs(os.path.dirname(self.rejected_file), exist_ok=True)
        record = {
            "ts": int(time.time()),
            "reason": reason,
            "skill_hash": self._skill_hash(skill_text),
            "target_category": patch_data.get("target_category"),
            "root_cause_type": patch_data.get("root_cause_type"),
            "confidence": patch_data.get("confidence"),
            "metrics": metrics or {},
            "skill_preview": skill_text[:300],
        }
        with open(self.rejected_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _file_sha256(self, path):
        if not os.path.exists(path):
            return "missing"
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _read_text_file(self, path, default=""):
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _append_version_log(self, event, patch_data, metrics=None):
        os.makedirs(os.path.dirname(self.version_log_file), exist_ok=True)
        record = {
            "ts": int(time.time()),
            "event": event,
            "skill_sha256": self._file_sha256(self.skill_file),
            "backup_sha256": self._file_sha256(self.backup_file),
            "target_category": patch_data.get("target_category"),
            "root_cause_type": patch_data.get("root_cause_type"),
            "evolution_action": patch_data.get("evolution_action"),
            "confidence": patch_data.get("confidence"),
            "tip_id": patch_data.get("tip_id"),
            "tip_count": patch_data.get("tip_count"),
            "affected_fields": patch_data.get("affected_fields", []),
            "risk_flags": patch_data.get("risk_flags", []),
            "metrics": metrics or {},
        }
        with open(self.version_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _passes_confidence_gate(self, patch_data, config):
        evolution_cfg = config.get("evolution", {})
        threshold = float(evolution_cfg.get("confidence_threshold", 0.7))
        confidence = float(patch_data.get("confidence") or 0.0)
        if confidence < threshold:
            print(f"[Evolver] Rejected low-confidence patch: {confidence:.2f} < {threshold:.2f}")
            self._append_version_log("rejected_low_confidence", patch_data, {"confidence_threshold": threshold})
            return False
        risk_flags = set(patch_data.get("risk_flags") or [])
        if "broad_schema_change" in risk_flags and confidence < max(0.85, threshold):
            print("[Evolver] Rejected broad schema patch without high confidence.")
            self._append_version_log("rejected_high_risk", patch_data, {"confidence_threshold": max(0.85, threshold)})
            return False
        return True

    def _chunk_dataset(self, dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]

    def _score_dataset(self, dataset, config, build_prompt_func, skills, batch_size=8, max_workers=6, use_cache=True, few_shots=""):
        if not dataset:
            return 0.0
        total_f1 = 0.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {}
            for batch_data in self._chunk_dataset(dataset, batch_size):
                batch_inputs = [data["input"] for data in batch_data]
                prompt = build_prompt_func(config, skills, few_shots=few_shots, batch_texts=batch_inputs, meta_intervention=False)
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

    def _load_replay_examples(self, limit=20):
        if not os.path.exists(self.examples_file):
            return []
        try:
            with open(self.examples_file, "r", encoding="utf-8") as f:
                examples = json.load(f)
        except Exception:
            return []
        replay = []
        for item in examples[-limit:]:
            output = item.get("output")
            if isinstance(output, dict):
                replay.append({"input": item.get("input", ""), "ground_truth": output})
        return replay

    def _load_few_shot_prompt(self, limit=1, max_chars=1000):
        if not os.path.exists(self.examples_file):
            return ""
        try:
            with open(self.examples_file, "r", encoding="utf-8") as f:
                examples = json.load(f)
        except Exception:
            return ""
        lines = ["Reference successful cases:"]
        for idx, item in enumerate(examples[-limit:], 1):
            output = json.dumps(item.get("output", {}), ensure_ascii=False, separators=(",", ":"))
            lines.append(f"{idx}. input={item.get('input', '')}\noutput={output}")
        prompt = "\n".join(lines)
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars].rstrip() + "\n[truncated]"
        return prompt

    def _balanced_category_replay(self, golden_set, per_category=2):
        selected = []
        counts = {category: 0 for category in self.valid_categories}
        for item in golden_set:
            category = item.get("ground_truth", {}).get("core_intent")
            if category in counts and counts[category] < per_category:
                selected.append(item)
                counts[category] += 1
            if all(count >= per_category for count in counts.values()):
                break
        return selected

    def _build_replay_set(self, patch_data, golden_set, sample_set, config):
        evolution_cfg = config.get("evolution", {})
        replay_limit = int(evolution_cfg.get("replay_size", 20))
        replay_set = []
        replay_set.extend(sample_set)
        replay_set.extend(self._balanced_category_replay(golden_set, per_category=2))
        replay_set.extend(self._load_replay_examples(limit=replay_limit))
        if "needs_broader_replay" in set(patch_data.get("risk_flags") or []):
            replay_set.extend(golden_set[:replay_limit])
        deduped = []
        seen_inputs = set()
        for item in replay_set:
            text = item.get("input", "")
            if text and text not in seen_inputs:
                seen_inputs.add(text)
                deduped.append(item)
        return deduped[: max(replay_limit, len(sample_set))]

    def _rollback(self):
        if os.path.exists(self.backup_file):
            shutil.copy(self.backup_file, self.skill_file)
        elif os.path.exists(self.skill_file):
            os.remove(self.skill_file)

    def _artifact_path(self, action):
        if action == "prompt_patch":
            return self.prompt_policy_file
        if action == "few_shot_patch":
            return self.examples_file
        return self.skill_file

    def _snapshot_artifact(self, action):
        path = self._artifact_path(action)
        if not os.path.exists(path):
            return {"path": path, "exists": False, "content": ""}
        return {"path": path, "exists": True, "content": self._read_text_file(path)}

    def _rollback_artifact(self, action, snapshot=None):
        if snapshot:
            path = snapshot["path"]
            if snapshot["exists"]:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(snapshot["content"])
            elif os.path.exists(path):
                os.remove(path)
            print(f"[Evolver] Rolled back {action}; restored previous artifact state.")
            return
        if action == "prompt_patch":
            if os.path.exists(self.prompt_policy_backup_file):
                shutil.copy(self.prompt_policy_backup_file, self.prompt_policy_file)
            elif os.path.exists(self.prompt_policy_file):
                os.remove(self.prompt_policy_file)
        elif action == "few_shot_patch":
            if os.path.exists(self.examples_backup_file):
                shutil.copy(self.examples_backup_file, self.examples_file)
            elif os.path.exists(self.examples_file):
                os.remove(self.examples_file)
        else:
            self._rollback()

    def _build_action_artifact(self, action, patch_data, config, use_rule_evolver):
        if action == "prompt_patch":
            return self._bounded_skill_text(self._build_prompt_policy(patch_data), config)
        if action == "few_shot_patch":
            example = self._build_few_shot_example(patch_data)
            return json.dumps(example, ensure_ascii=False, separators=(",", ":"))
        if use_rule_evolver:
            text = self._build_rule_skill(patch_data)
        else:
            evolve_prompt = self._build_evolve_prompt(patch_data)
            text = self.llm.generate(evolve_prompt, temperature=0.2, model_type="smart", max_tokens=500)
        return self._bounded_skill_text(text, config)

    def _write_action_artifact(self, action, artifact_text):
        if action == "prompt_patch":
            return self._safe_write_prompt_policy(artifact_text)
        if action == "few_shot_patch":
            return self._safe_write_few_shot(json.loads(artifact_text))
        return self._safe_write_skill_to_md(artifact_text)

    def _curate_skill_repo(self, config):
        if not os.path.exists(self.skill_file):
            return
        max_per_category = int(config.get("evolution", {}).get("max_skills_per_category", 5))
        with open(self.skill_file, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = re.compile(r"(##\s*\[(.*?)\]\s*.*?)(?=\n##|\Z)", re.DOTALL)
        grouped = {category: [] for category in self.valid_categories}
        for block, category in pattern.findall(content):
            category = category.strip()
            if category not in grouped:
                continue
            block = block.strip()
            block_hash = self._skill_hash(block)
            if any(existing[0] == block_hash for existing in grouped[category]):
                continue
            grouped[category].append((block_hash, block))
        curated_blocks = []
        for category in self.valid_categories:
            curated_blocks.extend(block for _, block in grouped[category][-max_per_category:])
        if not curated_blocks:
            return
        curated = "# Self-Evolving Skill Repository\n\n" + "\n\n".join(curated_blocks).strip() + "\n"
        if curated.strip() == content.strip():
            return
        shutil.copy(self.skill_file, self.backup_file)
        with open(self.skill_file, "w", encoding="utf-8") as f:
            f.write(curated)
        self._append_version_log("curated_skill_repo", {}, {"max_skills_per_category": max_per_category, "skill_count": len(curated_blocks)})

    def apply_patch_with_rollback(self, attributor, patch_data, config, golden_set, build_prompt_func, baseline_f1, local_score_func=None):
        if not patch_data or not patch_data.get("proposed_rule"):
            print("[Evolver] Invalid patch data; skipping this evolution step.")
            return baseline_f1, False
        if not self._passes_confidence_gate(patch_data, config):
            return baseline_f1, False

        action = patch_data.get("evolution_action") or "skill_patch"
        if action not in self.valid_actions:
            action = "skill_patch"
        patch_data["evolution_action"] = action

        evolution_cfg = config.get("evolution", {})
        if bool(evolution_cfg.get("enable_tip_memory", True)):
            tip = self.tip_memory.add_or_update(patch_data)
            patch_data["tip_id"] = tip.get("tip_id")
            patch_data["tip_count"] = tip.get("count")
            if tip.get("status") == "promoted":
                print(f"[Evolver] Tip {tip.get('tip_id')} is already promoted; skipping duplicate long-term write.")
                self._append_version_log("tip_already_promoted", patch_data, {"tip_count": tip.get("count")})
                return baseline_f1, False
            tip_promoted_by_policy = self.tip_memory.should_promote(tip, config)
            if not tip_promoted_by_policy:
                if action == "few_shot_patch" and bool(evolution_cfg.get("allow_immediate_few_shot", False)):
                    print(f"[Evolver] Trying low-risk few-shot patch from buffered tip {tip.get('tip_id')} ({tip.get('count')} hit).")
                    self._append_version_log("tip_try_few_shot", patch_data, {"tip_count": tip.get("count")})
                else:
                    print(f"[Evolver] Buffered transient tip {tip.get('tip_id')} ({tip.get('count')} hit); waiting for repeat or higher confidence.")
                    self._append_version_log("tip_buffered", patch_data, {"promotion_threshold": int(evolution_cfg.get("tip_promotion_threshold", 2))})
                    return baseline_f1, False
            else:
                print(f"[Evolver] Promoting tip {tip.get('tip_id')} from short-term memory ({tip.get('count')} hit).")
                self._append_version_log("tip_promoted", patch_data, {"tip_count": tip.get("count")})

        print(f"[Evolver] Generating a compact {action}...")
        runtime_cfg = config.get("runtime", {})
        use_rule_evolver = bool(runtime_cfg.get("use_rule_evolver", False))
        artifact_text = self._build_action_artifact(action, patch_data, config, use_rule_evolver)
        if self._is_rejected_before(artifact_text):
            print("[Evolver] Rejected repeated candidate from rejected-edit buffer.")
            self._append_version_log("rejected_repeated_candidate", patch_data, {})
            return baseline_f1, False
        artifact_snapshot = self._snapshot_artifact(action)
        written = self._write_action_artifact(action, artifact_text)
        if not written:
            self._append_rejected_skill("invalid_action_artifact", artifact_text, patch_data)
            if patch_data.get("tip_id"):
                self.tip_memory.mark_rejected(patch_data["tip_id"], "invalid_action_artifact")
            return baseline_f1, False
        self._append_version_log("candidate_written", patch_data, {"baseline_f1": baseline_f1})

        updated_skills = self._read_text_file(self.skill_file)
        regression_mode = evolution_cfg.get("regression_mode", "sample_then_full")
        threshold = float(evolution_cfg.get("f1_tolerance", 0.02))
        batch_size = int(runtime_cfg.get("batch_size", 8))
        max_workers = int(runtime_cfg.get("max_workers", 6))
        regression_few_shots = self._load_few_shot_prompt(limit=1) if action == "few_shot_patch" else ""

        sample_set = self._sample_regression_set(patch_data, golden_set, limit=int(evolution_cfg.get("sample_size", 10)))
        sample_f1 = local_score_func(sample_set) if local_score_func else self._score_dataset(sample_set, config, build_prompt_func, updated_skills, batch_size=batch_size, max_workers=max_workers, use_cache=True, few_shots=regression_few_shots)
        print(f"[Evolver] Sample regression F1: {sample_f1:.2f}; baseline: {baseline_f1:.2f}")
        if sample_f1 + threshold < baseline_f1:
            print(f"[Evolver] Rolling back {action}: sample regression drop {sample_f1:.2f} < {baseline_f1:.2f} - tolerance {threshold:.2f}.")
            self._rollback_artifact(action, artifact_snapshot)
            self._append_rejected_skill("sample_regression_drop", artifact_text, patch_data, {"baseline_f1": baseline_f1, "sample_f1": sample_f1})
            if patch_data.get("tip_id"):
                self.tip_memory.mark_rejected(patch_data["tip_id"], "sample_regression_drop", {"baseline_f1": baseline_f1, "sample_f1": sample_f1})
            self._append_version_log("rolled_back_sample_regression", patch_data, {"baseline_f1": baseline_f1, "sample_f1": sample_f1})
            return baseline_f1, False

        replay_set = self._build_replay_set(patch_data, golden_set, sample_set, config)
        replay_f1 = local_score_func(replay_set) if local_score_func else self._score_dataset(replay_set, config, build_prompt_func, updated_skills, batch_size=batch_size, max_workers=max_workers, use_cache=True, few_shots=regression_few_shots)
        print(f"[Evolver] Replay regression F1: {replay_f1:.2f}; baseline: {baseline_f1:.2f}; replay_size={len(replay_set)}")
        if replay_f1 + threshold < baseline_f1:
            print(f"[Evolver] Rolling back {action}: replay regression drop {replay_f1:.2f} < {baseline_f1:.2f} - tolerance {threshold:.2f}.")
            self._rollback_artifact(action, artifact_snapshot)
            self._append_rejected_skill("replay_regression_drop", artifact_text, patch_data, {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1})
            if patch_data.get("tip_id"):
                self.tip_memory.mark_rejected(patch_data["tip_id"], "replay_regression_drop", {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1})
            self._append_version_log("rolled_back_replay_regression", patch_data, {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1})
            return baseline_f1, False

        if regression_mode == "full" or abs(sample_f1 - baseline_f1) <= threshold or abs(replay_f1 - baseline_f1) <= threshold:
            new_avg_f1 = local_score_func(golden_set) if local_score_func else self._score_dataset(golden_set, config, build_prompt_func, updated_skills, batch_size=batch_size, max_workers=max_workers, use_cache=True, few_shots=regression_few_shots)
            print(f"[Evolver] Full regression F1: {new_avg_f1:.2f}; baseline: {baseline_f1:.2f}")
            if new_avg_f1 + threshold < baseline_f1:
                print(f"[Evolver] Rolling back {action}: full regression drop {new_avg_f1:.2f} < {baseline_f1:.2f} - tolerance {threshold:.2f}.")
                self._rollback_artifact(action, artifact_snapshot)
                self._append_rejected_skill("full_regression_drop", artifact_text, patch_data, {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1, "full_f1": new_avg_f1})
                if patch_data.get("tip_id"):
                    self.tip_memory.mark_rejected(patch_data["tip_id"], "full_regression_drop", {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1, "full_f1": new_avg_f1})
                self._append_version_log("rolled_back_full_regression", patch_data, {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1, "full_f1": new_avg_f1})
                return baseline_f1, False
            self._curate_skill_repo(config)
            if patch_data.get("tip_id"):
                self.tip_memory.mark_promoted(patch_data["tip_id"], {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1, "full_f1": new_avg_f1})
            self._append_version_log("accepted_full_regression", patch_data, {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1, "full_f1": new_avg_f1})
            return new_avg_f1, True

        self._curate_skill_repo(config)
        if patch_data.get("tip_id"):
            self.tip_memory.mark_promoted(patch_data["tip_id"], {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1})
        self._append_version_log("accepted_sample_replay", patch_data, {"baseline_f1": baseline_f1, "sample_f1": sample_f1, "replay_f1": replay_f1})
        return min(sample_f1, replay_f1), True
