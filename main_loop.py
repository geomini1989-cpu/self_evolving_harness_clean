import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import time

import yaml

from adapters.ticket_adapter import TicketAdapter
from core.llm_client import BaseLLMClient
from core.evaluator import TaskEvaluator
from core.f1_optimizer import F1PostProcessor
from core.memory_bank import MemoryBank
from optimizer.attributor import SkillAttributor
from optimizer.evolver import SkillEvolver

REFUND = "\u9000\u6b3e\u7ea0\u7eb7"
LOGISTICS = "\u7269\u6d41\u6295\u8bc9"
ACCOUNT = "\u8d26\u53f7\u5c01\u7981"
SYSTEM_BUG = "\u7cfb\u7edfBug"
FALSE_AD = "\u865a\u5047\u5ba3\u4f20"
HIGH = "\u9ad8"
MEDIUM = "\u4e2d"
POSITIVE_KEYWORDS = ["\u597d\u5403", "\u5f88\u5feb", "\u6001\u5ea6\u597d", "\u7ed9\u529b", "\u8c22\u8c22", "\u4e0d\u9519", "\u6ee1\u610f", "\u65b9\u4fbf", "\u53ef\u53e3", "\u53ca\u65f6"]
URGENT_KEYWORDS = ["\u9a6c\u4e0a", "\u7acb\u523b", "\u8d76\u7d27", "\u6295\u8bc9", "\u5c01\u7981", "\u5d29\u6e83", "\u9ed1\u5c4f", "\u8d85\u65f6", "\u6001\u5ea6\u5dee"]
ENTITY_PATTERNS = [
    re.compile(r"(?:\u8ba2\u5355\u53f7|\u5355\u53f7|\u7f16\u53f7|id|ID)[:\uff1a]?\s*([A-Za-z0-9-]{5,})"),
    re.compile(r"\b\d{6,}\b"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:\u5143|\u5757|\u4eba\u6c11\u5e01)"),
]

DEFAULT_RUNTIME = {
    "mode": "demo",
    "batch_size": 8,
    "max_workers": 6,
    "epochs": 6,
    "sample_size": 30,
    "dataset_path": "data/dataset.csv",
    "shuffle_seed": 2026,
    "enable_few_shots": False,
    "enable_skill_seed": True,
    "enable_llm_scout": False,
    "enable_f1_postprocess": True,
    "enable_rule_fast_path": False,
    "force_evolution_demo": False,
    "use_rule_attributor": False,
    "use_rule_evolver": False,
    "enable_evolution": True,
    "reset_state": False,
    "execution_max_tokens_floor": 240,
    "execution_max_tokens_per_item": 90,
    "enable_cheap_json_repair": True,
    "enable_quality_early_stop": True,
    "target_f1_stop": 0.95,
    "target_exact_stop": 0.80,
}
DEFAULT_CACHE = {"enabled": True, "path": "memory/llm_cache.jsonl"}
DEFAULT_EVOLUTION = {
    "regression_mode": "sample_then_full",
    "f1_tolerance": 0.02,
    "confidence_threshold": 0.7,
    "sample_size": 10,
    "replay_size": 20,
    "max_new_skill_chars": 900,
    "max_skills_per_category": 5,
    "enable_tip_memory": True,
    "tip_promotion_threshold": 2,
    "tip_immediate_confidence": 0.85,
}


def deep_merge_defaults(config):
    config.setdefault("runtime", {})
    config.setdefault("cache", {})
    config.setdefault("evolution", {})
    for key, value in DEFAULT_RUNTIME.items():
        config["runtime"].setdefault(key, value)
    for key, value in DEFAULT_CACHE.items():
        config["cache"].setdefault(key, value)
    for key, value in DEFAULT_EVOLUTION.items():
        config["evolution"].setdefault(key, value)
    return config


def load_config(config_path="adapters/ticket_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return deep_merge_defaults(yaml.safe_load(f) or {})


def load_prompt_policy():
    file_path = "memory/PROMPT_POLICY.md"
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Evolving Harness runner")
    parser.add_argument("--mode", choices=["demo", "benchmark", "evolve-demo", "llm-evolve"], default=None, help="demo: closed-loop run; benchmark: one-pass evaluation; evolve-demo: deterministic evolution showcase; llm-evolve: real LLM multi-round evolution")
    parser.add_argument("--sample-size", default=None, help="Number of rows to evaluate, or 'all' for the full dataset")
    parser.add_argument("--dataset-path", default=None, help="CSV dataset path. The text column can be review/text/content/评价.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--no-evolution", action="store_true", help="Skip attribution/evolution after bad cases")
    parser.add_argument("--llm-benchmark", action="store_true", help="Use the real model in benchmark mode instead of local skill execution")
    parser.add_argument("--force-evolution-demo", action="store_true", help="Force one bad-case attribution and skill evolution step")
    parser.add_argument("--use-llm-evolution", action="store_true", help="Use real LLM attribution and rule generation instead of local rule evolution helpers")
    parser.add_argument("--reset-state", action="store_true", help="Clear runtime memory/cache files before this run")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable quality-based early stop in multi-epoch runs")
    parser.add_argument("--no-skill-seed", action="store_true", help="Do not restore adapters/ticket_base_skill.md into memory/SKILL.md")
    parser.add_argument("--no-f1-postprocess", action="store_true", help="Evaluate raw LLM outputs without deterministic F1 post-processing")
    parser.add_argument("--enable-few-shots", action="store_true", help="Inject few-shot examples learned during this run")
    parser.add_argument("--prefer-skill-patch", action="store_true", help="Prefer SkillRepo patches over few-shot patches for non-schema bad cases")
    return parser.parse_args()


def apply_cli_overrides(config, args):
    runtime = config["runtime"]
    if args.mode:
        runtime["mode"] = args.mode
    if runtime.get("mode") == "benchmark":
        runtime["sample_size"] = None
        runtime["epochs"] = 1
        runtime["batch_size"] = max(int(runtime.get("batch_size", 8)), 16)
        runtime["max_workers"] = min(int(runtime.get("max_workers", 6)), 4)
        runtime["enable_evolution"] = False
        runtime["enable_few_shots"] = False
        runtime["enable_rule_fast_path"] = True
    if runtime.get("mode") == "evolve-demo":
        runtime["sample_size"] = int(runtime.get("sample_size") or 30)
        runtime["epochs"] = 1
        runtime["batch_size"] = min(int(runtime.get("batch_size", 8)), 8)
        runtime["max_workers"] = min(int(runtime.get("max_workers", 6)), 2)
        runtime["enable_evolution"] = True
        runtime["force_evolution_demo"] = True
        runtime["use_rule_attributor"] = True
        runtime["use_rule_evolver"] = True
        runtime["enable_rule_fast_path"] = False
    if runtime.get("mode") == "llm-evolve":
        runtime["sample_size"] = int(runtime.get("sample_size") or 60)
        runtime["epochs"] = int(runtime.get("epochs") or 3)
        runtime["batch_size"] = min(int(runtime.get("batch_size", 8)), 8)
        runtime["max_workers"] = min(int(runtime.get("max_workers", 6)), 2)
        runtime["enable_evolution"] = True
        runtime["enable_few_shots"] = bool(runtime.get("enable_few_shots", False))
        runtime["force_evolution_demo"] = False
        runtime["use_rule_attributor"] = False
        runtime["use_rule_evolver"] = False
        runtime["enable_rule_fast_path"] = False
        config["evolution"]["tip_promotion_threshold"] = max(int(config["evolution"].get("tip_promotion_threshold", 2)), 2)
        config["evolution"]["tip_immediate_confidence"] = max(float(config["evolution"].get("tip_immediate_confidence", 0.85)), 1.01)
        config["evolution"]["confidence_threshold"] = max(float(config["evolution"].get("confidence_threshold", 0.7)), 0.75)
        config["evolution"]["prefer_few_shot_first"] = True
        config["evolution"]["allow_immediate_few_shot"] = True
    if args.llm_benchmark:
        runtime["enable_rule_fast_path"] = False
    if args.force_evolution_demo:
        runtime["force_evolution_demo"] = True
        runtime["use_rule_attributor"] = True
        runtime["use_rule_evolver"] = True
    if args.use_llm_evolution:
        runtime["use_rule_attributor"] = False
        runtime["use_rule_evolver"] = False
        runtime["enable_rule_fast_path"] = False
    if args.sample_size is not None:
        runtime["sample_size"] = None if str(args.sample_size).lower() in {"all", "full", "none"} else int(args.sample_size)
    if args.dataset_path is not None:
        runtime["dataset_path"] = args.dataset_path
    if args.epochs is not None:
        runtime["epochs"] = args.epochs
    if args.batch_size is not None:
        runtime["batch_size"] = args.batch_size
    if args.max_workers is not None:
        runtime["max_workers"] = args.max_workers
    if args.no_evolution:
        runtime["enable_evolution"] = False
    if args.reset_state:
        runtime["reset_state"] = True
    if args.no_early_stop:
        runtime["enable_quality_early_stop"] = False
    if args.no_skill_seed:
        runtime["enable_skill_seed"] = False
    if args.no_f1_postprocess:
        runtime["enable_f1_postprocess"] = False
    if args.enable_few_shots:
        runtime["enable_few_shots"] = True
    if args.prefer_skill_patch:
        config["evolution"]["prefer_few_shot_first"] = False
        config["evolution"]["prefer_skill_patch"] = True
    return config


def build_batch_execution_prompt(config, current_skills, few_shots, batch_texts, meta_intervention=False):
    schema_str = json.dumps(config["schema"], ensure_ascii=False, separators=(",", ":"))
    meta_prompt = "Extra caution: inspect implicit logic and sarcasm.\n" if meta_intervention else ""
    skills_block = current_skills.strip() if current_skills else "No matched category-specific skills."
    prompt_policy = load_prompt_policy()
    prompt_policy_block = f"\nPrompt policy:\n{prompt_policy}\n" if prompt_policy else ""
    few_shot_block = f"\nExamples:\n{few_shots.strip()}\n" if few_shots else ""
    batch_text_str = "\n".join(f"{idx}. {text}" for idx, text in enumerate(batch_texts, 1))
    return f"""Task: parse customer complaint texts into structured JSON.
Output compact JSON only: one array with exactly {len(batch_texts)} objects in the same order as inputs. No Markdown, no explanation, no repeated input text.
Each object must include exactly these keys: core_intent, urgency_level, entities, summary.
Keep summary <= 20 Chinese chars. Keep entities as a short array. Use only schema labels for core_intent and urgency_level.
Schema: {schema_str}
Rules: {skills_block}
{prompt_policy_block}{meta_prompt}{few_shot_block}Inputs:
{batch_text_str}
"""


def build_execution_prompt(config, current_skills, few_shots, input_text, meta_intervention=False):
    return build_batch_execution_prompt(config, current_skills, few_shots, [input_text], meta_intervention)


def chunk_dataset(dataset, batch_size=8):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]


def execution_max_tokens(config, item_count):
    runtime = config.get("runtime", {})
    floor = int(runtime.get("execution_max_tokens_floor", 240))
    per_item = int(runtime.get("execution_max_tokens_per_item", 90))
    return max(floor, per_item * max(1, item_count))


def log_metrics(epoch, f1_score, llm_stats=None, elapsed_ms=None, sample_count=0, exact_match_rate=0.0):
    file_path = "memory/metrics.csv"
    file_exists = os.path.exists(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_exists:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
        if "exact_match_rate" not in header:
            legacy_path = f"memory/metrics_legacy_{int(time.time())}.csv"
            os.replace(file_path, legacy_path)
            print(f"[Metrics] Existing metrics.csv used the old schema; moved it to {legacy_path}.")
            file_exists = False
    llm_stats = llm_stats or {}
    sample_count = int(sample_count or 0)
    total_tokens = int(llm_stats.get("total_tokens", 0) or 0)
    cache_hits = int(llm_stats.get("cache_hits", 0) or 0)
    cache_misses = int(llm_stats.get("cache_misses", 0) or 0)
    elapsed_ms = int(elapsed_ms or 0)
    tokens_per_sample = total_tokens / sample_count if sample_count else 0.0
    latency_per_sample_ms = elapsed_ms / sample_count if sample_count else 0.0
    cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) else 0.0
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "f1_score", "exact_match_rate", "sample_count", "llm_calls", "cache_hits", "cache_misses", "cache_hit_rate", "prompt_tokens", "completion_tokens", "total_tokens", "tokens_per_sample", "elapsed_ms", "latency_per_sample_ms"])
        writer.writerow([epoch, f1_score, exact_match_rate, sample_count, llm_stats.get("calls", 0), cache_hits, cache_misses, cache_hit_rate, llm_stats.get("prompt_tokens", 0), llm_stats.get("completion_tokens", 0), total_tokens, tokens_per_sample, elapsed_ms, latency_per_sample_ms])


def log_latest_patch(patch_data):
    file_path = "memory/latest_patch.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(patch_data, f, ensure_ascii=False, indent=2)


def log_saf_trace(adapter, data, prediction, eval_result, epoch):
    file_path = "memory/saf_traces.jsonl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    trace = adapter.make_trace(
        data["input"],
        prediction,
        eval_result,
        epoch=epoch,
        ground_truth=data.get("ground_truth", {}),
    )
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace.to_record(), ensure_ascii=False, separators=(",", ":")) + "\n")


def extract_entities(text):
    entities = []
    for pattern in ENTITY_PATTERNS:
        for match in pattern.findall(text):
            value = match if isinstance(match, str) else match[0]
            value = str(value).strip()
            if value and value not in entities:
                entities.append(value)
    return entities


def infer_ground_truth(text, memory_bank=None):
    route_scores = memory_bank.score_categories(text) if memory_bank else []
    intent = route_scores[0][0] if route_scores else REFUND
    urgency = HIGH if intent in [LOGISTICS, SYSTEM_BUG, ACCOUNT] or any(keyword in text for keyword in URGENT_KEYWORDS) else MEDIUM
    return {
        "core_intent": intent,
        "urgency_level": urgency,
        "entities": extract_entities(text),
        "summary": "\u7528\u6237\u8d1f\u9762\u4f53\u9a8c\u5ba2\u8bc9\u5904\u7406",
    }


def is_complaint_candidate(text, label=None):
    if label is not None:
        return str(label).strip() == "0"
    return not any(keyword in text for keyword in POSITIVE_KEYWORDS)


def init_real_dataset(sample_size=30, shuffle_seed=2026, dataset_path="data/dataset.csv"):
    local_file = dataset_path
    print(f"[Dataset] Loading local dataset: {local_file}")
    try:
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"Missing local file: {local_file}")
        golden_dataset = []
        label_router = MemoryBank()
        with open(local_file, mode="r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            text_idx = 0
            if header:
                for i, col_name in enumerate(header):
                    if col_name.strip().lower() in ["review", "text", "content", "\u8bc4\u4ef7"]:
                        text_idx = i
                        break
            label_idx = None
            if header:
                for i, col_name in enumerate(header):
                    if col_name.strip().lower() in ["label", "sentiment"]:
                        label_idx = i
                        break
            all_rows = list(reader)
            random.seed(shuffle_seed if shuffle_seed is not None else int(time.time()))
            random.shuffle(all_rows)
            for row in all_rows:
                if sample_size is not None and len(golden_dataset) >= sample_size:
                    break
                if len(row) <= text_idx:
                    continue
                text = row[text_idx].strip()
                if len(text) < 6:
                    continue
                label = row[label_idx] if label_idx is not None and len(row) > label_idx else None
                if not is_complaint_candidate(text, label):
                    continue
                golden_dataset.append({"input": text, "ground_truth": infer_ground_truth(text, label_router)})
        if not golden_dataset:
            raise ValueError(f"No valid text rows found in {local_file}")
        print(f"[Dataset] Loaded {len(golden_dataset)} samples.")
        return golden_dataset
    except Exception as exc:
        print(f"[Dataset] Failed to load local dataset: {exc}")
        return [
            {"input": "\u6628\u5929\u4e70\u7684\u624b\u673a\u5230\u4e86\u5c31\u9ed1\u5c4f\uff0c\u5feb\u9012\u5458\u6001\u5ea6\u4e5f\u5dee\u3002\u6211\u8981\u9000\u6b3e\uff0c\u8fd8\u8981\u6295\u8bc9\u4ed6\uff01\u8ba2\u5355\u53f78829103\u3002", "ground_truth": {"core_intent": REFUND, "urgency_level": HIGH, "entities": ["8829103"], "summary": "\u624b\u673a\u9ed1\u5c4f\u8981\u6c42\u9000\u6b3e"}},
            {"input": "\u4f60\u4eec\u5bb6\u679c\u5b50\u53d7\u6f6e\u4e86\uff0c\u6839\u672c\u6ca1\u6cd5\u5403\uff0c\u6211\u8981\u9000\u94b1\uff01", "ground_truth": {"core_intent": REFUND, "urgency_level": MEDIUM, "entities": [], "summary": "\u5546\u54c1\u53d7\u6f6e\u8981\u6c42\u9000\u6b3e"}},
            {"input": "\u7cfb\u7edf\u63d0\u793a\u5f02\u5730\u767b\u5f55\u5c01\u7981\uff0c\u6211\u91cc\u9762\u8fd8\u6709\u94b1\uff0c\u9a6c\u4e0a\u89e3\u5c01\u3002", "ground_truth": {"core_intent": ACCOUNT, "urgency_level": HIGH, "entities": [], "summary": "\u8d26\u53f7\u5c01\u7981\u8981\u6c42\u89e3\u5c01"}},
        ]


def prepare_demo_env(reset_state=False, enable_skill_seed=True):
    os.makedirs("memory", exist_ok=True)
    os.makedirs("adapters", exist_ok=True)
    if not reset_state:
        if enable_skill_seed:
            ensure_base_skill_repo()
        return
    print("[Harness] Resetting runtime memory and cache files.")
    reset_files = [
        "memory/metrics.csv",
        "memory/latest_patch.json",
        "memory/token_usage.csv",
        "memory/llm_cache.jsonl",
        "memory/PROMPT_POLICY.md",
        "memory/PROMPT_POLICY_backup.md",
        "memory/tips.jsonl",
        "memory/SKILL.md",
        "memory/SKILL_backup.md",
        "memory/examples.json",
        "memory/examples_backup.json",
        "memory/saf_traces.jsonl",
        "memory/skill_versions.jsonl",
        "memory/rejected_skills.jsonl",
        "memory/transfer_report.json",
        "memory/transfer_traces.jsonl",
    ]
    for file in reset_files:
        if os.path.exists(file):
            os.remove(file)
    if enable_skill_seed:
        ensure_base_skill_repo()


def ensure_base_skill_repo():
    skill_file = "memory/SKILL.md"
    seed_file = "adapters/ticket_base_skill.md"
    if os.path.exists(skill_file) or not os.path.exists(seed_file):
        return
    os.makedirs(os.path.dirname(skill_file), exist_ok=True)
    with open(seed_file, "r", encoding="utf-8") as src:
        content = src.read().strip()
    with open(skill_file, "w", encoding="utf-8") as dst:
        dst.write(content + "\n")
    print("[Harness] Seeded base SkillRepo from adapters/ticket_base_skill.md.")


def evaluate_optimized_batch_items(evaluator, f1_optimizer, batch_data, parsed_array, prompt, enabled=True):
    results = []
    for idx, data in enumerate(batch_data):
        raw_prediction = parsed_array[idx]
        prediction = f1_optimizer.optimize_to_json(data["input"], raw_prediction) if enabled else json.dumps(raw_prediction, ensure_ascii=False, separators=(",", ":"))
        eval_result = evaluator.evaluate(prediction, data["ground_truth"])
        results.append({
            "data": data,
            "prompt": prompt,
            "raw_prediction": json.dumps(raw_prediction, ensure_ascii=False, separators=(",", ":")),
            "prediction": prediction,
            "eval_result": eval_result,
        })
    return results


def normalize_batch_array(evaluator, prediction_text, expected_len):
    parsed_array = evaluator._extract_json_from_text(prediction_text)
    if isinstance(parsed_array, dict):
        for key in ["items", "results", "data", "outputs"]:
            if isinstance(parsed_array.get(key), list):
                parsed_array = parsed_array[key]
                break
    is_valid_batch = isinstance(parsed_array, list) and len(parsed_array) == expected_len
    return parsed_array, is_valid_batch


def repair_batch_json_with_cheap_model(llm, evaluator, config, prediction_text, expected_len):
    if not bool(config.get("runtime", {}).get("enable_cheap_json_repair", True)):
        return prediction_text, None, False
    schema_str = json.dumps(config["schema"], ensure_ascii=False, separators=(",", ":"))
    clipped_output = str(prediction_text or "")[:5000]
    repair_prompt = f"""Repair the following malformed model output into valid compact JSON.
Return only one JSON array with exactly {expected_len} objects. Do not add explanations.
Each object must include exactly these keys: core_intent, urgency_level, entities, summary.
Preserve the original predicted values when possible; only fix JSON syntax, wrappers, missing brackets, trailing text, or object-array shape.
Schema labels: {schema_str}
Malformed output:
{clipped_output}
"""
    try:
        repaired_text = llm.generate(
            repair_prompt,
            temperature=0.0,
            model_type="cheap",
            use_cache=True,
            max_tokens=execution_max_tokens(config, expected_len),
        )
    except Exception as exc:
        print(f"[Batch] Cheap JSON repair failed: {exc}")
        return prediction_text, None, False
    repaired_array, ok = normalize_batch_array(evaluator, repaired_text, expected_len)
    if ok:
        print(f"[Batch] Cheap JSON repair succeeded for batch size {expected_len}.")
        return repaired_text, repaired_array, True
    print(f"[Batch] Cheap JSON repair did not produce a valid array for batch size {expected_len}.")
    return repaired_text, repaired_array, False


def run_rule_batch(evaluator, memory_bank, batch_data):
    predictions = [infer_ground_truth(data["input"], memory_bank) for data in batch_data]
    return evaluate_optimized_batch_items(evaluator, F1PostProcessor(memory_bank), batch_data, predictions, "local_skill_fast_path", enabled=True)


def score_rule_dataset(evaluator, memory_bank, dataset):
    if not dataset:
        return 0.0
    results = []
    for batch_data in chunk_dataset(dataset, batch_size=16):
        results.extend(run_rule_batch(evaluator, memory_bank, batch_data))
    return sum(item["eval_result"]["f1_score"] for item in results) / len(dataset)


def run_forced_evolution_demo(config, llm, evaluator, memory_bank, attributor, evolver, golden_dataset, domain_adapter):
    print("[Evolution Demo] Forcing one bad case to demonstrate execute-evaluate-reflect-evolve.")
    memory_bank.refresh_memory()
    target = next((item for item in golden_dataset if item["ground_truth"].get("core_intent") != FALSE_AD), golden_dataset[0])
    wrong_intent = FALSE_AD if target["ground_truth"].get("core_intent") != FALSE_AD else REFUND
    wrong_prediction = {
        "core_intent": wrong_intent,
        "urgency_level": MEDIUM,
        "entities": [],
        "summary": "\u6f14\u793a\u7528\u9519\u8bef\u9884\u6d4b",
    }
    prediction_text = json.dumps(wrong_prediction, ensure_ascii=False, separators=(",", ":"))
    bad_eval = evaluator.evaluate(prediction_text, target["ground_truth"])
    baseline_f1 = (bad_eval["f1_score"] + score_rule_dataset(evaluator, memory_bank, [item for item in golden_dataset if item is not target]) * max(len(golden_dataset) - 1, 0)) / len(golden_dataset)
    bad_case = {
        "input": target["input"],
        "ground_truth": target["ground_truth"],
        "prediction": prediction_text,
        "eval_result": bad_eval,
    }
    log_saf_trace(domain_adapter, bad_case, prediction_text, bad_eval, "evolve_demo_bad_case")
    print(f"[Evolution Demo] Bad case F1={bad_eval['f1_score']:.2f}; baseline set F1={baseline_f1:.2f}")
    patch = attributor.analyze_root_cause(
        bad_case["eval_result"],
        bad_case["input"],
        bad_case["prediction"],
        bad_case["ground_truth"],
        use_llm=not bool(config["runtime"].get("use_rule_attributor", False)),
    )
    print(f"[Evolution Demo] Root cause type: {patch.get('root_cause_type')} | confidence={float(patch.get('confidence', 0.0)):.2f} | risk={patch.get('risk_flags')}")
    print(f"[Evolution Demo] Attribution: {patch.get('root_cause_analysis')}")
    print(f"[Evolution Demo] Proposed rule: {patch.get('proposed_rule')}")
    log_latest_patch(patch)
    local_score = lambda dataset: score_rule_dataset(evaluator, memory_bank, dataset)
    new_f1, success = evolver.apply_patch_with_rollback(
        attributor=attributor,
        patch_data=patch,
        config=config,
        golden_set=golden_dataset,
        build_prompt_func=build_batch_execution_prompt,
        baseline_f1=baseline_f1,
        local_score_func=local_score,
    )
    memory_bank.refresh_skills()
    fixed_prediction = json.dumps(infer_ground_truth(target["input"], memory_bank), ensure_ascii=False, separators=(",", ":"))
    fixed_eval = evaluator.evaluate(fixed_prediction, target["ground_truth"])
    log_saf_trace(domain_adapter, target, fixed_prediction, fixed_eval, "evolve_demo_fixed_case")
    stats = llm.snapshot_stats()
    log_metrics("evolve_demo", new_f1, stats, 0, sample_count=len(golden_dataset), exact_match_rate=1.0 if fixed_eval.get("exact_match") else 0.0)
    print(f"[Evolution Demo] Patch success={success}; regression F1={new_f1:.2f}; fixed bad-case F1={fixed_eval['f1_score']:.2f}")
    if success:
        print("[Evolution Demo] Long-term memory updated after regression gates.")
    else:
        print("[Evolution Demo] Long-term memory unchanged; short-term memory kept the evidence.")


def run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data, meta_intervention, enable_few_shots, enable_f1_postprocess=True, enable_rule_fast_path=False):
    if enable_rule_fast_path:
        return run_rule_batch(evaluator, memory_bank, batch_data)

    batch_inputs = [data["input"] for data in batch_data]
    active_categories = memory_bank.route_categories(batch_inputs)
    current_skills = memory_bank.get_skills_by_categories(active_categories)
    few_shots = memory_bank.get_few_shots(batch_inputs[0], k=1, enabled=enable_few_shots)
    prompt = build_batch_execution_prompt(config, current_skills, few_shots, batch_inputs, meta_intervention)
    try:
        prediction_text = llm.generate(
            prompt,
            temperature=0.0,
            model_type="default",
            use_cache=True,
            max_tokens=execution_max_tokens(config, len(batch_inputs)),
        )
    except Exception as exc:
        if len(batch_data) > 1:
            midpoint = len(batch_data) // 2
            print(f"[Batch] LLM request failed for batch size {len(batch_data)}: {exc}; retrying as smaller batches.")
            return run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data[:midpoint], meta_intervention, enable_few_shots, enable_f1_postprocess, enable_rule_fast_path) + run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data[midpoint:], meta_intervention, enable_few_shots, enable_f1_postprocess, enable_rule_fast_path)
        prediction_text = "{}"
    parsed_array, valid_batch = normalize_batch_array(evaluator, prediction_text, len(batch_data))
    if not valid_batch:
        repaired_text, repaired_array, repair_ok = repair_batch_json_with_cheap_model(llm, evaluator, config, prediction_text, len(batch_data))
        if repair_ok:
            prediction_text = repaired_text
            parsed_array = repaired_array
            valid_batch = True
    if valid_batch:
        return evaluate_optimized_batch_items(evaluator, f1_optimizer, batch_data, parsed_array, prompt, enabled=enable_f1_postprocess)
    if len(batch_data) <= 1:
        raw_prediction = parsed_array if isinstance(parsed_array, dict) else {}
        prediction = f1_optimizer.optimize_to_json(batch_data[0]["input"], raw_prediction) if enable_f1_postprocess else json.dumps(raw_prediction, ensure_ascii=False, separators=(",", ":"))
        eval_result = evaluator.evaluate(prediction, batch_data[0]["ground_truth"])
        return [{
            "data": batch_data[0],
            "prompt": prompt,
            "raw_prediction": prediction_text or "{}",
            "prediction": prediction,
            "eval_result": eval_result,
        }]
    midpoint = len(batch_data) // 2
    print(f"[Batch] Invalid JSON array for batch size {len(batch_data)}; retrying as smaller batches.")
    return run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data[:midpoint], meta_intervention, enable_few_shots, enable_f1_postprocess, enable_rule_fast_path) + run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data[midpoint:], meta_intervention, enable_few_shots, enable_f1_postprocess, enable_rule_fast_path)


def run_harness_loop(config=None):
    print("[Harness] Starting Self-Evolving Harness with token-saving defaults...")
    config_path = "adapters/ticket_config.yaml"
    config = config or load_config(config_path)
    runtime = config["runtime"]
    prepare_demo_env(
        reset_state=bool(runtime.get("reset_state", False)),
        enable_skill_seed=bool(runtime.get("enable_skill_seed", True)),
    )
    llm = BaseLLMClient(cache_enabled=bool(config["cache"].get("enabled", True)), cache_path=config["cache"].get("path", "memory/llm_cache.jsonl"))
    evaluator = TaskEvaluator(config_path)
    domain_adapter = TicketAdapter(schema=config.get("schema", {}))
    memory_bank = MemoryBank()
    f1_optimizer = F1PostProcessor(memory_bank)
    attributor = SkillAttributor(llm)
    evolver = SkillEvolver(evaluator, llm)
    sample_size = runtime.get("sample_size", 30)
    sample_size = None if sample_size in [None, "all", "full"] else int(sample_size)
    golden_dataset = init_real_dataset(
        sample_size=sample_size,
        shuffle_seed=runtime.get("shuffle_seed", 2026),
        dataset_path=runtime.get("dataset_path", "data/dataset.csv"),
    )
    print(f"[Harness] Mode: {runtime.get('mode', 'demo')} | Dataset size: {len(golden_dataset)}")
    if bool(runtime.get("force_evolution_demo", False)):
        run_forced_evolution_demo(config, llm, evaluator, memory_bank, attributor, evolver, golden_dataset, domain_adapter)
        return
    epochs = int(runtime.get("epochs", 6))
    batch_size = int(runtime.get("batch_size", 8))
    max_workers = int(runtime.get("max_workers", 6))
    enable_few_shots = bool(runtime.get("enable_few_shots", False))
    enable_f1_postprocess = bool(runtime.get("enable_f1_postprocess", True))
    enable_rule_fast_path = bool(runtime.get("enable_rule_fast_path", False))
    enable_evolution = bool(runtime.get("enable_evolution", True))
    enable_quality_early_stop = bool(runtime.get("enable_quality_early_stop", True))
    target_f1_stop = float(runtime.get("target_f1_stop", 0.95))
    target_exact_stop = float(runtime.get("target_exact_stop", 0.80))
    estimated_batches = (len(golden_dataset) + batch_size - 1) // batch_size
    print(f"[Harness] Batch size: {batch_size} | Estimated model batches per epoch: {estimated_batches} | Epochs: {epochs}")
    print(f"[Harness] Execution max_tokens per batch: up to {execution_max_tokens(config, batch_size)} for batch_size={batch_size}.")
    if enable_quality_early_stop:
        print(f"[Harness] Quality early stop: F1>={target_f1_stop:.2f} and Exact>={target_exact_stop:.0%}.")
    if enable_rule_fast_path:
        print("[Harness] Benchmark fast path: local skill execution is enabled; LLM calls should stay near zero.")
    else:
        print("[Harness] Execution backend: LLM batch extraction.")
        if getattr(llm, "timeout_seconds", 0):
            print(f"[Harness] LLM request timeout: {int(llm.timeout_seconds)}s; timed-out batches will split and retry smaller.")
        if getattr(llm, "offline_mode", False):
            print("[Harness] Offline mode is active; LLM calls will use the local demo engine.")
    if enable_evolution:
        attribution_backend = "rule" if bool(runtime.get("use_rule_attributor", False)) else "LLM"
        evolver_backend = "rule" if bool(runtime.get("use_rule_evolver", False)) else "LLM"
        print(f"[Harness] Evolution backend: attribution={attribution_backend}, patch_generation={evolver_backend}.")
    stuck_counter = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        print(f"\n================ Epoch {epoch} ================")
        epoch_total_f1 = 0.0
        exact_match_count = 0
        bad_case = None
        results = []
        memory_bank.refresh_skills()
        meta_intervention = stuck_counter >= 2
        batches = list(chunk_dataset(golden_dataset, batch_size=batch_size))
        completed_batches = 0
        print(f"[Epoch {epoch}] Dispatching {len(batches)} batches with max_workers={max_workers}.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_batch_with_split,
                    llm,
                    evaluator,
                    config,
                    memory_bank,
                    f1_optimizer,
                    batch_data,
                    meta_intervention,
                    enable_few_shots,
                    enable_f1_postprocess,
                    enable_rule_fast_path,
                ): idx
                for idx, batch_data in enumerate(batches, 1)
            }
            for future in concurrent.futures.as_completed(futures):
                batch_idx = futures[future]
                completed_batches += 1
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    elapsed_s = int(time.time() - epoch_started)
                    print(f"[Epoch {epoch}] Batch {completed_batches}/{len(batches)} done (submitted #{batch_idx}, rows={len(batch_results)}, elapsed_s={elapsed_s}).")
                except Exception as exc:
                    print(f"[Epoch {epoch}] Batch {completed_batches}/{len(batches)} failed (submitted #{batch_idx}): {exc}")
        for result in results:
            eval_result = result["eval_result"]
            epoch_total_f1 += eval_result["f1_score"]
            log_saf_trace(domain_adapter, result["data"], result["prediction"], eval_result, epoch)
            if eval_result["exact_match"]:
                exact_match_count += 1
                memory_bank.add_successful_case(result["data"]["input"], evaluator._extract_json_from_text(result["prediction"]))
            elif bad_case is None:
                bad_case = {"input": result["data"]["input"], "ground_truth": result["data"]["ground_truth"], "prediction": result["prediction"], "eval_result": eval_result}
        avg_f1 = epoch_total_f1 / len(golden_dataset) if golden_dataset else 0.0
        exact_match_rate = exact_match_count / len(golden_dataset) if golden_dataset else 0.0
        elapsed_ms = (time.time() - epoch_started) * 1000
        stats = llm.snapshot_stats()
        log_metrics(epoch, avg_f1, stats, elapsed_ms, sample_count=len(golden_dataset), exact_match_rate=exact_match_rate)
        tokens_per_sample = stats["total_tokens"] / len(golden_dataset) if golden_dataset else 0.0
        latency_per_sample_ms = elapsed_ms / len(golden_dataset) if golden_dataset else 0.0
        print(f"[Epoch {epoch}] F1={avg_f1:.2f}, exact={exact_match_rate:.2%}, calls={stats['calls']}, cache_hits={stats['cache_hits']}, tokens={stats['total_tokens']}, tokens/sample={tokens_per_sample:.1f}, latency/sample_ms={latency_per_sample_ms:.0f}, elapsed_ms={int(elapsed_ms)}")
        if enable_quality_early_stop and avg_f1 >= target_f1_stop and exact_match_rate >= target_exact_stop:
            print("[Harness] Quality target reached; stopping before extra evolution/epochs to save tokens.")
            break
        if avg_f1 >= 1.0:
            print("[Harness] Reached 100% F1. Evolution complete.")
            break
        if bad_case and enable_evolution:
            print("[Harness] Found a bad case; starting adaptive patch flow.")
            patch = attributor.analyze_root_cause(
                bad_case["eval_result"],
                bad_case["input"],
                bad_case["prediction"],
                bad_case["ground_truth"],
                use_llm=not bool(runtime.get("use_rule_attributor", False)),
            )
            if bool(config.get("evolution", {}).get("prefer_few_shot_first", False)):
                root_type = patch.get("root_cause_type")
                if root_type not in {"json_format_error", "schema_field_error"}:
                    patch["evolution_action"] = "few_shot_patch"
                    patch.setdefault("risk_flags", [])
                    if "few_shot_first_policy" not in patch["risk_flags"]:
                        patch["risk_flags"].append("few_shot_first_policy")
                    print("[Harness] Few-shot-first policy: using a low-risk example patch for this bad case.")
            if bool(config.get("evolution", {}).get("prefer_skill_patch", False)):
                root_type = patch.get("root_cause_type")
                if root_type not in {"json_format_error", "schema_field_error"}:
                    patch["evolution_action"] = "skill_patch"
                    patch.setdefault("risk_flags", [])
                    if "skill_patch_preferred" not in patch["risk_flags"]:
                        patch["risk_flags"].append("skill_patch_preferred")
                    print("[Harness] Skill-first policy: trying to grow SkillRepo from this bad case.")
            log_latest_patch(patch)
            baseline_f1 = avg_f1
            new_f1, success = evolver.apply_patch_with_rollback(attributor=attributor, patch_data=patch, config=config, golden_set=golden_dataset, build_prompt_func=build_batch_execution_prompt, baseline_f1=baseline_f1)
            if success:
                memory_bank.refresh_memory()
            stuck_counter = stuck_counter + 1 if (not success or new_f1 <= baseline_f1) else 0
        elif bad_case:
            print("[Harness] Bad case found, but evolution is disabled for this run.")


if __name__ == "__main__":
    args = parse_args()
    run_harness_loop(apply_cli_overrides(load_config(), args))
