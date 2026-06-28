import argparse
import concurrent.futures
import csv
import json
import os
import random
import time

import yaml

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

DEFAULT_RUNTIME = {"mode": "demo", "batch_size": 8, "max_workers": 6, "epochs": 6, "sample_size": 30, "shuffle_seed": 2026, "enable_few_shots": False, "enable_llm_scout": False, "enable_f1_postprocess": True, "enable_evolution": True, "reset_state": False}
DEFAULT_CACHE = {"enabled": True, "path": "memory/llm_cache.jsonl"}
DEFAULT_EVOLUTION = {"regression_mode": "sample_then_full", "f1_tolerance": 0.02, "sample_size": 10}


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


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Evolving Harness runner")
    parser.add_argument("--mode", choices=["demo", "benchmark"], default=None, help="demo: 30-sample closed-loop run; benchmark: full-dataset one-pass evaluation")
    parser.add_argument("--sample-size", default=None, help="Number of rows to evaluate, or 'all' for the full dataset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--no-evolution", action="store_true", help="Skip attribution/evolution after bad cases")
    return parser.parse_args()


def apply_cli_overrides(config, args):
    runtime = config["runtime"]
    if args.mode:
        runtime["mode"] = args.mode
    if runtime.get("mode") == "benchmark":
        runtime["sample_size"] = None
        runtime["epochs"] = 1
        runtime["batch_size"] = max(int(runtime.get("batch_size", 8)), 32)
        runtime["max_workers"] = min(int(runtime.get("max_workers", 6)), 4)
        runtime["enable_evolution"] = False
        runtime["enable_few_shots"] = False
    if args.sample_size is not None:
        runtime["sample_size"] = None if str(args.sample_size).lower() in {"all", "full", "none"} else int(args.sample_size)
    if args.epochs is not None:
        runtime["epochs"] = args.epochs
    if args.batch_size is not None:
        runtime["batch_size"] = args.batch_size
    if args.max_workers is not None:
        runtime["max_workers"] = args.max_workers
    if args.no_evolution:
        runtime["enable_evolution"] = False
    return config


def load_skills():
    try:
        with open("memory/SKILL.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "No business skills are currently available. Use the base schema and the input text."


def build_execution_prompt(config, current_skills, few_shots, input_text, meta_intervention=False):
    return build_batch_execution_prompt(config, current_skills, few_shots, [input_text], meta_intervention)


def build_batch_execution_prompt(config, current_skills, few_shots, batch_texts, meta_intervention=False):
    schema_str = json.dumps(config["schema"], ensure_ascii=False, separators=(",", ":"))
    meta_prompt = "Extra caution: inspect implicit logic and sarcasm.\n" if meta_intervention else ""
    skills_block = current_skills.strip() if current_skills else "No matched category-specific skills."
    few_shot_block = f"\nExamples:\n{few_shots.strip()}\n" if few_shots else ""
    batch_text_str = "\n".join(f"{idx}. {text}" for idx, text in enumerate(batch_texts, 1))
    return f"""Task: parse customer complaint texts into structured JSON.
Output only a JSON array with exactly {len(batch_texts)} objects in the same order as inputs. No Markdown.
Schema: {schema_str}
Rules: {skills_block}
{meta_prompt}{few_shot_block}Inputs:
{batch_text_str}
"""


def chunk_dataset(dataset, batch_size=8):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]


def log_metrics(epoch, f1_score, llm_stats=None, elapsed_ms=None):
    file_path = "memory/metrics.csv"
    file_exists = os.path.exists(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    llm_stats = llm_stats or {}
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "f1_score", "llm_calls", "cache_hits", "cache_misses", "prompt_tokens", "completion_tokens", "total_tokens", "elapsed_ms"])
        writer.writerow([epoch, f1_score, llm_stats.get("calls", 0), llm_stats.get("cache_hits", 0), llm_stats.get("cache_misses", 0), llm_stats.get("prompt_tokens", 0), llm_stats.get("completion_tokens", 0), llm_stats.get("total_tokens", 0), int(elapsed_ms or 0)])


def log_latest_patch(patch_data):
    file_path = "memory/latest_patch.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(patch_data, f, ensure_ascii=False, indent=2)


def infer_ground_truth(text):
    intent = REFUND
    if any(keyword in text for keyword in ["\u9a91\u624b", "\u5916\u5356", "\u5feb\u9012", "\u7269\u6d41", "\u914d\u9001", "\u6001\u5ea6", "\u8fdf\u5230", "\u8d85\u65f6"]):
        intent = LOGISTICS
    elif any(keyword in text for keyword in ["\u5c01\u7981", "\u5c01\u53f7", "\u8d26\u53f7", "\u8d26\u6237", "\u89e3\u5c01"]):
        intent = ACCOUNT
    elif any(keyword in text for keyword in ["\u7cfb\u7edf", "\u7f51\u9875", "\u5d29\u6e83", "\u95ea\u9000", "\u5bc6\u7801", "\u62a5\u9519", "\u767b\u5f55"]):
        intent = SYSTEM_BUG
    elif any(keyword in text for keyword in ["\u5047", "\u9a97", "\u56fe\u6587\u4e0d\u7b26", "\u5ba3\u4f20", "\u865a\u5047"]):
        intent = FALSE_AD
    return {"core_intent": intent, "urgency_level": HIGH if intent in [LOGISTICS, SYSTEM_BUG, ACCOUNT] else MEDIUM, "entities": [], "summary": "\u7528\u6237\u8d1f\u9762\u4f53\u9a8c\u5ba2\u8bc9\u5904\u7406"}


def init_real_dataset(sample_size=30, shuffle_seed=2026):
    local_file = "data/dataset.csv"
    print(f"[Dataset] Loading local dataset: {local_file}")
    try:
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"Missing local file: {local_file}")
        golden_dataset = []
        with open(local_file, mode="r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            text_idx = 0
            if header:
                for i, col_name in enumerate(header):
                    if col_name.strip().lower() in ["review", "text", "content", "\u8bc4\u4ef7"]:
                        text_idx = i
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
                golden_dataset.append({"input": text, "ground_truth": infer_ground_truth(text)})
        if not golden_dataset:
            raise ValueError("No valid text rows found in dataset.csv")
        print(f"[Dataset] Loaded {len(golden_dataset)} samples.")
        return golden_dataset
    except Exception as exc:
        print(f"[Dataset] Failed to load local dataset: {exc}")
        return [
            {"input": "\u6628\u5929\u4e70\u7684\u624b\u673a\u5230\u4e86\u5c31\u9ed1\u5c4f\uff0c\u5feb\u9012\u5458\u6001\u5ea6\u4e5f\u5dee\u3002\u6211\u8981\u9000\u6b3e\uff0c\u8fd8\u8981\u6295\u8bc9\u4ed6\uff01\u8ba2\u5355\u53f78829103\u3002", "ground_truth": {"core_intent": REFUND, "urgency_level": HIGH, "entities": ["8829103"], "summary": "\u624b\u673a\u9ed1\u5c4f\u8981\u6c42\u9000\u6b3e"}},
            {"input": "\u4f60\u4eec\u5bb6\u679c\u5b50\u53d7\u6f6e\u4e86\uff0c\u6839\u672c\u6ca1\u6cd5\u5403\uff0c\u6211\u8981\u9000\u94b1\uff01", "ground_truth": {"core_intent": REFUND, "urgency_level": MEDIUM, "entities": [], "summary": "\u5546\u54c1\u53d7\u6f6e\u8981\u6c42\u9000\u6b3e"}},
            {"input": "\u7cfb\u7edf\u63d0\u793a\u5f02\u5730\u767b\u5f55\u5c01\u7981\uff0c\u6211\u91cc\u9762\u8fd8\u6709\u94b1\uff0c\u9a6c\u4e0a\u89e3\u5c01\u3002", "ground_truth": {"core_intent": ACCOUNT, "urgency_level": HIGH, "entities": [], "summary": "\u8d26\u53f7\u5c01\u7981\u8981\u6c42\u89e3\u5c01"}},
        ]


def prepare_demo_env(reset_state=False):
    os.makedirs("memory", exist_ok=True)
    os.makedirs("adapters", exist_ok=True)
    if not reset_state:
        return
    for file in ["memory/metrics.csv", "memory/latest_patch.json", "memory/token_usage.csv", "memory/llm_cache.jsonl"]:
        if os.path.exists(file):
            os.remove(file)
    for file in ["memory/SKILL.md", "memory/examples.json", "memory/SKILL_backup.md"]:
        if os.path.exists(file):
            os.remove(file)


def evaluate_batch_items(evaluator, batch_data, parsed_array, prompt):
    results = []
    for idx, data in enumerate(batch_data):
        prediction = json.dumps(parsed_array[idx], ensure_ascii=False, separators=(",", ":"))
        eval_result = evaluator.evaluate(prediction, data["ground_truth"])
        results.append({"data": data, "prompt": prompt, "prediction": prediction, "eval_result": eval_result})
    return results


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


def run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data, meta_intervention, enable_few_shots, enable_f1_postprocess=True):
    batch_inputs = [data["input"] for data in batch_data]
    active_categories = memory_bank.route_categories(batch_inputs)
    current_skills = memory_bank.get_skills_by_categories(active_categories)
    few_shots = memory_bank.get_few_shots(batch_inputs[0], k=1, enabled=enable_few_shots)
    prompt = build_batch_execution_prompt(config, current_skills, few_shots, batch_inputs, meta_intervention)
    prediction_text = llm.generate(prompt, temperature=0.0, model_type="default", use_cache=True, max_tokens=max(500, 220 * len(batch_inputs)))
    parsed_array = evaluator._extract_json_from_text(prediction_text)
    if isinstance(parsed_array, list) and len(parsed_array) == len(batch_data):
        return evaluate_optimized_batch_items(evaluator, f1_optimizer, batch_data, parsed_array, prompt, enabled=enable_f1_postprocess)
    if len(batch_data) <= 1:
        raw_prediction = evaluator._extract_json_from_text(prediction_text or "{}") or {}
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
    return run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data[:midpoint], meta_intervention, enable_few_shots, enable_f1_postprocess) + run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data[midpoint:], meta_intervention, enable_few_shots, enable_f1_postprocess)


def run_harness_loop(config=None):
    print("[Harness] Starting Self-Evolving Harness with token-saving defaults...")
    config_path = "adapters/ticket_config.yaml"
    config = config or load_config(config_path)
    runtime = config["runtime"]
    prepare_demo_env(reset_state=bool(runtime.get("reset_state", False)))
    llm = BaseLLMClient(cache_enabled=bool(config["cache"].get("enabled", True)), cache_path=config["cache"].get("path", "memory/llm_cache.jsonl"))
    evaluator = TaskEvaluator(config_path)
    memory_bank = MemoryBank()
    f1_optimizer = F1PostProcessor(memory_bank)
    attributor = SkillAttributor(llm)
    evolver = SkillEvolver(evaluator, llm)
    sample_size = runtime.get("sample_size", 30)
    sample_size = None if sample_size in [None, "all", "full"] else int(sample_size)
    golden_dataset = init_real_dataset(sample_size=sample_size, shuffle_seed=runtime.get("shuffle_seed", 2026))
    print(f"[Harness] Mode: {runtime.get('mode', 'demo')} | Dataset size: {len(golden_dataset)}")
    epochs = int(runtime.get("epochs", 6))
    batch_size = int(runtime.get("batch_size", 8))
    max_workers = int(runtime.get("max_workers", 6))
    enable_few_shots = bool(runtime.get("enable_few_shots", False))
    enable_f1_postprocess = bool(runtime.get("enable_f1_postprocess", True))
    enable_evolution = bool(runtime.get("enable_evolution", True))
    estimated_batches = (len(golden_dataset) + batch_size - 1) // batch_size
    print(f"[Harness] Batch size: {batch_size} | Estimated model batches per epoch: {estimated_batches} | Epochs: {epochs}")
    stuck_counter = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        print(f"\n================ Epoch {epoch} ================")
        epoch_total_f1 = 0.0
        bad_case = None
        results = []
        memory_bank.refresh_skills()
        meta_intervention = stuck_counter >= 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_batch_with_split, llm, evaluator, config, memory_bank, f1_optimizer, batch_data, meta_intervention, enable_few_shots, enable_f1_postprocess) for batch_data in chunk_dataset(golden_dataset, batch_size=batch_size)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as exc:
                    print(f"[Batch] Generation failed: {exc}")
        for result in results:
            eval_result = result["eval_result"]
            epoch_total_f1 += eval_result["f1_score"]
            if eval_result["exact_match"]:
                memory_bank.add_successful_case(result["data"]["input"], evaluator._extract_json_from_text(result["prediction"]))
            elif bad_case is None:
                bad_case = {"input": result["data"]["input"], "ground_truth": result["data"]["ground_truth"], "prediction": result["prediction"], "eval_result": eval_result}
        avg_f1 = epoch_total_f1 / len(golden_dataset) if golden_dataset else 0.0
        elapsed_ms = (time.time() - epoch_started) * 1000
        stats = llm.snapshot_stats()
        log_metrics(epoch, avg_f1, stats, elapsed_ms)
        print(f"[Epoch {epoch}] F1={avg_f1:.2f}, calls={stats['calls']}, cache_hits={stats['cache_hits']}, tokens={stats['total_tokens']}, elapsed_ms={int(elapsed_ms)}")
        if avg_f1 >= 1.0:
            print("[Harness] Reached 100% F1. Evolution complete.")
            break
        if bad_case and enable_evolution:
            print("[Harness] Found a bad case; starting adaptive patch flow.")
            patch = attributor.analyze_root_cause(bad_case["eval_result"], bad_case["input"], bad_case["prediction"], bad_case["ground_truth"])
            log_latest_patch(patch)
            baseline_f1 = avg_f1
            new_f1, success = evolver.apply_patch_with_rollback(attributor=attributor, patch_data=patch, config=config, golden_set=golden_dataset, build_prompt_func=build_batch_execution_prompt, baseline_f1=baseline_f1)
            stuck_counter = stuck_counter + 1 if (not success or new_f1 <= baseline_f1) else 0
        elif bad_case:
            print("[Harness] Bad case found, but evolution is disabled for this run.")


if __name__ == "__main__":
    args = parse_args()
    run_harness_loop(apply_cli_overrides(load_config(), args))
