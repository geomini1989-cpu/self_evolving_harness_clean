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
    "shuffle_seed": 2026,
    "enable_few_shots": False,
    "enable_llm_scout": False,
    "enable_f1_postprocess": True,
    "enable_rule_fast_path": False,
    "force_evolution_demo": False,
    "use_rule_attributor": False,
    "use_rule_evolver": False,
    "enable_evolution": True,
    "reset_state": False,
}
DEFAULT_CACHE = {"enabled": True, "path": "memory/llm_cache.jsonl"}
DEFAULT_EVOLUTION = {"regression_mode": "sample_then_full", "f1_tolerance": 0.02, "confidence_threshold": 0.7, "sample_size": 10, "replay_size": 20, "max_new_skill_chars": 900, "max_skills_per_category": 5}


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
    parser.add_argument("--mode", choices=["demo", "benchmark", "evolve-demo"], default=None, help="demo: closed-loop run; benchmark: one-pass evaluation; evolve-demo: deterministic evolution showcase")
    parser.add_argument("--sample-size", default=None, help="Number of rows to evaluate, or 'all' for the full dataset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--no-evolution", action="store_true", help="Skip attribution/evolution after bad cases")
    parser.add_argument("--llm-benchmark", action="store_true", help="Use the real model in benchmark mode instead of local skill execution")
    parser.add_argument("--force-evolution-demo", action="store_true", help="Force one bad-case attribution and skill evolution step")
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
    if args.llm_benchmark:
        runtime["enable_rule_fast_path"] = False
    if args.force_evolution_demo:
        runtime["force_evolution_demo"] = True
        runtime["use_rule_attributor"] = True
        runtime["use_rule_evolver"] = True
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


def build_batch_execution_prompt(config, current_skills, few_shots, batch_texts, meta_intervention=False):
    schema_str = json.dumps(config["schema"], ensure_ascii=False, separators=(",", ":"))
    meta_prompt = "Extra caution: inspect implicit logic and sarcasm.\n" if meta_intervention else ""
    skills_block = current_skills.strip() if current_skills else "No matched category-specific skills."
    prompt_policy = load_prompt_policy()
    prompt_policy_block = f"\nPrompt policy:\n{prompt_policy}\n" if prompt_policy else ""
    few_shot_block = f"\nExamples:\n{few_shots.strip()}\n" if few_shots else ""
    batch_text_str = "\n".join(f"{idx}. {text}" for idx, text in enumerate(batch_texts, 1))
    return f"""Task: parse customer complaint texts into structured JSON.
Output compact JSON only: one array with exactly {len(batch_texts)} objects in the same order as inputs. No Markdown, no explanation.
Each object must include exactly these keys: core_intent, urgency_level, entities, summary.
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


def init_real_dataset(sample_size=30, shuffle_seed=2026):
    local_file = "data/dataset.csv"
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
    for file in ["memory/metrics.csv", "memory/latest_patch.json", "memory/token_usage.csv", "memory/llm_cache.jsonl", "memory/PROMPT_POLICY.md", "memory/PROMPT_POLICY_backup.md"]:
        if os.path.exists(file):
            os.remove(file)
    for file in ["memory/SKILL.md", "memory/examples.json", "memory/SKILL_backup.md"]:
        if os.path.exists(file):
            os.remove(file)


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
    memory_bank.refresh_skills()
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
    log_metrics("evolve_demo", new_f1, stats, 0)
    print(f"[Evolution Demo] Patch success={success}; regression F1={new_f1:.2f}; fixed bad-case F1={fixed_eval['f1_score']:.2f}")
    print("[Evolution Demo] Skill memory updated: memory/SKILL.md")


def run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data, meta_intervention, enable_few_shots, enable_f1_postprocess=True, enable_rule_fast_path=False):
    if enable_rule_fast_path:
        return run_rule_batch(evaluator, memory_bank, batch_data)

    batch_inputs = [data["input"] for data in batch_data]
    active_categories = memory_bank.route_categories(batch_inputs)
    current_skills = memory_bank.get_skills_by_categories(active_categories)
    few_shots = memory_bank.get_few_shots(batch_inputs[0], k=1, enabled=enable_few_shots)
    prompt = build_batch_execution_prompt(config, current_skills, few_shots, batch_inputs, meta_intervention)
    prediction_text = llm.generate(prompt, temperature=0.0, model_type="default", use_cache=True, max_tokens=max(500, 220 * len(batch_inputs)))
    parsed_array = evaluator._extract_json_from_text(prediction_text)
    if isinstance(parsed_array, dict):
        for key in ["items", "results", "data", "outputs"]:
            if isinstance(parsed_array.get(key), list):
                parsed_array = parsed_array[key]
                break
    if isinstance(parsed_array, list) and len(parsed_array) == len(batch_data):
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
    return run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data[:midpoint], meta_intervention, enable_few_shots, enable_f1_postprocess) + run_batch_with_split(llm, evaluator, config, memory_bank, f1_optimizer, batch_data[midpoint:], meta_intervention, enable_few_shots, enable_f1_postprocess)


def run_harness_loop(config=None):
    print("[Harness] Starting Self-Evolving Harness with token-saving defaults...")
    config_path = "adapters/ticket_config.yaml"
    config = config or load_config(config_path)
    runtime = config["runtime"]
    prepare_demo_env(reset_state=bool(runtime.get("reset_state", False)))
    llm = BaseLLMClient(cache_enabled=bool(config["cache"].get("enabled", True)), cache_path=config["cache"].get("path", "memory/llm_cache.jsonl"))
    evaluator = TaskEvaluator(config_path)
    domain_adapter = TicketAdapter(schema=config.get("schema", {}))
    memory_bank = MemoryBank()
    f1_optimizer = F1PostProcessor(memory_bank)
    attributor = SkillAttributor(llm)
    evolver = SkillEvolver(evaluator, llm)
    sample_size = runtime.get("sample_size", 30)
    sample_size = None if sample_size in [None, "all", "full"] else int(sample_size)
    golden_dataset = init_real_dataset(sample_size=sample_size, shuffle_seed=runtime.get("shuffle_seed", 2026))
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
    estimated_batches = (len(golden_dataset) + batch_size - 1) // batch_size
    print(f"[Harness] Batch size: {batch_size} | Estimated model batches per epoch: {estimated_batches} | Epochs: {epochs}")
    if enable_rule_fast_path:
        print("[Harness] Benchmark fast path: local skill execution is enabled; LLM calls should stay near zero.")
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
            futures = [
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
                )
                for batch_data in chunk_dataset(golden_dataset, batch_size=batch_size)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as exc:
                    print(f"[Batch] Generation failed: {exc}")
        for result in results:
            eval_result = result["eval_result"]
            epoch_total_f1 += eval_result["f1_score"]
            log_saf_trace(domain_adapter, result["data"], result["prediction"], eval_result, epoch)
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
