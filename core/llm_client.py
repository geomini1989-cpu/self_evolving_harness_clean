from openai import OpenAI
from dotenv import load_dotenv
import csv
import hashlib
import json
import os
import threading
import time
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")


class BaseLLMClient:
    def __init__(self, cache_enabled=True, cache_path="memory/llm_cache.jsonl", usage_log_path="memory/token_usage.csv"):
        api_key = os.getenv("API_KEY")
        self.offline_mode = not api_key
        self.client = None

        if self.offline_mode:
            print("[LLM] API_KEY not found. Running in offline demo mode. Add API_KEY to .env to enable the real model.")
        else:
            custom_http_client = httpx.Client(limits=httpx.Limits(max_connections=100, max_keepalive_connections=50))
            self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", http_client=custom_http_client)
        self.cheap_model = os.getenv("CHEAP_MODEL", "kimi-k2.6")
        self.default_model = os.getenv("DEFAULT_MODEL", "kimi-k2.6")
        self.smart_model = os.getenv("SMART_MODEL", "kimi-k2.6")
        self.cache_enabled = cache_enabled
        self.cache_path = cache_path
        self.usage_log_path = usage_log_path
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self._lock = threading.RLock()
        self._load_disk_cache()

    def _target_model(self, model_type):
        if model_type == "cheap":
            return self.cheap_model
        if model_type == "smart":
            return self.smart_model
        return self.default_model

    def _cache_key(self, prompt, model_type, temperature, max_tokens):
        payload = {
            "model": self._target_model(model_type),
            "model_type": model_type,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_disk_cache(self):
        if not self.cache_enabled or not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = record.get("key")
                    if key and "result" in record:
                        self.cache[key] = record["result"]
        except Exception as exc:
            print(f"[Cache] Failed to load disk cache: {exc}")
            self.cache = {}

    def _append_disk_cache(self, key, result):
        if not self.cache_enabled or not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        record = {"key": key, "result": result, "ts": int(time.time())}
        with self._lock:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _log_usage(self, model_type, target_model, usage, elapsed_ms, cache_hit=False):
        os.makedirs(os.path.dirname(self.usage_log_path), exist_ok=True)
        file_exists = os.path.exists(self.usage_log_path)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0) if usage else 0
        with self._lock:
            if not cache_hit:
                self.call_count += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
        with open(self.usage_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ts", "model_type", "model", "cache_hit", "prompt_tokens", "completion_tokens", "total_tokens", "elapsed_ms"])
            writer.writerow([int(time.time()), model_type, target_model, int(cache_hit), prompt_tokens, completion_tokens, total_tokens, int(elapsed_ms)])

    def snapshot_stats(self):
        with self._lock:
            return {
                "calls": self.call_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def generate(self, prompt, temperature=0.7, model_type="default", use_cache=True, max_tokens=600):
        target_model = self._target_model(model_type)
        cacheable = self.cache_enabled and use_cache and temperature == 0.0
        cache_key = self._cache_key(prompt, model_type, temperature, max_tokens) if cacheable else None
        if cacheable:
            with self._lock:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    self.cache_hits += 1
                    print("[Cache Hit] Returning cached LLM result with 0 tokens.")
                    self._log_usage(model_type, target_model, None, 0, cache_hit=True)
                    return cached
                self.cache_misses += 1
        if self.offline_mode:
            result = self._offline_generate(prompt)
            self._log_usage(model_type, "offline-rule-engine", None, 0)
            if cacheable:
                with self._lock:
                    self.cache[cache_key] = result
                self._append_disk_cache(cache_key, result)
            return result
        started = time.time()
        try:
            response = self.client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content
            self._log_usage(model_type, target_model, getattr(response, "usage", None), (time.time() - started) * 1000)
            if cacheable:
                with self._lock:
                    self.cache[cache_key] = result
                self._append_disk_cache(cache_key, result)
            return result
        except Exception as exc:
            print(f"[LLM Error] Model {target_model} failed: {exc}")
            if target_model != self.default_model:
                print(f"[LLM Fallback] Retrying with {self.default_model}...")
                started = time.time()
                response = self.client.chat.completions.create(
                    model=self.default_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._log_usage(model_type, self.default_model, getattr(response, "usage", None), (time.time() - started) * 1000)
                return response.choices[0].message.content
            raise

    def _offline_generate(self, prompt):
        if "Return only one Markdown rule block" in prompt:
            return "## [物流投诉] 本地路由高置信度修正规则\nTrigger: 物流/快递/骑手/配送关键词命中时。\nAction: 将 core_intent 修正为物流投诉，并将 urgency_level 至少设为高。"
        if "proposed_rule" in prompt and "root_cause_analysis" in prompt:
            return json.dumps({
                "error_category": "offline_demo_attribution",
                "root_cause_analysis": "Rule-based offline attribution for demo mode.",
                "proposed_rule": "当本地路由高置信度命中业务类别时，优先采用路由类别修正 core_intent。"
            }, ensure_ascii=False)

        inputs = self._extract_prompt_inputs(prompt)
        predictions = [self._predict_single(text) for text in inputs]
        return json.dumps(predictions, ensure_ascii=False, separators=(",", ":"))

    def _extract_prompt_inputs(self, prompt):
        marker = "Inputs:"
        if marker not in prompt:
            return [prompt[-500:]]
        body = prompt.split(marker, 1)[1].strip()
        inputs = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            if ". " in line:
                _, text = line.split(". ", 1)
                inputs.append(text.strip())
            else:
                inputs.append(line)
        return inputs or [body]

    def _predict_single(self, text):
        intent = "退款纠纷"
        if any(keyword in text for keyword in ["快递", "物流", "骑手", "外卖", "配送", "态度", "迟到", "超时"]):
            intent = "物流投诉"
        elif any(keyword in text for keyword in ["封禁", "封号", "账号", "账户", "解封"]):
            intent = "账号封禁"
        elif any(keyword in text for keyword in ["系统", "网页", "崩溃", "闪退", "密码", "报错", "登录"]):
            intent = "系统Bug"
        elif any(keyword in text for keyword in ["假", "骗", "图文不符", "宣传", "虚假"]):
            intent = "虚假宣传"

        urgency = "高" if intent in {"物流投诉", "系统Bug", "账号封禁"} else "中"
        entities = []
        for token in text.replace("，", " ").replace("。", " ").replace("：", " ").split():
            digits = "".join(ch for ch in token if ch.isdigit())
            if len(digits) >= 6 and digits not in entities:
                entities.append(digits)
        return {
            "core_intent": intent,
            "urgency_level": urgency,
            "entities": entities,
            "summary": "用户负面体验客诉处理",
        }
