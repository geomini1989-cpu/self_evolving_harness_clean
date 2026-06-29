import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.saf import DomainAdapter, HarnessAction, HarnessFeedback, HarnessState


@dataclass
class TransferCase:
    case_id: str
    raw_input: Any
    ground_truth: dict[str, Any]


@dataclass
class TransferSuite:
    domain: str
    modality: str
    action_name: str
    schema_keys: list[str]
    cases: list[TransferCase]
    predictor: Callable[[Any], dict[str, Any]]


class TransferAdapter(DomainAdapter):
    def __init__(self, suite: TransferSuite):
        self.domain = suite.domain
        self.modality = suite.modality
        self.action_name = suite.action_name
        self.schema_keys = suite.schema_keys

    def to_state(self, raw_input, **metadata):
        return HarnessState(
            domain=self.domain,
            modality=self.modality,
            raw_input=raw_input,
            context={
                "schema_keys": self.schema_keys,
                "transfer_task": self.action_name,
            },
            metadata=metadata,
        )

    def build_action(self, prediction, **metadata):
        return HarnessAction(
            name=self.action_name,
            payload=prediction if isinstance(prediction, dict) else {"raw_output": prediction},
            confidence=float(metadata.get("confidence", 1.0)),
            skills_used=list(metadata.get("skills_used", ["local_transfer_policy"])),
            metadata={
                "adapter": self.domain,
                "modality": self.modality,
            },
        )

    def to_feedback(self, eval_result, **metadata):
        return HarnessFeedback(
            score=float(eval_result.get("score", 0.0)),
            exact_match=bool(eval_result.get("exact_match", False)),
            errors=list(eval_result.get("errors", [])),
            metrics={
                "primary_metric": "field_f1",
                "field_scores": eval_result.get("field_scores", {}),
            },
        )


class TransferEvaluator:
    def __init__(self, schema_keys):
        self.schema_keys = schema_keys

    def evaluate(self, prediction, ground_truth):
        errors = []
        field_scores = {}
        for key in self.schema_keys:
            score = self._score_value(prediction.get(key), ground_truth.get(key))
            field_scores[key] = score
            if score < 1.0:
                errors.append(f"{key}: pred={prediction.get(key)} gt={ground_truth.get(key)}")
        score = sum(field_scores.values()) / len(self.schema_keys) if self.schema_keys else 0.0
        return {
            "score": score,
            "exact_match": not errors,
            "errors": errors,
            "field_scores": field_scores,
        }

    def _score_value(self, pred, gt):
        if isinstance(gt, list):
            pred_set = {str(item).strip() for item in pred or [] if str(item).strip()}
            gt_set = {str(item).strip() for item in gt if str(item).strip()}
            if not pred_set and not gt_set:
                return 1.0
            if not pred_set or not gt_set:
                return 0.0
            overlap = len(pred_set & gt_set)
            precision = overlap / len(pred_set)
            recall = overlap / len(gt_set)
            return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return 1.0 if str(pred).strip() == str(gt).strip() else 0.0


def extract_entities(text):
    entities = []
    for match in re.findall(r"[A-Za-z]+-?[A-Za-z]*\d[A-Za-z0-9-]*", str(text)):
        if match not in entities:
            entities.append(match)
    for match in re.findall(r"\d+(?:\.\d+)?(?:元|块|GB|%)", str(text)):
        if match not in entities:
            entities.append(match)
    return entities


def predict_ticket(text):
    if any(key in text for key in ["退款", "退钱", "赔偿", "质量", "坏了"]):
        intent = "退款纠纷"
    elif any(key in text for key in ["物流", "快递", "配送", "超时", "丢件"]):
        intent = "物流投诉"
    elif any(key in text for key in ["账号", "封禁", "解封", "登录"]):
        intent = "账号封禁"
    elif any(key in text for key in ["系统", "闪退", "报错", "支付失败"]):
        intent = "系统Bug"
    else:
        intent = "虚假宣传"
    urgency = "高" if any(key in text for key in ["马上", "立刻", "投诉", "封禁", "超时"]) else "中"
    return {
        "core_intent": intent,
        "urgency_level": urgency,
        "entities": extract_entities(text),
        "summary": "用户负面体验客诉处理",
    }


def predict_visual_quality(image_meta):
    text = " ".join(str(value) for value in image_meta.values())
    if any(key in text for key in ["破损", "裂纹", "scratch", "crack"]):
        defect = "surface_damage"
    elif any(key in text for key in ["缺件", "missing"]):
        defect = "missing_part"
    else:
        defect = "label_mismatch"
    severity = "high" if any(key in text for key in ["严重", "crack", "破损"]) else "medium"
    return {
        "defect_type": defect,
        "severity": severity,
        "entities": extract_entities(text),
        "summary": "visual qa issue",
    }


def predict_speech_intent(audio_state):
    transcript = audio_state.get("transcript", "")
    if any(key in transcript for key in ["转人工", "客服", "人工"]):
        intent = "handoff_agent"
    elif any(key in transcript for key in ["取消", "退款"]):
        intent = "cancel_or_refund"
    else:
        intent = "order_status"
    urgency = "high" if any(key in transcript for key in ["马上", "现在", "投诉"]) else "medium"
    return {
        "intent": intent,
        "urgency": urgency,
        "entities": extract_entities(transcript),
        "summary": "voice request routed",
    }


def predict_recommendation(events):
    joined = " ".join(str(item) for item in events)
    if any(key in joined for key in ["hide", "not_interested", "跳过"]):
        action = "downrank_topic"
    elif any(key in joined for key in ["buy", "cart", "收藏"]):
        action = "boost_similar"
    else:
        action = "explore_more"
    confidence = "high" if len(events) >= 3 else "medium"
    return {
        "next_action": action,
        "confidence": confidence,
        "entities": extract_entities(joined),
        "summary": "recommendation policy update",
    }


def build_suites():
    return [
        TransferSuite(
            domain="customer_complaint",
            modality="text",
            action_name="extract_structured_ticket",
            schema_keys=["core_intent", "urgency_level", "entities", "summary"],
            predictor=predict_ticket,
            cases=[
                TransferCase("ticket_001", "订单号A8899的手机坏了，马上退款", {"core_intent": "退款纠纷", "urgency_level": "高", "entities": ["A8899"], "summary": "用户负面体验客诉处理"}),
                TransferCase("ticket_002", "快递超时三小时还没到，投诉配送", {"core_intent": "物流投诉", "urgency_level": "高", "entities": [], "summary": "用户负面体验客诉处理"}),
            ],
        ),
        TransferSuite(
            domain="visual_quality",
            modality="image_metadata",
            action_name="inspect_visual_defect",
            schema_keys=["defect_type", "severity", "entities", "summary"],
            predictor=predict_visual_quality,
            cases=[
                TransferCase("vision_001", {"ocr": "SN-X901", "caption": "外壳严重破损 crack"}, {"defect_type": "surface_damage", "severity": "high", "entities": ["SN-X901"], "summary": "visual qa issue"}),
                TransferCase("vision_002", {"ocr": "KIT7788", "caption": "missing screw 缺件"}, {"defect_type": "missing_part", "severity": "medium", "entities": ["KIT7788"], "summary": "visual qa issue"}),
            ],
        ),
        TransferSuite(
            domain="speech_service",
            modality="audio_transcript",
            action_name="route_voice_request",
            schema_keys=["intent", "urgency", "entities", "summary"],
            predictor=predict_speech_intent,
            cases=[
                TransferCase("speech_001", {"transcript": "我要马上转人工客服，订单B7712"}, {"intent": "handoff_agent", "urgency": "high", "entities": ["B7712"], "summary": "voice request routed"}),
                TransferCase("speech_002", {"transcript": "帮我查一下订单C3321到哪里了"}, {"intent": "order_status", "urgency": "medium", "entities": ["C3321"], "summary": "voice request routed"}),
            ],
        ),
        TransferSuite(
            domain="recommendation_flow",
            modality="event_stream",
            action_name="update_recommendation_policy",
            schema_keys=["next_action", "confidence", "entities", "summary"],
            predictor=predict_recommendation,
            cases=[
                TransferCase("rec_001", ["view:phone", "cart:SKU7788", "buy:SKU7788"], {"next_action": "boost_similar", "confidence": "high", "entities": ["SKU7788"], "summary": "recommendation policy update"}),
                TransferCase("rec_002", ["view:makeup", "hide:SKU9021", "not_interested:beauty"], {"next_action": "downrank_topic", "confidence": "high", "entities": ["SKU9021"], "summary": "recommendation policy update"}),
            ],
        ),
    ]


def run_suite(suite):
    adapter = TransferAdapter(suite)
    evaluator = TransferEvaluator(suite.schema_keys)
    traces = []
    scores = []
    for case in suite.cases:
        prediction = suite.predictor(case.raw_input)
        eval_result = evaluator.evaluate(prediction, case.ground_truth)
        trace = adapter.make_trace(
            case.raw_input,
            prediction,
            eval_result,
            case_id=case.case_id,
            ground_truth=case.ground_truth,
            transfer_runner=True,
        )
        traces.append(trace.to_record())
        scores.append(eval_result["score"])
    return {
        "domain": suite.domain,
        "modality": suite.modality,
        "action_name": suite.action_name,
        "case_count": len(suite.cases),
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "exact_match_rate": sum(1 for item in traces if item["feedback"]["exact_match"]) / len(traces) if traces else 0.0,
        "traces": traces,
    }


def write_outputs(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_traces = [trace for result in results for trace in result["traces"]]
    report = {
        "ts": int(time.time()),
        "runner": "transfer_test_runner",
        "suite_count": len(results),
        "case_count": sum(result["case_count"] for result in results),
        "avg_score": sum(result["avg_score"] for result in results) / len(results) if results else 0.0,
        "suites": [{key: value for key, value in result.items() if key != "traces"} for result in results],
    }
    with open(os.path.join(output_dir, "transfer_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "transfer_traces.jsonl"), "w", encoding="utf-8") as f:
        for trace in all_traces:
            f.write(json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run cross-domain transfer checks for the SAF harness.")
    parser.add_argument("--suite", default="all", help="Suite domain to run, or all.")
    parser.add_argument("--output-dir", default="memory", help="Directory for transfer_report.json and transfer_traces.jsonl.")
    parser.add_argument("--min-score", type=float, default=0.95, help="Fail if average transfer score is below this threshold.")
    args = parser.parse_args()

    suites = build_suites()
    if args.suite != "all":
        suites = [suite for suite in suites if suite.domain == args.suite]
    if not suites:
        raise SystemExit(f"Unknown suite: {args.suite}")

    results = [run_suite(suite) for suite in suites]
    report = write_outputs(results, args.output_dir)
    print(f"[Transfer] suites={report['suite_count']} cases={report['case_count']} avg_score={report['avg_score']:.2f}")
    for result in results:
        print(f"[Transfer] {result['domain']} ({result['modality']}): avg_score={result['avg_score']:.2f}, exact={result['exact_match_rate']:.2%}")
    if report["avg_score"] < args.min_score:
        raise SystemExit(f"[Transfer] Failed threshold: {report['avg_score']:.2f} < {args.min_score:.2f}")
    print(f"[Transfer] Report written to {os.path.join(args.output_dir, 'transfer_report.json')}")


if __name__ == "__main__":
    main()
