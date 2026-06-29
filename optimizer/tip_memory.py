import hashlib
import json
import os
import re
import time


class TipMemory:
    """Short-term evidence buffer before writing durable skills or examples."""

    def __init__(self, tip_file="memory/tips.jsonl"):
        self.tip_file = tip_file

    def _normalize_rule(self, text):
        return re.sub(r"\s+", " ", str(text or "").strip().lower())[:300]

    def fingerprint(self, patch_data):
        payload = {
            "root_cause_type": patch_data.get("root_cause_type"),
            "evolution_action": patch_data.get("evolution_action"),
            "target_category": patch_data.get("target_category"),
            "proposed_rule": self._normalize_rule(patch_data.get("proposed_rule")),
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _read_records(self):
        if not os.path.exists(self.tip_file):
            return []
        records = []
        with open(self.tip_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _write_records(self, records):
        os.makedirs(os.path.dirname(self.tip_file), exist_ok=True)
        with open(self.tip_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def add_or_update(self, patch_data):
        records = self._read_records()
        tip_id = self.fingerprint(patch_data)
        now = int(time.time())
        confidence = float(patch_data.get("confidence") or 0.0)

        for record in records:
            if record.get("tip_id") != tip_id:
                continue
            record["count"] = int(record.get("count") or 0) + 1
            record["last_ts"] = now
            record["confidence"] = max(float(record.get("confidence") or 0.0), confidence)
            record["source_error"] = patch_data.get("source_error") or record.get("source_error")
            record["input_excerpt"] = patch_data.get("input_excerpt") or record.get("input_excerpt")
            record["proposed_rule"] = patch_data.get("proposed_rule") or record.get("proposed_rule")
            if record.get("status") == "rejected":
                record["status"] = "buffered"
            self._write_records(records)
            return record

        record = {
            "tip_id": tip_id,
            "status": "buffered",
            "count": 1,
            "first_ts": now,
            "last_ts": now,
            "root_cause_type": patch_data.get("root_cause_type"),
            "evolution_action": patch_data.get("evolution_action"),
            "target_category": patch_data.get("target_category"),
            "confidence": confidence,
            "risk_flags": patch_data.get("risk_flags") or [],
            "source_error": patch_data.get("source_error"),
            "input_excerpt": patch_data.get("input_excerpt"),
            "proposed_rule": patch_data.get("proposed_rule"),
        }
        records.append(record)
        self._write_records(records)
        return record

    def should_promote(self, record, config):
        evolution_cfg = config.get("evolution", {})
        repeat_threshold = int(evolution_cfg.get("tip_promotion_threshold", 2))
        immediate_confidence = float(evolution_cfg.get("tip_immediate_confidence", 0.85))
        confidence = float(record.get("confidence") or 0.0)
        return int(record.get("count") or 0) >= repeat_threshold or confidence >= immediate_confidence

    def mark_promoted(self, tip_id, metrics=None):
        records = self._read_records()
        now = int(time.time())
        for record in records:
            if record.get("tip_id") == tip_id:
                record["status"] = "promoted"
                record["promoted_ts"] = now
                record["promotion_metrics"] = metrics or {}
                break
        self._write_records(records)

    def mark_rejected(self, tip_id, reason, metrics=None):
        records = self._read_records()
        for record in records:
            if record.get("tip_id") == tip_id:
                record["status"] = "rejected"
                record["reject_reason"] = reason
                record["reject_metrics"] = metrics or {}
                break
        self._write_records(records)

    def load_records(self, limit=1000):
        return self._read_records()[-limit:]
