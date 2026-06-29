import json

from adapters.saf import DomainAdapter, HarnessAction, HarnessFeedback, HarnessState


class TicketAdapter(DomainAdapter):
    domain = "customer_complaint"
    modality = "text"

    def __init__(self, schema=None):
        self.schema = schema or {}

    def to_state(self, raw_input, **metadata):
        return HarnessState(
            domain=self.domain,
            modality=self.modality,
            raw_input=raw_input,
            context={
                "schema": self.schema,
                "task": "complaint_extraction",
            },
            metadata=metadata,
        )

    def build_action(self, prediction, **metadata):
        payload = self._coerce_payload(prediction)
        return HarnessAction(
            name="extract_structured_ticket",
            payload=payload,
            confidence=float(metadata.get("confidence", 0.0)),
            skills_used=list(metadata.get("skills_used", [])),
            metadata={
                "adapter": self.domain,
                "modality": self.modality,
            },
        )

    def to_feedback(self, eval_result, **metadata):
        error_reason = eval_result.get("error_reason") or ""
        errors = [item.strip() for item in error_reason.split("|") if item.strip()]
        return HarnessFeedback(
            score=float(eval_result.get("f1_score", 0.0)),
            exact_match=bool(eval_result.get("exact_match", False)),
            errors=errors,
            metrics={
                "is_valid_json": bool(eval_result.get("is_valid_json", False)),
                "primary_metric": "f1_score",
            },
        )

    def _coerce_payload(self, prediction):
        if isinstance(prediction, dict):
            return prediction
        if isinstance(prediction, str):
            try:
                parsed = json.loads(prediction)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"raw_output": prediction}
        return {"raw_output": str(prediction)}
