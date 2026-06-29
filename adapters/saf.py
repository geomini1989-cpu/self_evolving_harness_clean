from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HarnessState:
    domain: str
    modality: str
    raw_input: Any
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessAction:
    name: str
    payload: dict[str, Any]
    confidence: float = 0.0
    skills_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessFeedback:
    score: float
    exact_match: bool
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateActionFeedbackTrace:
    state: HarnessState
    action: HarnessAction
    feedback: HarnessFeedback

    def to_record(self):
        return {
            "state": {
                "domain": self.state.domain,
                "modality": self.state.modality,
                "raw_input": self.state.raw_input,
                "context": self.state.context,
                "metadata": self.state.metadata,
            },
            "action": {
                "name": self.action.name,
                "payload": self.action.payload,
                "confidence": self.action.confidence,
                "skills_used": self.action.skills_used,
                "metadata": self.action.metadata,
            },
            "feedback": {
                "score": self.feedback.score,
                "exact_match": self.feedback.exact_match,
                "errors": self.feedback.errors,
                "metrics": self.feedback.metrics,
            },
        }


class DomainAdapter(ABC):
    domain = "generic"
    modality = "text"

    @abstractmethod
    def to_state(self, raw_input, **metadata):
        raise NotImplementedError

    @abstractmethod
    def build_action(self, prediction, **metadata):
        raise NotImplementedError

    @abstractmethod
    def to_feedback(self, eval_result, **metadata):
        raise NotImplementedError

    def make_trace(self, raw_input, prediction, eval_result, **metadata):
        return StateActionFeedbackTrace(
            state=self.to_state(raw_input, **metadata),
            action=self.build_action(prediction, **metadata),
            feedback=self.to_feedback(eval_result, **metadata),
        )
