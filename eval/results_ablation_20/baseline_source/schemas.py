from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class Message:
    role: str
    content: str


@dataclass(slots=True)
class ConversationState:
    session_id: str
    summary: str = ""
    known_entities: dict[str, Any] = field(default_factory=dict)
    asked_questions: list[str] = field(default_factory=list)
    last_intents: list[str] = field(default_factory=list)
    last_retrieval_targets: list[str] = field(default_factory=list)
    followup_count: int = 0
    pending_questions: list[str] = field(default_factory=list)
    initial_analysis: dict[str, Any] = field(default_factory=dict)
    pending_followup: bool = False
    satisfied_info: list[str] = field(default_factory=list)
    answer_generated: bool = False


@dataclass(slots=True)
class QueryUnderstandingResult:
    intents: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    missing_info: list[str] = field(default_factory=list)
    need_followup: bool = False
    retrieval_targets: list[str] = field(default_factory=list)
    query_type: str = "knowledge_question"
    user_goal: str = ""
    topic_relation: str = "same_topic"
    rewritten_query: str = ""
    satisfied_info: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FollowupAnalysisResult:
    entities: dict[str, Any] = field(default_factory=dict)
    missing_info: list[str] = field(default_factory=list)
    need_followup: bool = False
    satisfied_info: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TopicContinuationResult:
    is_continuation: bool = False
    rewritten_query: str = ""
    retained_entities: dict[str, Any] = field(default_factory=dict)
    retained_satisfied: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FollowupResult:
    need_followup: bool = False
    followup_questions: list[str] = field(default_factory=list)
    next_action: str = "continue"


@dataclass(slots=True)
class RetrievalPlan:
    query: str
    selected_skills: list[str] = field(default_factory=list)
    retrieval_targets: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int = 30
    final_k: int = 8


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    parent_id: str
    text: str
    title_path: str = ""
    source_name: str = "法律法规"
    score: float = 0.0
    rerank_score: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    parent_text: str = ""


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    content: str
    source_name: str
    title_path: str
    chunk_id: str = ""
    parent_id: str = ""
    score: float = 0.0
    support_type: str = "direct"
    matched_children: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvidencePack:
    items: list[EvidenceItem] = field(default_factory=list)
    status: str = "insufficient"
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SafetyReview:
    approved: bool = True
    issues: list[str] = field(default_factory=list)
    rewrite_instruction: str = ""


@dataclass(slots=True)
class FinalAnswer:
    session_id: str
    answer: str
    answer_type: str = "answer"
    citations: list[dict[str, Any]] = field(default_factory=list)
    need_followup: bool = False
    debug: dict[str, Any] = field(default_factory=dict)
