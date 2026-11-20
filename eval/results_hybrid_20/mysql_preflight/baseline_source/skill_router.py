from __future__ import annotations

from .schemas import QueryUnderstandingResult, RetrievalPlan
from .source_policy import source_quotas


def route_skills(query: str, understanding: QueryUnderstandingResult, *,
                 top_k: int = 30, final_k: int = 5,
                 retrieval_targets: list[str] | None = None) -> RetrievalPlan:
    skills: list[str] = []

    if "法律咨询" in understanding.intents:
        skills.append("legal_consultation")
    if "法条解释" in understanding.intents:
        skills.append("statute_explanation")
    if "程序咨询" in understanding.intents:
        skills.append("procedure_consultation")
    if "案例检索" in understanding.intents:
        skills.append("case_search")
    if not skills:
        skills.append("legal_consultation")

    targets = retrieval_targets if retrieval_targets is not None else ["statute_db", "case_db"]
    return RetrievalPlan(
        query=understanding.rewritten_query or query,
        selected_skills=sorted(set(skills)),
        retrieval_targets=targets,
        filters={},
        top_k=top_k,
        final_k=final_k,
        split_rerank=True,
        rerank_quotas=source_quotas(understanding, final_k * 4, targets),
        evidence_quotas=source_quotas(understanding, final_k, targets),
    )
