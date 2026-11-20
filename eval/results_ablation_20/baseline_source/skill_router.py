from __future__ import annotations

from .schemas import QueryUnderstandingResult, RetrievalPlan


def route_skills(query: str, understanding: QueryUnderstandingResult) -> RetrievalPlan:
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

    return RetrievalPlan(
        query=understanding.rewritten_query or query,
        selected_skills=sorted(set(skills)),
        retrieval_targets=["statute_db", "case_db"],
        filters={},
        top_k=30,
        final_k=5,
    )
