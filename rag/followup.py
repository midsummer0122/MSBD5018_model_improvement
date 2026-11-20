from __future__ import annotations

import re

from .schemas import ConversationState, FollowupResult, QueryUnderstandingResult


MAX_FOLLOWUP_ROUNDS = 4
MAX_QUESTIONS_PER_ROUND = 5


def _is_chinese(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _build_questions(
    understanding: QueryUnderstandingResult,
    state: ConversationState,
) -> list[str]:
    """追问问题完全由 LLM 的 missing_info 提供，这里只做数量上限和语义去重。"""
    questions: list[str] = []
    asked = set(state.asked_questions)
    for q in understanding.missing_info:
        if len(questions) >= MAX_QUESTIONS_PER_ROUND:
            break
        if q in asked:
            continue
        # 与已问过的问题关键词重叠过高则跳过
        if any(
            len(set(re.findall(r"[一-鿿]{2,}", q)) & set(re.findall(r"[一-鿿]{2,}", prev))) >= 3
            for prev in state.asked_questions
        ):
            continue
        questions.append(q)

    sent = set(questions)
    state.pending_questions = [q for q in understanding.missing_info if q not in sent]
    return questions


def decide_followup(
    query: str,
    understanding: QueryUnderstandingResult,
    state: ConversationState,
) -> FollowupResult:
    if understanding.query_type == "knowledge_question":
        return FollowupResult(False, [], "continue")

    if state.followup_count >= MAX_FOLLOWUP_ROUNDS:
        return FollowupResult(False, [], "max_rounds_reached")

    questions = _build_questions(understanding, state)

    has_pending = len(state.pending_questions) > 0
    need = understanding.need_followup or has_pending
    if need and questions:
        return FollowupResult(True, questions, "return_followup")
    return FollowupResult(False, [], "continue")


def render_followup(followup: FollowupResult, followup_count: int = 0, query: str = "") -> str:
    """渲染追问。引导语跟随用户提问语言（中文/英文），追问问题本身保持 LLM 生成的语言。"""
    if _is_chinese(query):
        if followup_count == 0:
            intro = "这个问题涉及具体个案，信息不足时不适合直接给出法律结论，请先补充以下信息："
        elif followup_count < MAX_FOLLOWUP_ROUNDS - 1:
            intro = "还有以下信息需要补充："
        else:
            intro = "这是最后需要确认的信息："
    else:
        if followup_count == 0:
            intro = "This question involves a specific case. More information is needed before giving a legal conclusion. Please provide the following:"
        elif followup_count < MAX_FOLLOWUP_ROUNDS - 1:
            intro = "A few more details are needed:"
        else:
            intro = "Last items to confirm:"

    lines = [intro]
    lines.extend(
        f"{idx}. {question}"
        for idx, question in enumerate(followup.followup_questions, start=1)
    )
    return "\n".join(lines)
