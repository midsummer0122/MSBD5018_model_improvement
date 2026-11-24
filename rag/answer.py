from __future__ import annotations

import json

from collections.abc import Callable

from .providers import ChatProvider
from .schemas import ConversationState, EvidencePack, Message, QueryUnderstandingResult
from .schemas import RetrievalPlan, to_dict


DISCLAIMER = "\n\n---\n以上内容仅供参考，不构成正式法律意见。具体劳动法律问题请咨询执业律师。"


def format_evidence_text(evidence: EvidencePack) -> str:
    blocks = []
    for item in evidence.items:
        child_lines = []
        for child in item.matched_children:
            score = child.get("rerank_score")
            score_text = "" if score is None else f" score={score:.4f}"
            child_lines.append(f"- {child.get('chunk_id')}{score_text}: {child.get('text')}")
        matched = "\n".join(child_lines) if child_lines else "- 无单独命中片段"
        blocks.append(
            f"[{item.evidence_id}] 来源：{item.source_name} / {item.title_path}\n"
            f"父块排序分：{item.score:.4f}\n"
            f"命中依据：\n{matched}\n"
            f"父块上下文：\n{item.content}"
        )
    return "\n\n".join(blocks)


# Compatibility for earlier callers; generation and coverage use the same formatter.
_evidence_text = format_evidence_text


def _answer_messages(
    query: str,
    messages: list[Message],
    state: ConversationState,
    understanding: QueryUnderstandingResult,
    plan: RetrievalPlan,
    evidence: EvidencePack,
    rewrite_instruction: str = "",
) -> tuple[list[dict[str, str]], str]:
    prompt = f"""
你是一个基于权威资料的劳动法知识问答助手。请基于 Evidence Pack 回答用户问题。
本项目主要面向劳动法问答（劳动合同、工资工时、解除/终止、经济补偿、工伤、社会保险、劳动争议仲裁等）。

回答语言：必须跟随用户提问的语言——用户用英文提问就用英文回答，用中文提问就用中文回答。引用法条名称、条文内容时，如来自中文证据可保留中文原样。

必须遵守：
1. 不替代执业律师，不构成正式法律意见，不替用户做出确定性的诉讼策略或结果判断。
2. 对具体个案咨询要谨慎，不承诺胜诉、不打包票、不给出确定的胜负结论。
3. 优先引用 Evidence Pack 中真实出现的法条；若 Evidence Pack 为空或未命中相关法条，可基于一般劳动法知识作答，但要明确说明"未检索到具体法条原文，以下为一般性说明，建议核实"，不得编造具体条号并谎称来自证据。
4. 引用法条时尽量精确到条/款/项；无法核实具体条文时如实说明，不凭空编造条号。
5. 如果证据不足，要明确说明不确定，并提示关键事实缺失可能影响结论。
6. 仅当 Evidence Pack 中确有命中证据时，才在文中用 [来源编号] 标注（如 [1]、[2]）；证据为空时不要加任何 [来源编号]。不要在末尾列出参考来源列表，前端已单独展示。
7. 用与用户提问一致的语言回答，结构清晰，避免过度冗长。
8. 对用户未明确的关键事实（如是否已签订书面合同、是否已申请仲裁/起诉、具体金额、时间节点、有无证据等），可做分类讨论（"如果已经……那么……；如果尚未……那么……"），不要臆断成单一事实或替用户认定。这有助于在信息不全时仍给出有用的分层分析。

如果有重写要求，请优先满足：
{rewrite_instruction or "无"}

用户问题：
{query}

最近对话：
{json.dumps([to_dict(m) for m in messages[-6:]], ensure_ascii=False)}

会话状态：
{json.dumps(to_dict(state), ensure_ascii=False)}

问题理解：
{json.dumps(to_dict(understanding), ensure_ascii=False)}

选中 Skill 与检索计划：
{json.dumps(to_dict(plan), ensure_ascii=False)}

Evidence Pack：
{format_evidence_text(evidence)}
""".strip()
    return (
        [
            {"role": "system", "content": "你是安全、谨慎、基于证据的劳动法问答助手。"},
            {"role": "user", "content": prompt},
        ],
        prompt,
    )


def generate_answer(
    chat: ChatProvider,
    query: str,
    messages: list[Message],
    state: ConversationState,
    understanding: QueryUnderstandingResult,
    plan: RetrievalPlan,
    evidence: EvidencePack,
    rewrite_instruction: str = "",
) -> tuple[str, str]:
    chat_messages, prompt = _answer_messages(
        query, messages, state, understanding, plan, evidence, rewrite_instruction=rewrite_instruction
    )
    answer = chat.chat(chat_messages, temperature=0.3)
    return answer + DISCLAIMER, prompt


def stream_generate_answer(
    chat: ChatProvider,
    query: str,
    messages: list[Message],
    state: ConversationState,
    understanding: QueryUnderstandingResult,
    plan: RetrievalPlan,
    evidence: EvidencePack,
    rewrite_instruction: str = "",
    on_delta: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    chat_messages, prompt = _answer_messages(
        query, messages, state, understanding, plan, evidence, rewrite_instruction=rewrite_instruction
    )
    parts: list[str] = []
    for delta in chat.stream_chat(chat_messages, temperature=0.3):
        parts.append(delta)
        if on_delta is not None:
            on_delta(delta)
    if on_delta is not None:
        on_delta(DISCLAIMER)
    return "".join(parts) + DISCLAIMER, prompt


def safe_fallback_answer() -> str:
    return (
        "这个问题涉及关键事实缺失或超出了当前系统可核实的范围，暂无法给出可靠回答。\n\n"
        "建议你整理好相关合同、证据材料（如合同、转账记录、聊天记录、书面通知、诉讼文书等），"
        "携带具体案情咨询执业律师，以获得针对性的法律意见。"
    )
