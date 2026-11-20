from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .answer import safe_fallback_answer, stream_generate_answer
from .config import AppConfig
from .evidence import build_evidence
from .followup import decide_followup, render_followup
from .memory import MemoryStore
from .providers import Providers
from .query_understanding import analyze_followup_response, classify_topic_continuation, merge_followup_into_understanding, understand_query
from .retrieval import Retriever
from .safety import llm_safety_check
from .schemas import ConversationState, FinalAnswer, Message, to_dict
from .skill_router import route_skills
from .source_policy import chunk_source
from .coverage import check_evidence_coverage,evidence_source


StepCallback = Callable[[str, Any], None]
AnswerDeltaCallback = Callable[[str], None]
AnswerEventCallback = Callable[[str, dict[str, Any]], None]


def _iter_answer_chunks(text: str, chunk_size: int = 18):
    buffer = ""
    boundaries = "。！？；\n"
    for char in text:
        buffer += char
        if char in boundaries or len(buffer) >= chunk_size:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


class AgenticRagWorkflow:
    def __init__(
        self,
        config: AppConfig | None = None,
        memory: MemoryStore | None = None,
        providers: Providers | None = None,
        show_prompt: bool = False,
    ) -> None:
        self.config = config or AppConfig.load()
        self.memory = memory or MemoryStore()
        self.providers = providers or Providers(self.config)
        self.show_prompt = show_prompt
        self.retriever = Retriever(
            config=self.config,
            embedding=self.providers.embedding,
            qdrant=self.providers.qdrant,
            rerank=self.providers.rerank,
        )

    def run(
        self,
        session_id: str | None,
        message: str,
        on_step: StepCallback | None = None,
        on_answer_delta: AnswerDeltaCallback | None = None,
        on_answer_event: AnswerEventCallback | None = None,
        *,
        enable_coverage_check: bool = False,
        retrieval_mode: str = "dense",
    ) -> FinalAnswer:
        session_id, messages, state = self.memory.load(session_id)
        query = message.strip()
        if not query:
            return FinalAnswer(session_id=session_id, answer="Please enter a question.", answer_type="error")

        messages.append(Message(role="user", content=query))
        debug: dict[str, object] = {}

        def emit(step: str, payload: Any) -> None:
            if on_step is not None:
                on_step(step, to_dict(payload))

        def emit_answer_event(event: str, payload: dict[str, Any]) -> None:
            if on_answer_event is not None:
                on_answer_event(event, payload)

        is_continuation = False
        continuation = None
        if state.pending_followup:
            # 追问回应分析：沿用当前话题的上下文
            emit("step_status", {"step": "followup_analysis", "status": "started"})
            followup_analysis = analyze_followup_response(
                self.providers.chat, query, messages, state
            )
            understanding = merge_followup_into_understanding(
                state.initial_analysis, followup_analysis, state, query
            )
            debug["followup_analysis"] = to_dict(followup_analysis)
            emit("followup_analysis", {
                **to_dict(followup_analysis),
                "followup_round": state.followup_count,
            })
            emit("step_status", {"step": "followup_analysis", "status": "completed"})
        else:
            if state.answer_generated and state.initial_analysis:
                emit("step_status", {"step": "topic_continuation", "status": "started"})
                continuation = classify_topic_continuation(
                    self.providers.chat, query, messages, state
                )
                debug["topic_continuation"] = to_dict(continuation)
                is_continuation = continuation.is_continuation
                emit("topic_continuation", {"action": "continue" if is_continuation else "restart"})
                emit("step_status", {"step": "topic_continuation", "status": "completed"})

            if is_continuation:
                # 延续原话题（答案后的追问）：复用初始理解，继续补信息
                continuation_analysis = analyze_followup_response(
                    self.providers.chat, query, messages, state
                )
                understanding = merge_followup_into_understanding(
                    state.initial_analysis,
                    continuation_analysis,
                    state,
                    continuation.rewritten_query or query,
                )
                understanding.rewritten_query = continuation.rewritten_query or query
                state.pending_questions.clear()
                state.followup_count = 0
            else:
                # 新话题（第一问或第二个话题）：按信息延续判断，选择性保留话题级信息
                emit("step_status", {"step": "pre_analysis", "status": "started"})
                if continuation is not None:
                    state.known_entities = dict(continuation.retained_entities or {})
                    state.satisfied_info = list(continuation.retained_satisfied or [])
                    retained = set(state.satisfied_info)
                    state.asked_questions = [q for q in state.asked_questions if q in retained]
                else:
                    state.known_entities.clear()
                    state.satisfied_info.clear()
                    state.asked_questions.clear()
                state.pending_questions.clear()
                state.followup_count = 0
                understanding = understand_query(self.providers.chat, query, [messages[-1]], state)
                for item in understanding.satisfied_info:
                    if item not in state.satisfied_info:
                        state.satisfied_info.append(item)
                state.initial_analysis = to_dict(understanding)
                debug["query_understanding"] = to_dict(understanding)
                emit("query_understanding", understanding)
                emit("step_status", {"step": "pre_analysis", "status": "completed"})

        # 追问决策：目的是补充缺失信息，而非风险把控
        emit("status_line", {"text": "Checking if more information is needed..."})
        followup = decide_followup(query, understanding, state)
        debug["followup_result"] = to_dict(followup)
        emit("followup_result", followup)
        if followup.need_followup:
            answer = render_followup(followup, state.followup_count, query)
            messages.append(Message(role="assistant", content=answer))
            self._update_state(state, understanding, followup.followup_questions)
            state.followup_count += 1
            state.pending_followup = True
            self.memory.save(session_id, messages, state)
            return FinalAnswer(
                session_id=session_id,
                answer=answer,
                answer_type="followup",
                need_followup=True,
                debug=debug,
            )

        emit("step_status", {"step": "planning", "status": "started"})
        plan = route_skills(query, understanding, enable_coverage_check=enable_coverage_check,
                            retrieval_mode=retrieval_mode)
        debug["retrieval_plan"] = to_dict(plan)
        emit('retrieval_progress',{'stage':'plan','status':'planned','mode':plan.retrieval_mode,
             'targets':plan.retrieval_targets,'top_k':plan.top_k,'rerank_k':plan.final_k*4,
             'final_k':plan.final_k,'rerank_quotas':plan.rerank_quotas,'evidence_quotas':plan.evidence_quotas})
        emit("step_status", {"step": "planning", "status": "completed"})

        emit("step_status", {"step": "retrieval", "status": "started"})
        chunks = self.retriever.retrieve(plan,on_progress=lambda payload:emit('retrieval_progress',payload))
        debug["retrieved_chunks"] = [to_dict(chunk) for chunk in chunks]
        debug["initial_retrieval"] = self.retriever.last_initial_trace
        debug["rerank_fallback"] = self.retriever.last_rerank_fallback
        emit("step_status", {"step": "retrieval", "status": "completed"})

        emit("step_status", {"step": "evidence", "status": "started"})
        emit('retrieval_progress',{'stage':'evidence','status':'started'})
        evidence = build_evidence(chunks, final_k=plan.final_k, evidence_quotas=plan.evidence_quotas)
        source_by_parent={c.parent_id:chunk_source(c,required=False) for c in chunks}
        emit('retrieval_progress',{'stage':'evidence','status':'completed','count':len(evidence.items),
             'statute_count':sum(source_by_parent.get(e.parent_id)=='statute' for e in evidence.items),
             'case_count':sum(source_by_parent.get(e.parent_id)=='case' for e in evidence.items)})
        debug["evidence_pack"] = to_dict(evidence)
        emit("step_status", {"step": "evidence", "status": "completed"})

        debug['initial_evidence_pack']=to_dict(evidence)
        initial_sources={e.parent_id:e.source_name for e in evidence.items}
        def coverage_display(payload):
            diagnostic=payload.get('diagnosis',{})
            comparison=payload.get('comparison',{})
            candidates=payload.get('new_candidates',[])
            new_sources={e['parent_id']:e['source_name'] for e in candidates}
            return {'status':payload['status'],'intent':payload.get('intent',''),
                'reason':payload.get('reason',''),'checked':payload.get('checked',False),
                'missing_aspect':diagnostic.get('missing_aspect',''),
                'supplementary_query':diagnostic.get('supplementary_query',''),
                'target_source':diagnostic.get('target_source',''),
                'triggered_search':payload.get('triggered_search',False),'replaced':payload.get('replaced',False),
                'child_count':payload.get('supplementary_child_count'),
                'new_parent_count':len(candidates) if 'new_candidates' in payload else None,
                'comparison_count':payload.get('unified_rerank_input_count',len(evidence.items)+len(candidates)) if candidates else None,
                'comparison':{**comparison,'new_title':new_sources.get(comparison.get('new_parent_id'),''),
                              'old_title':initial_sources.get(comparison.get('old_parent_id'),'')} if comparison else None,
                'gap_resolution_verified':False}
        def coverage_progress(payload):
            if payload['status']=='checking':emit('step_status',{'step':'coverage','status':'started'})
            if payload['status'] in ('checking','searching','reranking'):
                emit('coverage_check',coverage_display(payload))
        evidence,coverage=check_evidence_coverage(understanding,plan,evidence,self.retriever,
                                                self.providers.chat,on_progress=coverage_progress)
        debug['coverage_check']=coverage
        debug['evidence_pack']=to_dict(evidence)
        if coverage['checked']:emit('step_status',{'step':'coverage','status':'completed'})
        emit('coverage_check',coverage_display(coverage))
        if coverage['replaced']:
            emit('retrieval_progress',{'stage':'evidence','status':'completed','count':len(evidence.items),
                 'statute_count':sum(evidence_source(e)=='statute' for e in evidence.items),
                 'case_count':sum(evidence_source(e)=='case' for e in evidence.items)})

        emit("step_status", {"step": "answer_generation", "status": "started"})
        answer, final_prompt = stream_generate_answer(
            self.providers.chat,
            query,
            messages,
            state,
            understanding,
            plan,
            evidence,
            on_delta=on_answer_delta,
        )
        debug["final_prompt"] = final_prompt
        emit("step_status", {"step": "answer_generation", "status": "completed"})
        if self.show_prompt:
            print("\n===== FINAL PROMPT =====")
            print(final_prompt)
            print("===== END FINAL PROMPT =====\n")

        emit("step_status", {"step": "safety_review", "status": "started"})
        safety = llm_safety_check(self.providers.chat, query, answer, evidence)
        debug["safety_review"] = to_dict(safety)
        emit("step_status", {"step": "safety_review", "status": "completed"})

        if not safety.approved:
            emit_answer_event(
                "answer_rejected",
                {
                    "stage": "draft",
                    "issues": safety.issues,
                    "rewrite_instruction": safety.rewrite_instruction,
                },
            )
            emit("step_status", {"step": "answer_rewrite", "status": "started"})
            answer, rewrite_prompt = stream_generate_answer(
                self.providers.chat,
                query,
                messages,
                state,
                understanding,
                plan,
                evidence,
                rewrite_instruction=safety.rewrite_instruction,
                on_delta=on_answer_delta,
            )
            debug["rewrite_prompt"] = rewrite_prompt
            emit("step_status", {"step": "answer_rewrite", "status": "completed"})
            if self.show_prompt:
                print("\n===== REWRITE PROMPT =====")
                print(rewrite_prompt)
                print("===== END REWRITE PROMPT =====\n")

            emit("step_status", {"step": "safety_review_after_rewrite", "status": "started"})
            safety = llm_safety_check(self.providers.chat, query, answer, evidence)
            debug["safety_review_after_rewrite"] = to_dict(safety)
            emit("step_status", {"step": "safety_review_after_rewrite", "status": "completed"})

            if not safety.approved:
                emit_answer_event(
                    "answer_rejected",
                    {
                        "stage": "rewrite",
                        "issues": safety.issues,
                        "rewrite_instruction": safety.rewrite_instruction,
                    },
                )
                answer = safe_fallback_answer()
                emit_answer_event(
                    "answer_fallback",
                    {
                        "reason": "rewrite_failed_safety_review",
                        "issues": safety.issues,
                    },
                )
                if on_answer_delta is not None:
                    for chunk in _iter_answer_chunks(answer):
                        on_answer_delta(chunk)

        citations = [
            {"source_name": item.source_name, "title_path": item.title_path}
            for item in evidence.items
        ]
        messages.append(Message(role="assistant", content=answer))
        self._update_state(state, understanding, [])
        state.answer_generated = True
        # 保留会话历史（短期记忆），供后续话题延续；长期记忆已通过“不落盘 + 前端不持久化 session_id”去除
        self.memory.save(session_id, messages, state)

        return FinalAnswer(
            session_id=session_id,
            answer=answer,
            answer_type="answer",
            citations=citations,
            need_followup=False,
            debug=debug,
        )

    def _update_state(
        self,
        state: ConversationState,
        understanding,
        asked_questions: list[str],
    ) -> None:
        state.last_intents = understanding.intents
        state.last_retrieval_targets = understanding.retrieval_targets
        for key, value in understanding.entities.items():
            if not value:
                continue
            current = state.known_entities.get(key)
            if isinstance(current, list) and isinstance(value, list):
                state.known_entities[key] = sorted({*current, *value})
            else:
                state.known_entities[key] = value
        for question in asked_questions:
            if question not in state.asked_questions:
                state.asked_questions.append(question)
        for item in getattr(understanding, "satisfied_info", []):
            if item not in state.satisfied_info:
                state.satisfied_info.append(item)
        if not asked_questions:
            state.pending_followup = False


def run_agentic_rag(
    session_id: str | None,
    message: str,
    on_step: StepCallback | None = None,
    on_answer_delta: AnswerDeltaCallback | None = None,
    on_answer_event: AnswerEventCallback | None = None,
    *,
    enable_coverage_check: bool = False,
    retrieval_mode: str = "dense",
) -> FinalAnswer:
    return AgenticRagWorkflow().run(
        session_id,
        message,
        on_step=on_step,
        on_answer_delta=on_answer_delta,
        on_answer_event=on_answer_event,
        enable_coverage_check=enable_coverage_check,
        retrieval_mode=retrieval_mode,
    )
