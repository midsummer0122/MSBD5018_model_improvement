from __future__ import annotations

import json
import queue
import threading
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from rag.memory import MemoryStore
from rag.schemas import to_dict
from rag.skill_router import RETRIEVAL_MODES
from rag.workflow import AgenticRagWorkflow, run_agentic_rag


app = Flask(__name__)
memory_store = MemoryStore()


STEP_LABELS = {
    "query_understanding": "Understanding",
    "followup_result": "Follow-up Decision",
    "followup_analysis": "Follow-up Analysis",
    "status_line": "Status",
    "step_status": "Progress",
    "retrieval_progress": "Retrieval route",
    "coverage_check": "Evidence coverage",
}


STATUS_LABELS = {
    "pre_analysis": "Analyzing question",
    "followup_analysis": "Analyzing follow-up",
    "planning": "Planning retrieval",
    "retrieval": "Retrieving",
    "evidence": "Building evidence",
    "coverage": "Checking evidence coverage",
    "answer_generation": "Generating answer",
    "safety_review": "Safety review",
    "answer_rewrite": "Rewriting answer",
    "safety_review_after_rewrite": "Re-reviewing",
}


def compact_payload(step: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if step == "query_understanding":
        entities = {
            key: value
            for key, value in (payload.get("entities") or {}).items()
            if value not in (None, "", [], {})
        }
        return {
            "intents": payload.get("intents") or [],
            "entities": entities,
            "user_goal": payload.get("user_goal") or "",
            "rewritten_query": payload.get("rewritten_query") or "",
            "query_type": payload.get("query_type") or "",
            "topic_relation": payload.get("topic_relation") or "",
            "missing_info": payload.get("missing_info") or [],
        }
    if step == "followup_result":
        next_action = payload.get("next_action") or "continue"
        need_raw = bool(payload.get("need_followup"))
        if next_action == "max_rounds_reached":
            need_display = "Max follow-up rounds reached"
        else:
            need_display = need_raw
        return {
            "need_followup": need_display,
            "next_action": next_action,
            "followup_questions": payload.get("followup_questions") or [],
        }
    if step == "followup_analysis":
        entities = {
            key: value
            for key, value in (payload.get("entities") or {}).items()
            if value not in (None, "", [], {})
        }
        return {
            "entities": entities,
            "missing_info": payload.get("missing_info") or [],
            "need_followup": bool(payload.get("need_followup")),
            "followup_round": payload.get("followup_round") or 0,
        }
    if step == "status_line":
        return {"text": str(payload.get("text", ""))}
    if step == "step_status":
        step_name = str(payload.get("step") or "")
        return {
            "step": step_name,
            "label": STATUS_LABELS.get(step_name, step_name),
            "status": payload.get("status") or "started",
        }
    return payload


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/session/new")
def new_session() -> Response:
    return jsonify({"session_id": memory_store.new_session_id()})


@app.get("/api/session/<session_id>/messages")
def session_messages(session_id: str) -> Response:
    loaded_session_id, messages, state = memory_store.load(session_id)
    return jsonify(
        {
            "session_id": loaded_session_id,
            "messages": [to_dict(item) for item in messages],
            "state": to_dict(state),
        }
    )


@app.post("/api/chat")
def chat() -> Response:
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or "").strip()
    session_id = payload.get("session_id")
    enable_coverage_check = payload.get("enable_coverage_check", False)
    retrieval_mode = payload.get("retrieval_mode", "dense")
    if retrieval_mode not in RETRIEVAL_MODES:
        return jsonify({"error": "retrieval_mode must be dense, dense_bm25 or hybrid."}), 400
    if type(enable_coverage_check) is not bool:
        return jsonify({"error": "enable_coverage_check must be a boolean."}), 400
    if not message:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        result = run_agentic_rag(session_id, message, enable_coverage_check=enable_coverage_check,
                                retrieval_mode=retrieval_mode)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(to_dict(result))


@app.post("/api/chat/stream")
def chat_stream() -> Response:
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or "").strip()
    session_id = payload.get("session_id")
    enable_coverage_check = payload.get("enable_coverage_check", False)
    retrieval_mode = payload.get("retrieval_mode", "dense")
    if retrieval_mode not in RETRIEVAL_MODES:
        return Response(sse_event("error", {"error": "retrieval_mode must be dense, dense_bm25 or hybrid."}),
                        mimetype="text/event-stream", status=400)
    if type(enable_coverage_check) is not bool:
        return Response(sse_event("error", {"error": "enable_coverage_check must be a boolean."}),
                        mimetype="text/event-stream", status=400)
    if not message:
        return Response(
            sse_event("error", {"error": "Please enter a question."}),
            mimetype="text/event-stream",
            status=400,
        )

    events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

    def on_step(step: str, step_payload: Any) -> None:
        events.put(
            (
                "step",
                {
                    "step": step,
                    "label": STEP_LABELS.get(step, step),
                    "payload": compact_payload(step, step_payload),
                },
            )
        )

    def on_answer_delta(delta: str) -> None:
        events.put(("answer_delta", {"delta": delta}))

    def on_answer_event(event_name: str, payload: dict[str, Any]) -> None:
        events.put(("answer_event", {"event": event_name, **payload}))

    def worker() -> None:
        try:
            workflow = AgenticRagWorkflow(memory=memory_store)
            result = workflow.run(
                session_id,
                message,
                on_step=on_step,
                on_answer_delta=on_answer_delta,
                on_answer_event=on_answer_event,
                enable_coverage_check=enable_coverage_check,
                retrieval_mode=retrieval_mode,
            )
            events.put(("final", to_dict(result)))
        except Exception as exc:
            events.put(("error", {"error": str(exc)}))
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    @stream_with_context
    def generate():
        yield sse_event("status", {"message": "Received. Processing..."})
        while True:
            item = events.get()
            if item is None:
                break
            event_name, data = item
            yield sse_event(event_name, data)

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
