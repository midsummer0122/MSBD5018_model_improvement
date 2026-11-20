from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.workflow import AgenticRagWorkflow


def print_debug(debug: dict) -> None:
    print("\n--- 中间结果 ---")
    for key in [
        "query_understanding",
        "followup_result",
        "retrieval_plan",
        "retrieved_chunks",
        "evidence_pack",
        "final_prompt",
        "rewrite_prompt",
        "safety_review",
        "safety_review_after_rewrite",
    ]:
        if key not in debug:
            continue
        value = debug[key]
        if key == "retrieved_chunks":
            preview = []
            for item in value[:5]:
                preview.append(
                    {
                        "chunk_id": item.get("chunk_id"),
                        "parent_id": item.get("parent_id"),
                        "title_path": item.get("title_path"),
                        "score": item.get("score"),
                        "rerank_score": item.get("rerank_score"),
                    }
                )
            value = preview
        elif key == "evidence_pack":
            preview_items = []
            for item in value.get("items", []):
                preview_items.append(
                    {
                        "evidence_id": item.get("evidence_id"),
                        "parent_id": item.get("parent_id"),
                        "parent_rank_score": item.get("score"),
                        "title_path": item.get("title_path"),
                        "matched_children": [
                            {
                                "chunk_id": child.get("chunk_id"),
                                "qdrant_score": child.get("qdrant_score"),
                                "rerank_score": child.get("rerank_score"),
                                "title_path": child.get("title_path"),
                                "text": child.get("text"),
                            }
                            for child in item.get("matched_children", [])
                        ],
                        "parent_context_preview": item.get("content", "")[:500],
                    }
                )
            value = {
                "status": value.get("status"),
                "issues": value.get("issues"),
                "items": preview_items,
            }
        elif key in {"final_prompt", "rewrite_prompt"}:
            print(f"\n[{key}]")
            print(value)
            continue
        print(f"\n[{key}]")
        print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> int:
    workflow = AgenticRagWorkflow(show_prompt=True)
    session_id: str | None = None
    if getattr(workflow.providers, "offline", False):
        print("[提示] 离线演示模式：未检测到 LLM/检索服务密钥，将跳过真实模型与知识库检索，返回占位回答。")
    print("Agentic RAG 法律问答调试模式")
    print("输入问题后回车；输入 /new 新会话；输入 /debug 切换中间结果显示；输入 /prompt 切换总提示词实时打印；输入 /exit 退出。")
    show_debug = True
    show_prompt = True

    while True:
        query = input("\n你：").strip()
        if not query:
            continue
        if query == "/exit":
            break
        if query == "/new":
            session_id = None
            print("已切换到新会话。")
            continue
        if query == "/debug":
            show_debug = not show_debug
            print(f"中间结果显示：{'开启' if show_debug else '关闭'}")
            continue
        if query == "/prompt":
            show_prompt = not show_prompt
            workflow.show_prompt = show_prompt
            print(f"总提示词实时打印：{'开启' if show_prompt else '关闭'}")
            continue

        try:
            result = workflow.run(session_id, query)
        except Exception as exc:
            print(f"\n[错误] {exc}")
            continue

        session_id = result.session_id
        print(f"\n助手（session={session_id}, type={result.answer_type}）：")
        print(result.answer)
        if result.citations:
            print("\n参考来源：")
            for idx, citation in enumerate(result.citations, start=1):
                print(f"{idx}. {citation.get('source_name')} / {citation.get('title_path')}")
        if show_debug:
            print_debug(result.debug)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
