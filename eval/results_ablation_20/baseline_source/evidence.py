from __future__ import annotations

import math
import re

from .schemas import EvidenceItem, EvidencePack, RetrievedChunk


def compact_text(text: str, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", "\n", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_evidence(chunks: list[RetrievedChunk], final_k: int = 8) -> EvidencePack:
    grouped: dict[str, list[RetrievedChunk]] = {}

    for chunk in chunks:
        key = chunk.parent_id or chunk.chunk_id
        grouped.setdefault(key, []).append(chunk)

    parent_candidates: list[tuple[float, str, list[RetrievedChunk]]] = []
    for parent_id, parent_chunks in grouped.items():
        parent_chunks.sort(
            key=lambda item: item.rerank_score if item.rerank_score is not None else item.score,
            reverse=True,
        )
        best_score = parent_chunks[0].rerank_score
        if best_score is None:
            best_score = parent_chunks[0].score
        parent_score = float(best_score or 0.0) + 0.02 * math.log1p(len(parent_chunks))
        parent_candidates.append((parent_score, parent_id, parent_chunks))

    parent_candidates.sort(key=lambda item: item[0], reverse=True)

    items: list[EvidenceItem] = []
    for parent_score, parent_id, parent_chunks in parent_candidates[:final_k]:
        best = parent_chunks[0]
        content = compact_text(best.parent_text or best.text, max_chars=900)
        if not content:
            continue
        matched_children = [
            {
                "chunk_id": child.chunk_id,
                "qdrant_score": child.score,
                "rerank_score": child.rerank_score,
                "title_path": child.title_path,
                "text": compact_text(child.text, max_chars=260),
            }
            for child in parent_chunks[:3]
        ]
        title_path = best.title_path
        parts = title_path.split(" > ") if title_path else []
        if len(parts) > 4:
            title_path = " > ".join(parts[:4])
        items.append(
            EvidenceItem(
                evidence_id=f"ev_{len(items) + 1:03d}",
                content=content,
                source_name=best.source_name,
                title_path=title_path,
                chunk_id=best.chunk_id,
                parent_id=parent_id,
                score=parent_score,
                matched_children=matched_children,
            )
        )

    if not items:
        return EvidencePack(items=[], status="insufficient", issues=["未检索到可用证据"])
    status = "sufficient" if len(items) >= 2 else "partial"
    issues = [] if status == "sufficient" else ["证据数量较少，回答需要更谨慎"]
    return EvidencePack(items=items, status=status, issues=issues)
