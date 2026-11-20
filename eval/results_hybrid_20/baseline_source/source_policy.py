"""Shared intent mapping and score-preserving source quota selection."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable, TypeVar

from .config import ROOT_DIR

T = TypeVar("T")
SOURCE_TYPES = ("statute", "case")
TARGET_TYPES = {"statute_db": "statute", "case_db": "case"}


def resolve_intent(value: Any) -> str:
    """Case request > individual consultation > procedure > statute; default consultation."""
    if isinstance(value, str):
        labels = {value}
    else:
        labels = set(getattr(value, "intents", []) or [])
        labels.add(getattr(value, "query_type", ""))
    if labels & {"案例检索", "case_search"}:
        return "案例检索"
    if labels & {"法律咨询", "legal_consultation"}:
        return "法律咨询"
    if labels & {"程序咨询", "procedure_consultation"}:
        return "程序咨询"
    if labels & {"法条解释", "knowledge_question", "statute_explanation"}:
        return "法条解释"
    return "法律咨询"


def source_quotas(intent: Any, total: int, targets=None) -> dict[str, int]:
    if total < 0:
        raise ValueError("Quota total must be nonnegative")
    sources = {TARGET_TYPES[t] for t in (targets or TARGET_TYPES) if t in TARGET_TYPES}
    if len(sources) == 1:
        source = next(iter(sources))
        return {s: total if s == source else 0 for s in SOURCE_TYPES}
    policy = resolve_intent(intent)
    weights = (2, 3) if policy == "案例检索" else (3, 2) if policy == "法律咨询" else (4, 1)
    # Largest remainder; ties favor statute, keeping the total exactly unchanged.
    counts = [total * w // 5 for w in weights]
    remaining = total - sum(counts)
    order = sorted(range(2), key=lambda i: (-(total * weights[i] % 5), i))
    for i in order[:remaining]:
        counts[i] += 1
    return dict(zip(SOURCE_TYPES, counts))


@lru_cache(maxsize=1)
def local_source_index() -> dict[str, str]:
    index = {}
    for directory, kind in (("json", "statute"), ("json_cases", "case")):
        for path in sorted((ROOT_DIR / "knowledge_data" / directory).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            for key in [str(data["doc_id"])] + [str(p["parent_id"]) for p in data.get("parents", [])] + [str(c["chunk_id"]) for c in data.get("children", [])]:
                if key in index and index[key] != kind:
                    raise ValueError("Conflicting source metadata")
                index[key] = kind
    return index


def chunk_source(chunk, *, required: bool = True) -> str:
    payload = chunk.payload
    kind = payload.get("doc_type")
    if kind in SOURCE_TYPES:
        return kind
    target = payload.get("_retrieval_target")
    if target in TARGET_TYPES:
        return TARGET_TYPES[target]
    index = local_source_index()
    for key in (str(payload.get("doc_id") or ""), chunk.parent_id, chunk.chunk_id):
        if key in index:
            return index[key]
    if required:
        raise ValueError("Source type missing from payload, origin and local corpus mapping")
    return "unknown"


def select_by_source(items: list[T], total: int, quotas: dict[str, int] | None,
                     source: Callable[[T], str], score: Callable[[T], float]) -> list[T]:
    ordered = sorted(items, key=score, reverse=True)
    if quotas is None:
        return ordered[:total]
    if set(quotas) - set(SOURCE_TYPES) or any(type(v) is not int or v < 0 for v in quotas.values()) or sum(quotas.values()) != total:
        raise ValueError("Source quotas must be nonnegative integers summing to total")
    selected, remaining, counts = [], [], dict.fromkeys(SOURCE_TYPES, 0)
    for item in ordered:
        kind = source(item)
        if kind not in SOURCE_TYPES:
            raise ValueError("Unknown candidate source")
        if counts[kind] < quotas.get(kind, 0):
            selected.append(item)
            counts[kind] += 1
        else:
            remaining.append(item)
    selected.extend(remaining[:max(0, total - len(selected))])
    return sorted(selected, key=score, reverse=True)[:total]
