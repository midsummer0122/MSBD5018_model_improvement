from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import AppConfig, ROOT_DIR
from .providers import EmbeddingProvider, QdrantClient, RerankProvider
from .schemas import RetrievedChunk, RetrievalPlan
from .source_policy import chunk_source, select_by_source


JSON_DIR = ROOT_DIR / "knowledge_data" / "json"


class ParentChunkStore:
    def __init__(self, config: AppConfig, json_dir: Path = JSON_DIR,
                 case_json_dir: Path | None = None) -> None:
        self.config = config
        self.json_dir = json_dir
        self.case_json_dir = case_json_dir or json_dir.with_name("json_cases")
        self._local_index: dict[str, dict[str, Any]] | None = None

    def get_many(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        result = self._get_many_mysql(parent_ids)
        missing = [pid for pid in parent_ids if pid not in result]
        if missing:
            result.update(self._get_many_local(missing))
        return result

    def _get_many_mysql(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not parent_ids:
            return {}
        try:
            import pymysql
        except ImportError:
            return {}

        section = self.config.section("mysql")
        try:
            conn = pymysql.connect(
                host=str(section.get("host") or "127.0.0.1"),
                port=int(section.get("port") or 3306),
                user=str(section.get("user") or ""),
                password=str(section.get("password") or ""),
                database=str(section.get("database") or ""),
                charset=str(section.get("charset") or "utf8mb4"),
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=5,
            )
        except Exception:
            return {}

        placeholders = ",".join(["%s"] * len(parent_ids))
        sql = f"""
        SELECT pc.parent_id, pc.title_path, pc.text, d.title AS document_title, d.source_path
        FROM parent_chunks pc
        LEFT JOIN documents d ON d.id = pc.document_id
        WHERE pc.parent_id IN ({placeholders})
        """
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, parent_ids)
                rows = cursor.fetchall()
            return {str(row["parent_id"]): dict(row) for row in rows}
        except Exception:
            return {}
        finally:
            conn.close()

    def _load_local_index(self) -> dict[str, dict[str, Any]]:
        if self._local_index is not None:
            return self._local_index
        index: dict[str, dict[str, Any]] = {}
        paths = sorted({*self.json_dir.glob("*.json"), *self.case_json_dir.glob("*.json")})
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            for parent in data.get("parents", []):
                parent_id = str(parent.get("parent_id") or "")
                if parent_id:
                    if parent_id in index:
                        raise ValueError(f"Duplicate local parent_id: {parent_id}")
                    index[parent_id] = {
                        "parent_id": parent_id,
                        "title_path": parent.get("title_path") or data.get("title") or "",
                        "text": parent.get("text") or "",
                        "document_title": data.get("title") or "",
                        "source_path": data.get("source_path") or "",
                    }
        self._local_index = index
        return index

    def _get_many_local(self, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
        index = self._load_local_index()
        return {pid: index[pid] for pid in parent_ids if pid in index}


class Retriever:
    def __init__(
        self,
        config: AppConfig,
        embedding: EmbeddingProvider,
        qdrant: QdrantClient,
        rerank: RerankProvider,
    ) -> None:
        self.config = config
        self.embedding = embedding
        self.qdrant = qdrant
        self.rerank_provider = rerank
        self.parents = ParentChunkStore(config)

    def _collection_for(self, target: str) -> str:
        """把检索目标映射到 Qdrant 集合名；未配置的集合返回空串（检索时自动跳过）。"""
        section = self.config.section("qdrant")
        if target == "case_db":
            return str(section.get("cases_collection_name") or "")
        return str(section.get("collection_name") or "")

    def retrieve(self, plan: RetrievalPlan) -> list[RetrievedChunk]:
        vector = self.embedding.embed(plan.query)
        targets = plan.retrieval_targets or ["statute_db"]
        raw_hits: list[dict[str, Any]] = []
        for target in targets:
            collection = self._collection_for(target)
            if not collection:
                continue
            for hit in self.qdrant.search(vector=vector, limit=plan.top_k, collection=collection):
                hit = dict(hit)
                hit["payload"] = dict(hit.get("payload") or {})
                hit["payload"]["_retrieval_target"] = target
                hit["payload"]["_source_collection"] = collection
                raw_hits.append(hit)
        chunks = [self._from_qdrant_hit(hit) for hit in raw_hits]

        parent_ids = sorted({chunk.parent_id for chunk in chunks if chunk.parent_id})
        parent_map = self.parents.get_many(parent_ids)
        for chunk in chunks:
            parent = parent_map.get(chunk.parent_id) or {}
            chunk.parent_text = str(parent.get("text") or "")
            chunk.title_path = chunk.title_path or str(parent.get("title_path") or "")
            if parent.get("document_title"):
                chunk.source_name = str(parent["document_title"]).split(">")[0].strip() or chunk.source_name

        return self.rerank(plan.query, chunks, plan.final_k,
                           split_rerank=plan.split_rerank, rerank_quotas=plan.rerank_quotas)

    def _from_qdrant_hit(self, hit: dict[str, Any]) -> RetrievedChunk:
        payload = hit.get("payload") or {}
        return RetrievedChunk(
            chunk_id=str(payload.get("chunk_id") or hit.get("id") or ""),
            parent_id=str(payload.get("parent_id") or ""),
            text=str(payload.get("text") or payload.get("content") or ""),
            title_path=str(payload.get("title_path") or ""),
            source_name=str(payload.get("source_name") or "法律法规"),
            score=float(hit.get("score") or 0.0),
            payload=dict(payload),
        )

    def rerank(self, query: str, chunks: list[RetrievedChunk], final_k: int, *,
               split_rerank: bool = False, rerank_quotas: dict[str, int] | None = None) -> list[RetrievedChunk]:
        if not chunks:
            return []
        documents = [
            f"标题：{chunk.title_path}\n子块内容：{chunk.text}"
            for chunk in chunks
        ]
        rerank_k = min(max(final_k * 4, final_k), len(documents))
        mixed = split_rerank and len({chunk_source(c, required=False) for c in chunks}) > 1
        top_n = len(documents) if mixed else rerank_k
        try:
            ranked = self.rerank_provider.rerank(query, documents, top_n=top_n)
            indices = [item.get("index") for item in ranked]
            if len(ranked) != top_n or len(set(indices)) != top_n:
                raise ValueError("Incomplete or duplicate rerank response")
            if any(type(i) is not int or not 0 <= i < len(chunks) for i in indices):
                raise ValueError("Invalid rerank index")
            if any(not math.isfinite(float(item.get("relevance_score", item.get("score")))) for item in ranked):
                raise ValueError("Invalid rerank score")
        except Exception:
            return sorted(chunks, key=lambda item: item.score, reverse=True)[:rerank_k]

        result: list[RetrievedChunk] = []
        for item in ranked:
            index = int(item.get("index", 0))
            if 0 <= index < len(chunks):
                chunk = chunks[index]
                chunk.rerank_score = float(item.get("relevance_score") or item.get("score") or 0.0)
                result.append(chunk)
        # Scale configured quotas when fewer than final_k*4 candidates exist.
        quotas = rerank_quotas if split_rerank else None
        if quotas is not None and sum(quotas.values()) != rerank_k:
            # Preserve the configured ratio even for nonstandard plans.
            total = sum(quotas.values())
            statute = rerank_k * quotas.get("statute", 0) / max(total, 1)
            n_statute = int(statute + 0.5)
            quotas = {"statute": n_statute, "case": rerank_k - n_statute}
        return select_by_source(result, rerank_k, quotas, chunk_source,
                                lambda item: float(item.rerank_score or 0.0))
