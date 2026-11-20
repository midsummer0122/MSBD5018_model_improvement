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
        self.last_rerank_fallback = False
        self.last_initial_trace: dict[str, Any] = {}

    def _collection_for(self, target: str) -> str:
        """把检索目标映射到 Qdrant 集合名；未配置的集合返回空串（检索时自动跳过）。"""
        section = self.config.section("qdrant")
        if target == "case_db":
            return str(section.get("cases_collection_name") or "")
        return str(section.get("collection_name") or "")

    def initial_candidates(self, plan: RetrievalPlan, *, vector=None, dense_hits=None, on_progress=None) -> list[RetrievedChunk]:
        """Same initial-candidate path online and in cached, read-only experiments."""
        from .lexical import get_lexical_index, fuse_candidates
        if plan.retrieval_mode not in ("dense", "dense_bm25", "hybrid"):
            raise ValueError("Unknown retrieval mode")
        if vector is None and dense_hits is None:
            vector = self.embedding.embed(plan.query)
        targets = plan.retrieval_targets or ["statute_db"]
        chunks: list[RetrievedChunk] = []
        index, cache_info = (get_lexical_index() if plan.retrieval_mode != "dense" else (None, {}))
        self.last_initial_trace = {"retrieval_mode": plan.retrieval_mode,
            "lexical_index": {k: v for k, v in cache_info.items() if k != "files"}, "targets": {}}
        for target in targets:
            collection = self._collection_for(target)
            if not collection:
                if on_progress:on_progress({'stage':'library','status':'skipped','target':target})
                continue
            def progress(payload):
                if on_progress:on_progress({**payload,'target':target})
            progress({'stage':'dense','status':'started'})
            hits = (dense_hits[target] if dense_hits is not None else
                    self.qdrant.search(vector=vector, limit=plan.top_k, collection=collection))
            progress({'stage':'dense','status':'completed','count':len(hits)})
            dense = []
            for rank, hit in enumerate(hits, 1):
                hit = dict(hit)
                hit["payload"] = dict(hit.get("payload") or {})
                hit["payload"]["_retrieval_target"] = target
                hit["payload"]["_source_collection"] = collection
                chunk = self._from_qdrant_hit(hit)
                chunk.dense_score = chunk.score
                chunk.dense_rank = rank
                dense.append(chunk)
            kind = "case" if target == "case_db" else "statute"
            target_trace = {'dense_count':len(dense)}
            def lexical_progress(payload):
                if payload['status']=='completed':
                    if payload['stage']=='bm25':target_trace['bm25_count']=payload['count']
                    elif payload['stage']=='exact':
                        target_trace['exact_high_count']=payload['high_count']
                        target_trace['exact_weak_count']=payload['weak_count']
                    elif payload['stage']=='fusion':target_trace['exact_reserved_count']=payload['reserved_count']
                progress(payload)
            fused = fuse_candidates(dense, index, plan.query, kind, plan.retrieval_mode,
                                    plan.top_k, raw_query=plan.raw_query,on_progress=lexical_progress)
            for chunk in fused:
                chunk.payload["_retrieval_target"] = target
                chunk.payload["_source_collection"] = collection
            self.last_initial_trace["targets"][target] = {**target_trace,"fused_count":len(fused)}
            chunks.extend(fused)
        return chunks

    def enrich_parents(self, chunks: list[RetrievedChunk]) -> dict[str, dict[str, Any]]:
        parent_ids = sorted({chunk.parent_id for chunk in chunks if chunk.parent_id})
        parent_map = self.parents.get_many(parent_ids)
        for chunk in chunks:
            parent = parent_map.get(chunk.parent_id) or {}
            chunk.parent_text = str(parent.get("text") or "")
            chunk.title_path = chunk.title_path or str(parent.get("title_path") or "")
            if parent.get("document_title"):
                chunk.source_name = str(parent["document_title"]).split(">")[0].strip() or chunk.source_name
        return parent_map

    def retrieve(self, plan: RetrievalPlan, *, on_progress=None) -> list[RetrievedChunk]:
        def progress(stage,status,**values):
            if on_progress:on_progress({'stage':stage,'status':status,**values})
        chunks = self.initial_candidates(plan,on_progress=on_progress)
        expected=len({c.parent_id for c in chunks if c.parent_id})
        progress('parents','started',expected_count=expected)
        parent_map=self.enrich_parents(chunks)
        progress('parents','completed',count=len(parent_map),expected_count=expected)
        progress('rerank','started',input_count=len(chunks))
        ranked = self.rerank(plan.query, chunks, plan.final_k,
                           split_rerank=plan.split_rerank, rerank_quotas=plan.rerank_quotas,
                           retrieval_mode=plan.retrieval_mode)
        progress('rerank','fallback' if self.last_rerank_fallback else 'completed',count=len(ranked),
                 statute_count=sum(chunk_source(c,required=False)=='statute' for c in ranked),
                 case_count=sum(chunk_source(c,required=False)=='case' for c in ranked))
        return ranked

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
               split_rerank: bool = False, rerank_quotas: dict[str, int] | None = None,
               retrieval_mode: str = "dense") -> list[RetrievedChunk]:
        self.last_rerank_fallback = False
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
            self.last_rerank_fallback = True
            for chunk in chunks:
                chunk.rerank_score = None
                chunk.payload["_rerank_fallback"] = True
            if retrieval_mode == "dense":
                return sorted(chunks, key=lambda item: item.dense_score if item.dense_score is not None else item.score,
                              reverse=True)[:rerank_k]
            return chunks[:rerank_k]  # Preserve the unified fusion order; never mix score scales.

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
