#!/usr/bin/env python3
"""Source-checked retrieval evaluation; legacy CLI/summary keys remain available.

No answer-generation or query-understanding LLM is called. Production collection
mapping, parent enrichment, reranking and evidence construction are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.config import AppConfig
from rag.evidence import build_evidence
from rag.providers import Providers
from rag.retrieval import ParentChunkStore, Retriever
from rag.schemas import to_dict

STAGES = {"recall": [1, 5, 10, 20, 30, 60], "rerank": [1, 3, 5, 10, 20], "evidence": [1, 3, 5]}
RECALL_K, RERANK_K, EVIDENCE_K = STAGES.values()


@dataclass
class CorpusIndex:
    chunk_to_doc: dict
    parent_to_doc: dict
    chunk_to_parent: dict
    doc_meta: dict
    parents: dict
    children: dict

    def __iter__(self):
        # Preserve the original three-value unpacking API.
        yield self.chunk_to_doc
        yield self.parent_to_doc
        yield set(self.doc_meta)


def load_queries(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            item = json.loads(line)
            if not str(item.get("query", "")).strip() or not item.get("relevant_doc_ids"):
                raise ValueError(f"Invalid query record at line {number}")
            for key in ("relevant_doc_ids", "relevant_parent_ids", "relevant_statute_doc_ids", "relevant_case_doc_ids"):
                if key in item and (not isinstance(item[key], list) or any(not isinstance(v, str) for v in item[key])):
                    raise ValueError(f"IDs must be string lists: line {number}, {key}")
            item.setdefault("query_id", f"q{len(rows)+1:03d}")
            item.setdefault("split", "unspecified")
            item.setdefault("query_type", "unspecified")
            rows.append(item)
    if len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate query_id")
    return rows


def build_indexes(json_dir: Path, case_json_dir: Path | None = None) -> CorpusIndex:
    case_json_dir = case_json_dir if case_json_dir is not None else ROOT_DIR / 'knowledge_data/json_cases'
    cd, pd, cp, meta, parents, children = {}, {}, {}, {}, {}, {}
    for directory, kind in ((json_dir, 'statute'), (case_json_dir, 'case')):
        for path in sorted(directory.glob('*.json')):
            d = json.loads(path.read_text(encoding='utf-8-sig'))
            did = str(d['doc_id'])
            if did in meta:
                raise ValueError(f'Duplicate doc_id: {did}')
            meta[did] = {'doc_id': did, 'doc_type': kind, 'title': d['title'], 'json_path': str(path),
                         'source_path': d.get('source_path', '')}
            for parent in d['parents']:
                pid = parent['parent_id']
                if pid in pd: raise ValueError(f'Duplicate parent_id: {pid}')
                pd[pid], parents[pid] = did, parent
            for child in d['children']:
                cid, pid = child['chunk_id'], child['parent_id']
                if cid in cd or pid not in pd or pd[pid] != did:
                    raise ValueError(f'Invalid/duplicate child: {cid}')
                if cid not in parents[pid]['child_ids']: raise ValueError(f'Broken child link: {cid}')
                cd[cid], cp[cid], children[cid] = did, pid, child
    if not meta: raise ValueError('Empty corpus')
    for pid, parent in parents.items():
        if any(cp.get(cid) != pid for cid in parent['child_ids']): raise ValueError(f'Broken parent link: {pid}')
    return CorpusIndex(cd, pd, cp, meta, parents, children)


def validate_labels(rows, index):
    for r in rows:
        for did in r['relevant_doc_ids']:
            if did not in index.doc_meta: raise ValueError(f"Unknown doc_id {did} in {r['query_id']}")
        for pid in r.get('relevant_parent_ids', []):
            if pid not in index.parent_to_doc or index.parent_to_doc[pid] not in r['relevant_doc_ids']:
                raise ValueError(f"Unknown/inconsistent parent_id {pid}")
        for kind in ('statute', 'case'):
            key = f'relevant_{kind}_doc_ids'
            inferred = [d for d in r['relevant_doc_ids'] if index.doc_meta[d]['doc_type'] == kind]
            if key in r and set(r[key]) != set(inferred): raise ValueError(f'Inconsistent {key}')
            r[key] = inferred
        for check in r.get('source_checks', []):
            p = index.parents[check['parent_id']]
            norm = lambda s: ''.join(s.split())
            if norm(check['quote']) not in norm(p['text']): raise ValueError('Source quotation changed')
            expected = check.get('parent_text_sha256')
            if expected and hashlib.sha256(p['text'].encode()).hexdigest() != expected:
                raise ValueError('Source parent changed after annotation')


def to_doc_rank(ids, id_to_doc):
    result = []
    for identifier in ids:
        if identifier not in id_to_doc: raise ValueError(f'Unmapped retrieved ID: {identifier}')
        doc = id_to_doc[identifier]
        if doc not in result: result.append(doc)
    return result


def unique(ids):
    return list(dict.fromkeys(ids))


def compute_metrics(doc_rank, relevant, k_values):
    relevant = set(relevant)
    if not relevant: return {}  # Empty scopes are excluded, rather than scored as zero.
    rank = unique(doc_rank)
    result = {'mrr': next((1/i for i, d in enumerate(rank, 1) if d in relevant), 0.0)}
    for k in k_values:
        hits = len(set(rank[:k]) & relevant)
        result.update({f'recall@{k}': hits/len(relevant), f'precision@{k}': hits/k,
                       f'hit@{k}': float(hits > 0)})
    result['recall@pool'] = len(set(rank) & relevant)/len(relevant)
    result['hit@pool'] = float(bool(set(rank) & relevant))
    return result


def error_label(exc):
    # Header-validation exceptions can contain an API key. Never stringify them.
    chain = exc
    while chain:
        if isinstance(chain, urllib.error.HTTPError): return f'HTTP_{chain.code}'
        chain = chain.__cause__
    return type(exc).__name__


class ApiMonitor:
    def __init__(self, retries=2, delay=0.25):
        self.events = []
        self.retries, self.delay = retries, delay
        self.context = {}

    def call(self, phase, action):
        for attempt in range(1, self.retries+1):
            started = time.perf_counter()
            event = {**self.context, 'phase': phase, 'attempt': attempt}
            try:
                value = action()
                event.update(ok=True, seconds=time.perf_counter()-started)
                self.events.append(event)
                return value
            except Exception as exc:
                event.update(ok=False, error=error_label(exc), seconds=time.perf_counter()-started)
                self.events.append(event)
                if attempt == self.retries:
                    raise RuntimeError(f'{phase} failed: {error_label(exc)}') from None
                time.sleep(min(2**(attempt-1), 4))


class AuditedReranker:
    def __init__(self, provider, monitor):
        self.provider, self.monitor = provider, monitor

    def rerank(self, query, documents, top_n):
        def request():
            rows = self.provider.rerank(query, documents, top_n)
            if len(rows) != min(top_n, len(documents)):
                raise ValueError('Incomplete rerank response')
            ids = [r.get('index') for r in rows]
            if len(set(ids)) != len(ids) or any(not isinstance(i, int) or not 0<=i<len(documents) for i in ids):
                raise ValueError('Invalid rerank indices')
            for r in rows:
                score = r.get('relevance_score', r.get('score'))
                if score is None or not math.isfinite(float(score)):
                    raise ValueError('Missing/nonfinite rerank score')
            return rows
        return self.monitor.call('rerank', request)


def make_retriever(config, index, monitor, json_dir, case_json_dir):
    p = Providers(config)
    if any(getattr(getattr(p, n), 'offline', False) for n in ('embedding','qdrant','rerank')):
        raise ValueError('Benchmark requires real embedding/Qdrant/rerank providers; Mock prohibited')
    r = Retriever(config, p.embedding, p.qdrant, AuditedReranker(p.rerank, monitor))
    r.parents = ParentChunkStore(config, json_dir, case_json_dir)
    r.parents._load_local_index()  # Exclude one-time JSON loading from steady-state stage timings.
    return r


def choose_targets(raw, config):
    targets = [s.strip() for s in raw.split(',') if s.strip()] if raw else ['statute_db'] + (
        ['case_db'] if config.section('qdrant').get('cases_collection_name') else [])
    if not targets or len(set(targets)) != len(targets) or set(targets)-{'statute_db','case_db'}:
        raise ValueError('Invalid targets')
    return targets


def metrics_for(item, doc_ranks, parent_ranks, index):
    result = {}
    for scope, docs in [('all_gt', item['relevant_doc_ids']),
                        ('statute_gt', item['relevant_statute_doc_ids']),
                        ('case_gt', item['relevant_case_doc_ids'])]:
        parents = [p for p in item.get('relevant_parent_ids', []) if index.parent_to_doc[p] in docs]
        # Retrieved lists are NOT filtered by source: cases consume statute positions and vice versa.
        result[scope] = {
            'doc': {s:compute_metrics(doc_ranks[s],docs,ks) for s,ks in STAGES.items()},
            'parent': {s:compute_metrics(parent_ranks[s],parents,ks) for s,ks in STAGES.items()}}
    return result


def evaluate_one(item, retriever, top_k, final_k, index, targets, vector, monitor):
    started = time.perf_counter()
    raw, search_counts = [], {}
    for target in targets:
        collection = retriever._collection_for(target)
        if not collection: raise ValueError(f'Missing collection for {target}')
        hits = monitor.call('qdrant_search', lambda: retriever.qdrant.search(
            vector=vector, limit=top_k, collection=collection))
        search_counts[target] = len(hits)
        raw.extend(hits)
    search_seconds = time.perf_counter()-started
    chunks = [retriever._from_qdrant_hit(hit) for hit in raw]
    if len({c.chunk_id for c in chunks}) != len(chunks): raise ValueError('Duplicate recalled chunks across targets')
    for c in chunks:
        if c.chunk_id not in index.chunk_to_doc or index.chunk_to_parent[c.chunk_id] != c.parent_id:
            raise ValueError('Remote chunk/parent not present in local corpus')
        if c.text != index.children[c.chunk_id]['text']: raise ValueError('Remote/local child content differs')
    parent_started = time.perf_counter()
    pids = sorted({c.parent_id for c in chunks if c.parent_id})
    parent_map = retriever.parents.get_many(pids)
    for c in chunks:
        parent = parent_map.get(c.parent_id) or {}
        c.parent_text = str(parent.get('text') or '')
        c.title_path = c.title_path or str(parent.get('title_path') or '')
        if parent.get('document_title'):
            c.source_name = str(parent['document_title']).split('>')[0].strip() or c.source_name
        if not c.parent_text or c.parent_text != index.parents[c.parent_id]['text']:
            raise ValueError('Parent enrichment absent or differs from frozen local corpus')
    parent_seconds = time.perf_counter()-parent_started
    before = [to_dict(c) for c in chunks]
    rank_started = time.perf_counter()
    ranked = retriever.rerank(item['query'], chunks, final_k)
    fallback = int(bool(chunks) and (not ranked or any(c.rerank_score is None for c in ranked)))
    if fallback:
        monitor.events.append({**monitor.context,'phase':'rerank_fallback','ok':False})
        raise RuntimeError('Rerank fallback detected; abort rather than silently score degraded results')
    rerank_seconds = time.perf_counter()-rank_started
    ev_started = time.perf_counter()
    evidence = build_evidence(ranked, final_k=final_k)
    evidence_seconds = time.perf_counter()-ev_started
    doc_ranks = {'recall':to_doc_rank([c.chunk_id for c in chunks],index.chunk_to_doc),
                 'rerank':to_doc_rank([c.chunk_id for c in ranked],index.chunk_to_doc),
                 'evidence':to_doc_rank([e.parent_id for e in evidence.items],index.parent_to_doc)}
    parent_ranks = {'recall':unique([c.parent_id for c in chunks]),
                    'rerank':unique([c.parent_id for c in ranked]),
                    'evidence':[e.parent_id for e in evidence.items]}
    scoped = metrics_for(item, doc_ranks, parent_ranks, index)
    def brief(c):
        return {'chunk_id':c.chunk_id,'parent_id':c.parent_id,'doc_id':index.chunk_to_doc[c.chunk_id],
                'doc_type':index.doc_meta[index.chunk_to_doc[c.chunk_id]]['doc_type'],
                'score':c.score,'rerank_score':c.rerank_score,'title_path':c.title_path,'text':c.text}
    composition = {kind:sum(index.doc_meta[index.parent_to_doc[e.parent_id]]['doc_type']==kind for e in evidence.items)
                   for kind in ('statute','case')}
    return {**item,'doc_ranks':doc_ranks,'parent_ranks':parent_ranks,
            'metrics':scoped['all_gt']['doc'], 'scoped_metrics':scoped,
            'recall_chunks':[{k:v for k,v in c.items() if k not in ('parent_text','payload')} for c in before],
            'rerank_chunks':[brief(c) for c in ranked], 'evidence_pack':to_dict(evidence),
            'parent_contexts':{pid:parent_map[pid]['text'] for pid in pids},
            'search_counts':search_counts,'rerank_fallback_count':fallback,'evidence_counts':composition,
            'timings':{'search_seconds':search_seconds,'parent_seconds':parent_seconds,
                       'rerank_seconds':rerank_seconds,'evidence_seconds':evidence_seconds,
                       'retrieval_seconds':time.perf_counter()-started}}


def aggregate(results):
    # Legacy flat all-GT doc-level summary.
    values = {}
    for r in results:
        for stage, metrics in r['metrics'].items():
            for name, value in metrics.items(): values.setdefault(f'{stage}.{name}',[]).append(value)
    return {key:sum(v)/len(v) for key,v in values.items()}


def summarize(results):
    scoped = {}
    for scope in ('statute_gt','all_gt','case_gt'):
        scoped[scope] = {}
        for grain in ('doc','parent'):
            scoped[scope][grain] = {}
            for stage in STAGES:
                rows = [r['scoped_metrics'][scope][grain][stage] for r in results
                        if r['scoped_metrics'][scope][grain][stage]]
                scoped[scope][grain][stage] = {'n':len(rows), **({k:sum(row[k] for row in rows)/len(rows)
                    for k in rows[0]} if rows else {})}
    n = len(results)
    return {'n_queries':n, 'scoped_metrics':scoped,
            'avg_statute_evidence_count':sum(r['evidence_counts']['statute'] for r in results)/n if n else 0,
            'avg_case_evidence_count':sum(r['evidence_counts']['case'] for r in results)/n if n else 0,
            'mean_retrieval_seconds':sum(r['timings']['retrieval_seconds'] for r in results)/n if n else 0,
            'rerank_fallback_count':sum(r['rerank_fallback_count'] for r in results)}


def grouped(results):
    out = {}
    for split in ['all']+sorted({r['split'] for r in results}):
        subset = results if split=='all' else [r for r in results if r['split']==split]
        out[split] = {'all':summarize(subset)}
        for kind in sorted({r['query_type'] for r in subset}):
            out[split][kind] = summarize([r for r in subset if r['query_type']==kind])
        out[split]['with_case_gt'] = summarize([r for r in subset if r['relevant_case_doc_ids']])
        out[split]['statute_only_gt'] = summarize([r for r in subset if not r['relevant_case_doc_ids']])
    return out


def save_result(path, details, metadata, events):
    content = {'metadata':metadata,'summary':aggregate(details),'groups':grouped(details),'details':details,
               'api_events':events,'rerank_fallback_count':sum(r['rerank_fallback_count'] for r in details)}
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    return content


def write_md_report(summary, n_queries, path):
    lines=['# 检索评测汇总','',f'样本数：{n_queries}；旧版兼容表使用 all_gt 文档级口径。','',
           '| 阶段.指标 | 值 |','|---|---:|']
    lines += [f'| {key} | {value:.4f} |' for key,value in summary.items()]
    lines += ['', '## 代码与存放路径', '',
              '本报告由修改后的 `eval/eval_retrieval.py` 生成，入口为 `main()`；指标计算、分组和报告输出均在该文件中。',
              f'完整路径：`{ROOT_DIR / "eval/eval_retrieval.py"}`。',
              '源代码及JSON结果放在相应代码目录；实验Markdown报告统一放在 `Refinement and Optimization`。']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries',type=Path,default=ROOT_DIR/'eval/queries.jsonl')
    parser.add_argument('--json-dir',type=Path,default=ROOT_DIR/'knowledge_data/json')
    parser.add_argument('--case-json-dir',type=Path,default=ROOT_DIR/'knowledge_data/json_cases')
    parser.add_argument('--targets',default=None)
    parser.add_argument('--output',type=Path,default=ROOT_DIR/'eval/report.json')
    parser.add_argument('--report-md',type=Path,default=ROOT_DIR/'Refinement and Optimization/report_retrieval.md')
    parser.add_argument('--top-k',type=int,default=30);parser.add_argument('--final-k',type=int,default=5)
    args=parser.parse_args()
    if args.top_k<1 or args.final_k<1:raise ValueError('k must be positive')
    index=build_indexes(args.json_dir,args.case_json_dir);rows=load_queries(args.queries);validate_labels(rows,index)
    config=AppConfig.load();monitor=ApiMonitor();r=make_retriever(config,index,monitor,args.json_dir,args.case_json_dir)
    targets=choose_targets(args.targets,config);details=[]
    for item in rows:
        monitor.context={'query_id':item['query_id']}
        vector=monitor.call('embedding',lambda:r.embedding.embed(item['query']))
        if len(vector)!=1024 or not all(math.isfinite(v) for v in vector):raise ValueError('Invalid embedding')
        details.append(evaluate_one(item,r,args.top_k,args.final_k,index,targets,vector,monitor))
        print(f"[{len(details)}/{len(rows)}] {item['query_id']}",flush=True)
        time.sleep(monitor.delay)
    result=save_result(args.output,details,{'targets':targets,'top_k':args.top_k,'final_k':args.final_k},monitor.events)
    write_md_report(result['summary'],len(details),args.report_md)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print('Evaluation failed:',error_label(exc),flush=True)
        raise SystemExit(1)
