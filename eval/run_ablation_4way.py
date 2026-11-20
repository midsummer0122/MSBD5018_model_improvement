"""20-query real-service ablation, atomic per-call cache and deterministic replay.

Only Qdrant read APIs are used. Secrets are never serialized. Run from project:
  .venv\Scripts\python.exe -X utf8 eval\run_ablation_4way.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from eval_retrieval import (ROOT_DIR, ApiMonitor, AuditedReranker, AppConfig, build_indexes,
    load_queries, validate_labels, to_doc_rank, unique, metrics_for, summarize, grouped, error_label)
from rag.providers import Providers
from rag.retrieval import Retriever
from rag.schemas import RetrievedChunk, ConversationState, to_dict
from rag.evidence import build_evidence
from rag.query_understanding import understand_query, _extract_json
from rag.skill_router import route_skills
from rag.source_policy import source_quotas, resolve_intent, chunk_source

OUT = ROOT_DIR / 'eval/results_ablation_20'
QUERY_PATH = ROOT_DIR / 'eval/queries_ablation_20.jsonl'


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def freeze_queries(index):
    original=load_queries(ROOT_DIR/'eval/queries_windtunnel_60.jsonl')
    validate_labels(original,index)
    selected=[]
    allocation={'法律咨询':8,'法条解释':5,'程序咨询':4,'案例检索':3}
    for intent,n in allocation.items():
        pool=[r for r in original if r['query_type']==intent]
        # Fixed evenly-spaced positions in original order; no ranking outcome input.
        selected.extend(copy.deepcopy(pool[round(i*(len(pool)-1)/(n-1))]) for i in range(n))
    expected=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
    if QUERY_PATH.exists() and QUERY_PATH.read_text(encoding='utf-8-sig')!=expected:
        raise ValueError('Frozen subset differs; refusing to overwrite')
    QUERY_PATH.write_text(expected,encoding='utf-8')
    validate_labels(selected,index)
    manifest={'algorithm':'evenly spaced round(i*(N-1)/(n-1)), original per-intent order',
        'allocation':allocation,'query_ids':[r['query_id'] for r in selected],
        'queries_sha256':hashlib.sha256(QUERY_PATH.read_bytes()).hexdigest(),
        'oracle_intent':True,'with_case_gt':sum(bool(r['relevant_case_doc_ids']) for r in selected),
        'pure_statute':sum(not r['relevant_case_doc_ids'] for r in selected),
        'source_quote_checks':sum(len(r.get('source_checks',[])) for r in selected)}
    write_json(OUT/'subset_manifest.json',manifest)
    return selected


class CachedCalls:
    def __init__(self,monitor,cache_only=False):
        self.monitor=monitor; self.hits=0; self.cache_only=cache_only

    def get(self,key,signature,phase,action):
        path=OUT/'cache'/f'{key}.json'; fingerprint=digest(signature)
        if path.exists():
            record=json.loads(path.read_text(encoding='utf-8'))
            if record['signature']!=fingerprint:
                raise ValueError('Stale cache signature; use a distinct output directory')
            self.hits+=1
            return record['value'],record['seconds']
        if self.cache_only:
            raise ValueError('Replay requires a complete real-service cache')
        start=time.perf_counter()
        try:
            value=self.monitor.call(phase,action)
        finally:
            write_json(OUT/'api_events.json',self.monitor.events)
        seconds=time.perf_counter()-start
        write_json(path,{'signature':fingerprint,'value':value,'seconds':seconds})
        return value,seconds


class RecordedScores:
    """Replay only actual API responses from this candidate list; never invent scores."""
    def __init__(self,rows): self.rows=rows
    def rerank(self,query,documents,top_n):
        return copy.deepcopy(sorted(self.rows,key=lambda r:r['relevance_score'],reverse=True)[:top_n])


def snapshot(providers,config):
    q=providers.qdrant; result={}
    for name in (config.section('qdrant')['collection_name'],config.section('qdrant')['cases_collection_name']):
        raw=q.http.request('GET',f'{q.url}/collections/{q.quote(name)}',{'api-key':q.api_key})['result']
        result[name]={'points_count':raw['points_count'],'vectors':raw['config']['params']['vectors']}
    return result


def route_checks(providers,cache):
    samples=[('法律咨询','我在上海一家私营公司工作两年，有劳动合同和银行工资流水，公司拖欠我三个月工资共18000元，应该怎样维权？'),
        ('法条解释','请解释劳动合同法第四十七条中经济补偿的计算规则，不涉及具体纠纷。'),
        ('程序咨询','劳动争议仲裁申请的时效一般多久？中断和中止分别怎么处理？'),
        ('案例检索','请查找有关追索劳动报酬的类似案例，说明法院的裁判思路。')]
    result=[]
    for number,(expected,query) in enumerate(samples,1):
        captured=[]
        class CapturedChat:
            def chat(self,messages,temperature=0.2):
                text,seconds=cache.get(f'route_{number}',{'query':query,'model':providers.chat.model,
                    'messages':messages},'query_understanding',lambda:providers.chat.chat(messages,temperature))
                captured.append(text); return text
        understanding=understand_query(CapturedChat(),query,[],ConversationState(f'ablation_route_{number}'))
        if not captured: raise ValueError('No live/cached real query-understanding response')
        raw=_extract_json(captured[0])  # A parse/rule fallback must not count as a successful LLM check.
        if not raw.get('query_type') or not raw.get('intents'): raise ValueError('Incomplete LLM intent output')
        plan=route_skills(query,understanding); actual=resolve_intent(understanding)
        result.append({'query':query,'expected':expected,'actual':actual,'matches_expected':expected==actual,
            'raw_llm_query_type':raw['query_type'],'raw_llm_intents':raw['intents'],
            'understanding':to_dict(understanding),'plan':to_dict(plan)})
        print(f'route {number}/4: {actual}; expected_match={expected==actual}',flush=True)
        write_json(OUT/'route_checks.json',result)
    return result


def prepare(item,r,providers,cache,index):
    qid,query=item['query_id'],item['query'];cache.monitor.context={'query_id':qid}
    vector,embedding_seconds=cache.get(qid+'_embedding',{'query':query,'model':providers.embedding.model,
        'dimension':providers.embedding.dimension},'embedding',lambda:r.embedding.embed(query))
    if len(vector)!=1024 or not all(math.isfinite(v) for v in vector):raise ValueError('Invalid real embedding')
    chunks=[];search_seconds=0
    for target,kind in [('statute_db','statute'),('case_db','case')]:
        collection=r._collection_for(target)
        hits,seconds=cache.get(qid+'_'+target,{'vector':digest(vector),'collection':collection,'top_k':30},
            'qdrant_search',lambda:r.qdrant.search(vector=vector,limit=30,collection=collection))
        search_seconds+=seconds
        if len(hits)!=30:raise ValueError('Expected 30 real hits per collection')
        for hit in hits:
            chunk=r._from_qdrant_hit(hit)
            chunk.payload['_retrieval_target']=target;chunk.payload['_source_collection']=collection
            if index.doc_meta[index.chunk_to_doc[chunk.chunk_id]]['doc_type']!=kind or chunk_source(chunk)!=kind:
                raise ValueError('Qdrant/local corpus source conflict')
            chunks.append(chunk)
    if len({c.chunk_id for c in chunks})!=60:raise ValueError('Duplicate candidate ID')
    pids=unique([c.parent_id for c in chunks]);start=time.perf_counter()
    with patch.object(r.parents,'_get_many_local',side_effect=AssertionError('JSON fallback prohibited')) as fallback:
        parent_map=r.parents.get_many(pids)
        if fallback.call_count:raise ValueError('Unexpected JSON fallback')
    mysql_seconds=time.perf_counter()-start
    for chunk in chunks:
        parent=parent_map[chunk.parent_id]
        if parent['text']!=index.parents[chunk.parent_id]['text'] or parent['title_path']!=index.parents[chunk.parent_id]['title_path']:
            raise ValueError('MySQL parent mismatch')
        chunk.parent_text=parent['text'];chunk.title_path=chunk.title_path or parent['title_path']
        if parent.get('document_title'):chunk.source_name=parent['document_title'].split('>')[0].strip()
    documents=[f'标题：{c.title_path}\n子块内容：{c.text}' for c in chunks]
    def rank(key,docs,top_n):
        def request():
            # Strict auditing outside Retriever's production fallback catch.
            return AuditedReranker(providers.rerank,ApiMonitor(retries=1,delay=0)).rerank(query,docs,top_n)
        return cache.get(qid+key,{'query':query,'documents':docs,'model':providers.rerank.model,'top_n':top_n},
                         'rerank',request)
    statute_rows,statute_seconds=rank('_rerank_statute',documents[:30],20)
    full_rows,full_seconds=rank('_rerank_all',documents,60)
    return {'chunks':chunks,'documents':documents,'statute_scores':statute_rows,'full_scores':full_rows,
        'timings':{'embedding_seconds':embedding_seconds,'search_seconds':search_seconds,
            'parent_seconds':mysql_seconds,'statute_rerank_seconds':statute_seconds,'full_rerank_seconds':full_seconds},
        'mysql_parent_count':len(pids),'rank_request':rank}


def top_n_check(item,prepared):
    truncated,seconds=prepared['rank_request']('_rerank_check20',prepared['documents'],20)
    full=sorted(prepared['full_scores'],key=lambda r:r['relevance_score'],reverse=True)[:20]
    ids_equal=[r['index'] for r in truncated]==[r['index'] for r in full]
    max_difference=max(abs(float(a['relevance_score'])-float(b['relevance_score'])) for a,b in zip(truncated,full))
    return {'query_id':item['query_id'],'candidate_count':60,'same_top20_indices':ids_equal,
        'max_score_difference':max_difference,'exact_scores_equal':all(a['relevance_score']==b['relevance_score'] for a,b in zip(truncated,full)),
        'equivalent_tolerance_1e_8':ids_equal and max_difference<=1e-8,'extra_request_seconds':seconds,
        'top20_response':truncated,'full_response_top20':full}


def evaluate(item,prepared,index,config,overrides=None):
    chunks=prepared['chunks'];allrows=prepared['full_scores'];variants={}
    ranked3=None
    for name in ('V1','V2','V3','V4'):
        start=time.perf_counter();pool=copy.deepcopy(chunks[:30] if name=='V1' else chunks)
        rows=prepared['statute_scores'] if name=='V1' else (overrides if name=='V2' and overrides else allrows)
        replay=Retriever(config,None,None,RecordedScores(rows))
        ranked=replay.rerank(item['query'],pool,5,split_rerank=name in ('V3','V4'),
            rerank_quotas=source_quotas(item['query_type'],20) if name in ('V3','V4') else None)
        if len(ranked)!=20 or any(c.rerank_score is None for c in ranked):raise ValueError('Replay fallback detected')
        if name=='V3':ranked3=[(c.chunk_id,c.rerank_score) for c in ranked]
        if name=='V4' and ranked3!=[(c.chunk_id,c.rerank_score) for c in ranked]:raise ValueError('V3/V4 rerank differs')
        evidence=build_evidence(ranked,5,source_quotas(item['query_type'],5) if name=='V4' else None)
        pr={'recall':unique([c.parent_id for c in pool]),'rerank':unique([c.parent_id for c in ranked]),
            'evidence':[e.parent_id for e in evidence.items]}
        dr={'recall':to_doc_rank([c.chunk_id for c in pool],index.chunk_to_doc),
            'rerank':to_doc_rank([c.chunk_id for c in ranked],index.chunk_to_doc),
            'evidence':to_doc_rank(pr['evidence'],index.parent_to_doc)}
        counts=Counter(index.doc_meta[index.parent_to_doc[e.parent_id]]['doc_type'] for e in evidence.items)
        variants[name]={**item,'variant':name,'doc_ranks':dr,'parent_ranks':pr,
            'scoped_metrics':metrics_for(item,dr,pr,index),'rerank_chunks':[to_dict(c) for c in ranked],
            'evidence_pack':to_dict(evidence),'evidence_counts':dict(counts),
            'rerank_fallback_count':0,'rerank_quotas':source_quotas(item['query_type'],20) if name in ('V3','V4') else None,
            'evidence_quotas':source_quotas(item['query_type'],5) if name=='V4' else None,
            'timings':{**prepared['timings'],'local_replay_seconds':time.perf_counter()-start,
                'retrieval_seconds':prepared['timings']['search_seconds']+prepared['timings']['parent_seconds']+
                    prepared['timings']['statute_rerank_seconds' if name=='V1' else 'full_rerank_seconds']}}
        variants[name]['evidence_counts']={kind:counts[kind] for kind in ('statute','case')}
    return variants


def run(offline=False):
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    items=freeze_queries(index);config=AppConfig.load();providers=Providers(config)
    if any(getattr(getattr(providers,n),'offline',False) for n in ('chat','embedding','qdrant','rerank')):
        raise ValueError('Real service providers required')
    monitor=ApiMonitor(retries=2,delay=0.25)
    if (OUT/'api_events.json').exists():monitor.events=json.loads((OUT/'api_events.json').read_text())
    cache=CachedCalls(monitor,cache_only=offline);retriever=Retriever(config,providers.embedding,providers.qdrant,providers.rerank)
    if not offline:
        if not (OUT/'qdrant_before.json').exists():write_json(OUT/'qdrant_before.json',snapshot(providers,config))
        route_checks(providers,cache)
    precheck_ids=[items[0]['query_id'],next(r['query_id'] for r in items if r['query_type']=='法条解释'),
                  next(r['query_id'] for r in items if r['query_type']=='案例检索')]
    prepared_by_id={};checks=[]
    for item in items:
        if item['query_id'] not in precheck_ids:continue
        prepared=prepare(item,retriever,providers,cache,index);prepared_by_id[item['query_id']]=prepared
        checks.append(top_n_check(item,prepared));write_json(OUT/'top_n_checks.json',checks)
        print(f'top_n check {item["query_id"]}: equivalent={checks[-1]["equivalent_tolerance_1e_8"]}',flush=True)
    equivalent=all(c['equivalent_tolerance_1e_8'] for c in checks)
    results={v:[] for v in ('V1','V2','V3','V4')};per_query=[]
    for number,item in enumerate(items,1):
        prepared=prepared_by_id.pop(item['query_id'],None)
        if prepared is None:prepared=prepare(item,retriever,providers,cache,index)
        override=None
        if not equivalent:override=prepared['rank_request']('_rerank_check20',prepared['documents'],20)[0]
        variants=evaluate(item,prepared,index,config,override)
        for name,result in variants.items():
            results[name].append(result)
            write_json(OUT/f'{name}.json',{'summary':summarize(results[name]),'groups':grouped(results[name]),'details':results[name]})
        per_query.append({'query_id':item['query_id'],'mysql_parent_count':prepared['mysql_parent_count'],
                          'timings':prepared['timings']})
        write_json(OUT/'query_timings.json',per_query)
        print(f'completed {number}/20 {item["query_id"]}: '+', '.join(f'{v}='+str(x['evidence_counts']) for v,x in variants.items()),flush=True)
    if not offline:
        after=snapshot(providers,config);write_json(OUT/'qdrant_after.json',after)
        if after!=json.loads((OUT/'qdrant_before.json').read_text()):raise ValueError('Qdrant metadata changed')
    protected=json.loads((OUT/'preflight_mysql.json').read_text())['protected_hashes']
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in protected.items()):
        raise ValueError('Protected config or historical report changed')
    invocation={'complete':True,'n_queries':20,'variants':list(results),
        'elapsed_this_invocation_seconds':time.perf_counter()-started,'cache_hits_this_invocation':cache.hits,
        'api_success_counts':dict(Counter(e['phase'] for e in monitor.events if e['ok'])),
        'api_failed_attempts':sum(not e['ok'] for e in monitor.events),'top_n_equivalent':equivalent,
        'v3_v4_same_rerank_all_queries':True,'config_and_historical_reports_unchanged':True,
        'qdrant_read_only':True,'mysql_json_fallback_calls':0,
        'timing_note':'Original real-call durations retained on cache replay; local replay is separately timed. Shared dual search/parent cost is charged to all variants for controlled comparison; V1 is not isolated single-library latency.'}
    first_path=OUT/'first_run_manifest.json'
    if not first_path.exists():write_json(first_path,invocation)
    if offline:write_json(OUT/'replay_manifest.json',invocation)
    else:write_json(OUT/'run_manifest.json',invocation)
    print('complete: 20 queries x 4 variants; protected files unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true',help='Reuse populated API cache; no route/snapshot API requests')
    args=parser.parse_args()
    try:run(args.replay)
    except Exception as exc:
        print('ablation_stopped: '+error_label(exc),flush=True)
        sys.exit(1)
