"""M1/M2/M3 diagnostic; strict wire signatures, atomic cache, no Qdrant mutation."""
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

from eval_retrieval import (ROOT_DIR,AppConfig,ApiMonitor,AuditedReranker,build_indexes,load_queries,
    validate_labels,metrics_for,to_doc_rank,unique,summarize,grouped,error_label)
from rag.providers import Providers
from rag.retrieval import Retriever
from rag.lexical import get_lexical_index,corpus_fingerprint
from rag.schemas import QueryUnderstandingResult,ConversationState,Message,to_dict
from rag.skill_router import route_skills
from rag.evidence import build_evidence
from rag.query_understanding import understand_query,classify_topic_continuation,_extract_json
from rag.source_policy import chunk_source
from preflight_hybrid import cloud_state

OUT=ROOT_DIR/'eval/results_hybrid_20'
MODES={'M1':'dense','M2':'dense_bm25','M3':'hybrid'}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def signature(payload):return hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode()).hexdigest()

def rerank_payload(provider,query,documents,top_n):
    wire=provider.request_parameters(query,documents,top_n)
    return {'query':query,'documents':wire['documents'],'model':wire['model'],'top_n':wire['top_n'],
            'endpoint':provider.url,'instruct':wire['instruct']}

class Calls:
    def __init__(self,replay=False):
        self.monitor=ApiMonitor(retries=2,delay=.1);self.lookups=[];self.replay=replay
        if (OUT/'api_events.json').exists():self.monitor.events=json.loads((OUT/'api_events.json').read_text())

    def get(self,phase,payload,action,legacy=None):
        key=signature(payload);path=OUT/'cache'/phase/(key+'.json')
        origin='signature_cache'
        if path.exists():
            record=json.loads(path.read_text());assert record['signature']==key
            value,seconds=record['value'],record['seconds'];hit=True
        elif legacy is not None:
            value,seconds=legacy;hit=True;origin='validated_old_embedding'
        else:
            if self.replay:raise ValueError('Replay requires an exact full-signature cache hit')
            start=time.perf_counter()
            try:value=self.monitor.call(phase,action)
            finally:write(OUT/'api_events.json',self.monitor.events)
            seconds=time.perf_counter()-start;hit=False;origin='real_api'
        if not path.exists():
            # The endpoint participates in the hash but is not exported, nor are any headers/keys.
            public={k:v for k,v in payload.items() if k!='endpoint'}
            public['endpoint_sha256']=hashlib.sha256(payload['endpoint'].encode()).hexdigest()
            write(path,{'signature':key,'request':public,'value':value,'seconds':seconds,'origin':origin})
        self.lookups.append({**self.monitor.context,'phase':phase,'signature':key,'hit':hit,'origin':origin,'original_call_seconds':seconds})
        write(OUT/'cache_events.json',self.lookups)
        return value,seconds,hit

class StrictScores:
    def __init__(self,provider,calls):self.provider,self.calls=provider,calls;self.last={}
    def rerank(self,query,documents,top_n):
        request=rerank_payload(self.provider,query,documents,top_n)
        def action():return AuditedReranker(self.provider,ApiMonitor(retries=1,delay=0)).rerank(query,documents,top_n)
        value,seconds,hit=self.calls.get('rerank',request,action)
        self.last={'request_signature':signature(request),'seconds':seconds,'cache_hit':hit,'top_n':top_n}
        return value

def old_embedding(item,provider,config):
    old=ROOT_DIR/'eval/results_ablation_20'
    proof=json.loads((old/'preflight_mysql.json').read_text())
    previous=proof['protected_hashes'].get(str(ROOT_DIR/'api_key/api_config.yaml'))
    if previous!=hashlib.sha256((ROOT_DIR/'api_key/api_config.yaml').read_bytes()).hexdigest():return None
    path=old/'cache'/(item['query_id']+'_embedding.json')
    if not path.exists():return None
    cached=json.loads(path.read_text())
    payload={'query':item['query'],'model':provider.model,'dimension':provider.dimension}
    expected=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if cached['signature']!=expected:return None
    vector=cached['value']
    if len(vector)!=1024 or not all(math.isfinite(v) for v in vector):raise ValueError('Invalid old real embedding')
    return vector,cached['seconds']

def route_checks(providers,calls):
    class Chat:
        def chat(self,messages,temperature=.2):
            value,_,_=calls.get('route_llm',{'query':messages,'model':providers.chat.model,
                'temperature':temperature,'endpoint':providers.chat.base_url+'/chat/completions'},
                lambda:providers.chat.chat(messages,temperature))
            _extract_json(value)  # Do not silently count parsing/rule fallbacks.
            return value
    samples=['请解释劳动合同法第四十七条的经济补偿规则。','劳动争议司法解释（二）第二十条讲什么？',
        '请检索指导案例182号彭宇翔案的裁判思路。','换个话题，请解释劳动法第四十四条的加班工资规则。']
    results=[];lexical,_=get_lexical_index()
    for number,raw in enumerate(samples,1):
        calls.monitor.context={'query_id':'route_'+str(number),'variant':'routing'}
        state=ConversationState('hybrid_route_'+str(number));continuation=None
        if number==4:
            state.known_entities={'cases':['指导案例182号']};state.satisfied_info=['原话题：指导案例182号奖金']
            messages=[Message('user',samples[2]),Message('assistant','上述是奖金案例的说明。')]
            continuation=classify_topic_continuation(Chat(),raw,messages,state)
            if continuation.is_continuation:raise ValueError('Explicit topic switch misclassified')
            state.known_entities=continuation.retained_entities;state.satisfied_info=continuation.retained_satisfied
        understanding=understand_query(Chat(),raw,[Message('user',raw)],state)
        plan=route_skills(raw,understanding)
        hits={kind:lexical.exact_search(plan.query,kind,plan.raw_query) for kind in ('statute','case')}
        probe=Retriever(providers.config,None,None,None)
        selected=probe.initial_candidates(plan,dense_hits={'statute_db':[],'case_db':[]})
        assert plan.raw_query==raw and plan.retrieval_mode=='hybrid'
        if number==4 and any('guiding_case_182' in r.get('matched_owned_case_ids',[]) for r in hits['case']):
            raise ValueError('Old topic identifier leaked')
        results.append({'sample':number,'understanding':to_dict(understanding),'plan':to_dict(plan),
            'topic_continuation':to_dict(continuation) if continuation else None,
            'exact_high_ids':{k:[r['chunk_id'] for r in v if r['level']=='high'] for k,v in hits.items()},
            'initial_candidates_without_dense_probe':len(selected),'old_topic_identifier_leaked':False,
            'answer_generation_calls':0})
        write(OUT/'route_checks.json',results);print(f'route {number}/4 passed',flush=True)

def row_result(item,plan,pool,ranked,evidence,index,timings,request):
    pr={'recall':unique([c.parent_id for c in pool]),'rerank':unique([c.parent_id for c in ranked]),'evidence':[e.parent_id for e in evidence.items]}
    dr={'recall':to_doc_rank([c.chunk_id for c in pool],index.chunk_to_doc),
        'rerank':to_doc_rank([c.chunk_id for c in ranked],index.chunk_to_doc),'evidence':to_doc_rank(pr['evidence'],index.parent_to_doc)}
    counts=Counter(index.doc_meta[index.parent_to_doc[e.parent_id]]['doc_type'] for e in evidence.items)
    return {**item,'plan':to_dict(plan),'recall_chunks':[to_dict(c) for c in pool],
        'rerank_chunks':[to_dict(c) for c in ranked],'evidence_pack':to_dict(evidence),
        'doc_ranks':dr,'parent_ranks':pr,'scoped_metrics':metrics_for(item,dr,pr,index),
        'rerank_request':request,'timings':timings,'rerank_fallback_count':0,
        'evidence_counts':{k:counts[k] for k in ('statute','case')}}

def run(replay=False,limit=20):
    start=time.perf_counter();proof=json.loads((OUT/'preflight_hybrid.json').read_text())
    if corpus_fingerprint()[0]!=proof['corpus_fingerprint']:raise ValueError('Corpus changed since preflight')
    implementation=hashlib.sha256((ROOT_DIR/'rag/lexical.py').read_bytes()).hexdigest()
    if proof.get('lexical_implementation_sha256',implementation)!=implementation:raise ValueError('Lexical code changed since preflight')
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    rows=load_queries(ROOT_DIR/'eval/queries_ablation_20.jsonl');validate_labels(rows,index);rows=rows[:limit]
    providers=Providers();calls=Calls(replay)
    if any(getattr(getattr(providers,n),'offline',False) for n in ('embedding','qdrant','rerank','chat')):raise ValueError('Real providers required')
    for p in (providers.embedding,providers.qdrant,providers.rerank,providers.chat):p.http.timeout=min(p.http.timeout,60)
    if not replay:route_checks(providers,calls)
    scorer=StrictScores(providers.rerank,calls);r=Retriever(providers.config,providers.embedding,providers.qdrant,scorer)
    get_lexical_index();details={v:[] for v in MODES}
    for number,item in enumerate(rows,1):
        qid=item['query_id'];calls.monitor.context={'query_id':qid,'variant':'shared'}
        vector,embedding_seconds,embedding_hit=calls.get('embedding',{'query':item['query'],'model':providers.embedding.model,
            'dimension':providers.embedding.dimension,'endpoint':providers.embedding.base_url+'/embeddings'},
            lambda:providers.embedding.embed(item['query']),legacy=old_embedding(item,providers.embedding,providers.config))
        dense_hits={};search_seconds=0
        for target,kind in [('statute_db','statute'),('case_db','case')]:
            collection=r._collection_for(target)
            hits,seconds,_=calls.get('qdrant_search',{'vector_sha256':signature(vector),'collection':collection,
                'limit':30,'state_signature':proof['qdrant'][kind]['state_signature'],
                'payload_sha256':proof['qdrant'][kind]['payload_sha256'],'endpoint':providers.qdrant.url},
                lambda:providers.qdrant.search(vector=vector,limit=30,collection=collection))
            if len(hits)!=30:raise ValueError('Incomplete dense recall')
            for hit in hits:
                p=hit['payload'];local=index.children[p['chunk_id']]
                if any(p.get(k)!=local.get(k) for k in ('chunk_id','parent_id','title_path','text')):raise ValueError('Dense payload drift')
            dense_hits[target]=hits;search_seconds+=seconds
        for variant,mode in MODES.items():
            calls.monitor.context={'query_id':qid,'variant':variant};understanding=QueryUnderstandingResult(intents=[item['query_type']])
            plan=route_skills(item['query'],understanding,retrieval_mode=mode)
            local_start=time.perf_counter();pool=r.initial_candidates(plan,dense_hits=dense_hits)
            initial_seconds=time.perf_counter()-local_start
            if len(pool)!=60 or len({c.chunk_id for c in pool})!=60:raise ValueError('Fusion must retain 30 per source')
            parent_start=time.perf_counter()
            with patch.object(r.parents,'_get_many_local',side_effect=AssertionError('JSON fallback prohibited')) as fallback:
                parent_map=r.enrich_parents(pool)
                if fallback.call_count:raise ValueError('JSON fallback detected')
            parent_seconds=time.perf_counter()-parent_start
            for c in pool:
                if c.parent_text!=index.parents[c.parent_id]['text']:raise ValueError('MySQL context mismatch')
            before=copy.deepcopy(pool);rank_start=time.perf_counter();events_before=len(calls.monitor.events)
            ranked=r.rerank(plan.query,pool,5,split_rerank=True,rerank_quotas=plan.rerank_quotas,retrieval_mode=mode)
            if r.last_rerank_fallback or len(ranked)!=20 or any(c.rerank_score is None for c in ranked):
                write(OUT/'fallback_failure.json',{'query_id':qid,'variant':variant,'rerank_fallback':True})
                raise ValueError('Rerank fallback; results not scored')
            rank_wall=time.perf_counter()-rank_start;ev_start=time.perf_counter()
            evidence=build_evidence(ranked,5,plan.evidence_quotas);ev_seconds=time.perf_counter()-ev_start
            timing={'embedding_seconds_original':embedding_seconds,'search_seconds_original':search_seconds,
                'local_initial_seconds':initial_seconds,'parent_seconds':parent_seconds,
                'rerank_seconds_original':scorer.last['seconds'],'rerank_wall_seconds':rank_wall,
                'evidence_seconds':ev_seconds,
                'retrieval_seconds':search_seconds+initial_seconds+parent_seconds+scorer.last['seconds']+ev_seconds}
            result=row_result(item,plan,before,ranked,evidence,index,timing,dict(scorer.last));details[variant].append(result)
            write(OUT/(variant+'.json'),{'summary':summarize(details[variant]),'groups':grouped(details[variant]),'details':details[variant]})
        print(f'completed {number}/{len(rows)} {qid}: '+', '.join(v+'='+str(details[v][-1]['evidence_counts']) for v in MODES),flush=True)
    if not replay:
        after=cloud_state(providers,providers.config,index,full=True)
        if after!=proof['qdrant']:raise ValueError('Cloud state changed during experiment')
        write(OUT/'qdrant_after.json',after)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in proof['protected_hashes'].items()):
        raise ValueError('Protected downstream policy/config/data changed')
    if hashlib.sha256((ROOT_DIR/'rag/lexical.py').read_bytes()).hexdigest()!=implementation:raise ValueError('Lexical code changed during run')
    manifest={'complete':len(rows)==20,'queries':len(rows),'variants':MODES,'oracle_intent':True,'mysql_json_fallback_calls':0,
        'downstream_v4_and_parent_score_frozen':True,'qdrant_writes':0,'elapsed_seconds':time.perf_counter()-start,
        'api_success_counts':dict(Counter(e['phase'] for e in calls.monitor.events if e['ok'])),
        'api_failed_attempts':sum(not e['ok'] for e in calls.monitor.events),
        'cache_lookups':len(calls.lookups),'cache_hits':sum(e['hit'] for e in calls.lookups),
        'unique_rerank_signatures':len({e['signature'] for e in calls.lookups if e['phase']=='rerank'}),
        'old_rerank_reused':False,'old_rerank_reason':'No old endpoint/instruct in the stored signature; conservative fresh M1 scoring',
        'replay':replay}
    write(OUT/('replay_manifest.json' if replay else 'run_manifest.json'),manifest)
    print('hybrid experiment complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');p.add_argument('--limit',type=int,default=20)
    args=p.parse_args()
    try:run(args.replay,args.limit)
    except Exception as e:print('hybrid_stopped: '+error_label(e),flush=True);sys.exit(1)
