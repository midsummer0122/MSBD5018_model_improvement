"""C0 reuses locked M1; C1 invokes the real lightweight module, with no JSON fallback."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval_retrieval import (ROOT_DIR,AppConfig,build_indexes,load_queries,validate_labels,
    compute_metrics,to_doc_rank,error_label)
from run_hybrid_m123 import signature,rerank_payload
from rag.providers import Providers
from rag.retrieval import Retriever,ParentChunkStore
from rag.schemas import EvidencePack,EvidenceItem,QueryUnderstandingResult,to_dict
from rag.coverage import check_evidence_coverage
from rag.skill_router import route_skills
from rag.evidence import compact_text
from rag.lexical import corpus_fingerprint

OUT=ROOT_DIR/'eval/results_coverage_20'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def hashfile(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def protected_path(value):
    """Resolve exported old-machine paths against the current checkout when relocated."""
    p=Path(value)
    if not p.is_absolute():return ROOT_DIR/p
    try:p.relative_to(ROOT_DIR);return p
    except ValueError:
        normalized=value.replace('\\','/')
        for folder in ('api_key/','eval/','Refinement and Optimization/'):
            if '/'+folder in normalized:return ROOT_DIR/(folder+normalized.split('/'+folder,1)[1])
        raise ValueError('Unrecognized protected path')

def hydrate(value):return EvidencePack([EvidenceItem(**e) for e in value['items']],status=value['status'],issues=list(value['issues']))

def evidence_metrics(item,pack,index):
    pids=[e.parent_id for e in pack.items];docs=to_doc_rank(pids,index.parent_to_doc);result={}
    for scope,ref in [('statute_gt',item['relevant_statute_doc_ids']),('case_gt',item['relevant_case_doc_ids']),('all_gt',item['relevant_doc_ids'])]:
        parents=[pid for pid in item['relevant_parent_ids'] if index.parent_to_doc[pid] in ref]
        result[scope]={'parent':compute_metrics(pids,parents,[5]),'doc':compute_metrics(docs,ref,[5])}
    return result

def aggregate(rows):
    result={'queries':len(rows),'metrics':{}}
    for scope in ('statute_gt','case_gt','all_gt'):
        result['metrics'][scope]={}
        for grain in ('parent','doc'):
            values=[r['metrics'][scope][grain] for r in rows if r['metrics'][scope][grain]]
            result['metrics'][scope][grain]={'n':len(values),**({k:sum(v[k] for v in values)/len(values) for k in values[0]} if values else {})}
    return result

class Cache:
    def __init__(self,replay=False):
        self.replay=replay;self.context={};self.lookups=[]
        self.events=json.loads((OUT/'api_events.json').read_text()) if (OUT/'api_events.json').exists() else []
    def call(self,phase,payload,action):
        key=signature(payload);path=OUT/'cache'/phase/(key+'.json')
        if path.exists():
            value=json.loads(path.read_text());assert value['signature']==key
            self.lookups.append({**self.context,'phase':phase,'signature':key,'hit':True,'original_seconds':value['seconds']})
            write(OUT/'cache_events.json',self.lookups);return value['value']
        if self.replay:raise RuntimeError('Exact cache missing; no API request permitted')
        started=time.perf_counter()
        try:value=action()
        except Exception as exc:
            self.events.append({**self.context,'phase':phase,'signature':key,'ok':False,'error':error_label(exc),'seconds':time.perf_counter()-started})
            write(OUT/'api_events.json',self.events);raise
        seconds=time.perf_counter()-started
        self.events.append({**self.context,'phase':phase,'signature':key,'ok':True,'seconds':seconds})
        public={k:v for k,v in payload.items() if k!='endpoint'};public['endpoint_sha256']=hashlib.sha256(payload['endpoint'].encode()).hexdigest()
        write(path,{'signature':key,'request':public,'value':value,'seconds':seconds})
        write(OUT/'api_events.json',self.events)
        self.lookups.append({**self.context,'phase':phase,'signature':key,'hit':False,'original_seconds':seconds})
        write(OUT/'cache_events.json',self.lookups);return value

def providers_with_cache(config,cache,index):
    actual=Providers(config)
    if any(getattr(getattr(actual,n),'offline',False) for n in ('chat','embedding','qdrant','rerank')):raise ValueError('Real providers required')
    for p in (actual.chat,actual.embedding,actual.rerank,actual.qdrant):p.http.timeout=min(p.http.timeout,60)
    class Chat:
        def chat(self,messages,temperature):
            return cache.call('diagnosis',{'messages':messages,'temperature':temperature,'model':actual.chat.model,
                'endpoint':actual.chat.base_url+'/chat/completions'},lambda:actual.chat.chat(messages,temperature))
    class Embed:
        def embed(self,query):
            value=cache.call('embedding',{'query':query,'dimension':actual.embedding.dimension,'model':actual.embedding.model,
                'endpoint':actual.embedding.base_url+'/embeddings'},lambda:actual.embedding.embed(query))
            import math
            if len(value)!=1024 or not all(math.isfinite(v) for v in value):raise ValueError('Invalid real embedding')
            return value
    class Search:
        def search(self,**kwargs):
            rows=cache.call('qdrant_search',{'vector_sha256':signature(kwargs['vector']),'limit':kwargs['limit'],
                'collection':kwargs['collection'],'corpus_fingerprint':corpus_fingerprint()[0],'endpoint':actual.qdrant.url},
                lambda:actual.qdrant.search(**kwargs))
            for h in rows:
                p=h['payload'];local=index.children[p['chunk_id']]
                if any(p.get(k)!=local.get(k) for k in ('chunk_id','parent_id','title_path','text')):raise ValueError('Supplement payload drift')
            return rows
    class Rerank:
        def rerank(self,query,documents,top_n):
            return cache.call('unified_rerank',rerank_payload(actual.rerank,query,documents,top_n),
                lambda:actual.rerank.rerank(query,documents,top_n))
    return SimpleNamespace(chat=Chat(),embedding=Embed(),qdrant=Search(),rerank=Rerank()),actual

def snapshot(actual,config):
    q=actual.qdrant;result={}
    for key in ('collection_name','cases_collection_name'):
        name=config.section('qdrant')[key];base=q.url+'/collections/'+q.quote(name);headers={'api-key':q.api_key}
        data=q.http.request('GET',base,headers)['result']
        count=q.http.request('POST',base+'/points/count',headers,{'exact':True})['result']['count']
        result[name]={'count':count,'vectors':data['config']['params']['vectors']}
    return result

def run(replay=False):
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    lock_path=OUT/'protected_inputs.json'
    if not lock_path.exists():
        paths=[ROOT_DIR/p for p in ['api_key/api_config.yaml','eval/results_hybrid_20/M1.json','eval/queries_ablation_20.jsonl']]
        paths += [p for p in (ROOT_DIR/'Refinement and Optimization').glob('*.md') if p.name!='report3.md']
        write(lock_path,{p.relative_to(ROOT_DIR).as_posix():hashfile(p) for p in paths})
    locked=json.loads(lock_path.read_text())
    if any(hashfile(protected_path(p))!=sha for p,sha in locked.items()):raise ValueError('Protected input changed')
    source=ROOT_DIR/'eval/results_hybrid_20/M1.json';query_path=ROOT_DIR/'eval/queries_ablation_20.jsonl'
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases');queries=load_queries(query_path);validate_labels(queries,index)
    old_preflight=json.loads((ROOT_DIR/'eval/results_hybrid_20/preflight_hybrid.json').read_text())
    old_query_hashes=[sha for p,sha in old_preflight['protected_hashes'].items() if p.replace('\\','/').endswith('/eval/queries_ablation_20.jsonl')]
    if len(old_query_hashes)!=1 or old_query_hashes[0]!=hashfile(query_path):raise ValueError('Query hash differs from validated baseline')
    baseline=json.loads(source.read_text());assert len(baseline['details'])==len(queries)==20
    codes=['rag/coverage.py','rag/answer.py','rag/schemas.py','rag/skill_router.py','rag/workflow.py','eval/run_coverage_light.py']
    inputs={'M1_sha256':hashfile(source),'queries_sha256':hashfile(query_path),'corpus_fingerprint':corpus_fingerprint()[0],
            'code_sha256':{p:hashfile(ROOT_DIR/p) for p in codes}}
    if (OUT/'input_manifest.json').exists():
        if json.loads((OUT/'input_manifest.json').read_text())!=inputs:raise ValueError('Input/code changed; preserve results and use a fresh result directory')
    else:write(OUT/'input_manifest.json',inputs)
    c0=[]
    for q in queries:
        old=next(r for r in baseline['details'] if r['query_id']==q['query_id'])
        for key in ['query','query_type','relevant_doc_ids','relevant_statute_doc_ids','relevant_case_doc_ids','relevant_parent_ids','source_checks']:
            if old[key]!=q[key]:raise ValueError('M1 question/labels differ')
        pack=hydrate(old['evidence_pack']);m=evidence_metrics(q,pack,index)
        for e in pack.items:
            if e.content!=compact_text(index.parents[e.parent_id]['text'],900):raise ValueError('Baseline parent content mismatch')
        for scope in m:
            for grain in m[scope]:
                if m[scope][grain] and any(m[scope][grain][key]!=old['scoped_metrics'][scope][grain]['evidence'][key] for key in m[scope][grain]):raise ValueError('M1 metric mismatch')
        c0.append({**q,'initial_evidence_pack':old['evidence_pack'],'evidence_pack':old['evidence_pack'],
                   'metrics':m,'incremental_seconds':0.,'new_api_requests':0})
    write(OUT/'C0.json',{'details':c0,'all':aggregate(c0),'consultation':aggregate([r for r in c0 if r['query_type']=='法律咨询']),'new_api_requests':0})
    print('C0: 20 baseline rows, hashes and all metrics checked; 0 new requests',flush=True)
    cache=Cache(replay);config=AppConfig.load();p,actual=providers_with_cache(config,cache,index)
    if not replay and not (OUT/'qdrant_before.json').exists():write(OUT/'qdrant_before.json',snapshot(actual,config))
    parents=ParentChunkStore(config)
    with patch.object(parents,'_get_many_local',side_effect=AssertionError('JSON fallback prohibited')) as local:
        t=time.perf_counter();all_parents=parents.get_many(list(index.parents));seconds=time.perf_counter()-t
        assert set(all_parents)==set(index.parents) and local.call_count==0
        assert all(all_parents[pid]['text']==x['text'] and all_parents[pid]['title_path']==x['title_path'] for pid,x in index.parents.items())
    write(OUT/'mysql_preflight.json',{'parents':len(all_parents),'all_text_title_path_match':True,'json_fallback_calls':0,'seconds':seconds})
    r=Retriever(config,p.embedding,p.qdrant,p.rerank);c1=[]
    for number,row in enumerate(c0,1):
        path=OUT/'checkpoints'/(row['query_id']+'.json')
        if path.exists():
            value=json.loads(path.read_text());c1.append(value);print('resume '+row['query_id']+': '+value['coverage_check']['status'],flush=True);continue
        cache.context={'query_id':row['query_id']};before=len(cache.events);pack=hydrate(row['evidence_pack'])
        understanding=QueryUnderstandingResult(intents=[row['query_type']]);plan=route_skills(row['query'],understanding,enable_coverage_check=True)
        with patch.object(r.parents,'_get_many_local',side_effect=AssertionError('JSON fallback prohibited')) as fallback:
            result,trace=check_evidence_coverage(understanding,plan,pack,r,p.chat)
            if fallback.call_count:raise ValueError('MySQL proof failed: fallback attempted')
        if row['query_type']!='法律咨询':assert trace['status']=='gate_skipped' and len(cache.events)==before
        if not trace['replaced']:assert result is pack and to_dict(result)==row['evidence_pack']
        delta_events=cache.events[before:]
        value={**row,'evidence_pack':to_dict(result),'metrics':evidence_metrics(row,result,index),
            'coverage_check':trace,'incremental_seconds':trace['elapsed_seconds'],'new_api_requests':len(delta_events)}
        write(path,value);c1.append(value)
        write(OUT/'C1.json',{'details':c1,'all':aggregate(c1),'consultation':aggregate([x for x in c1 if x['query_type']=='法律咨询'])})
        print(f'{number}/20 {row["query_id"]}: {trace["status"]}; search={trace["triggered_search"]}, replaced={trace["replaced"]}',flush=True)
    assert sum(x['coverage_check']['status']=='gate_skipped' for x in c1)==12
    assert sum(x['coverage_check']['checked'] for x in c1)==8
    # Also rebuild the result if a prior run stopped after its last atomic checkpoint.
    write(OUT/'C1.json',{'details':c1,'all':aggregate(c1),'consultation':aggregate([x for x in c1 if x['query_type']=='法律咨询'])})
    if not replay:
        after=snapshot(actual,config);write(OUT/'qdrant_after.json',after)
        assert after==json.loads((OUT/'qdrant_before.json').read_text())
    assert all(hashfile(protected_path(p))==sha for p,sha in locked.items())
    manifest={'complete':True,'queries':20,'checked':8,'gate_skipped':12,'diagnosis_failures':sum(x['coverage_check']['status']=='check_failed' for x in c1),
        'searched_query_ids':[x['query_id'] for x in c1 if x['coverage_check']['triggered_search']],
        'replaced_query_ids':[x['query_id'] for x in c1 if x['coverage_check']['replaced']],
        'api_success_counts':dict(Counter(e['phase'] for e in cache.events if e['ok'])),'api_failed_attempts':sum(not e['ok'] for e in cache.events),
        'elapsed_this_invocation_seconds':time.perf_counter()-started,'total_query_incremental_seconds':sum(x['incremental_seconds'] for x in c1),
        'mysql_json_fallback_calls':0,'private_config_and_all_historical_reports_unchanged':True,'qdrant_writes':0,'replay':replay}
    write(OUT/('replay_manifest.json' if replay else 'run_manifest.json'),manifest)
    print('coverage experiment complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');args=p.parse_args()
    try:run(args.replay)
    except Exception as exc:print('coverage_experiment_stopped: '+error_label(exc));sys.exit(1)
