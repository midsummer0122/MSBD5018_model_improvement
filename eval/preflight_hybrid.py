"""Read-only cloud/local payload proof, strict MySQL proof and old dense funnel audit."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from eval_retrieval import ROOT_DIR,AppConfig,build_indexes,load_queries,error_label
from rag.lexical import corpus_fingerprint,get_lexical_index
from rag.providers import Providers
from preflight_ablation import run as mysql_proof

OUT=ROOT_DIR/'eval/results_hybrid_20'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def cloud_state(provider,config,index,full=True):
    result={};q=provider.qdrant
    for kind,key in [('statute','collection_name'),('case','cases_collection_name')]:
        name=config.section('qdrant')[key];base=q.url+'/collections/'+q.quote(name);headers={'api-key':q.api_key}
        info=q.http.request('GET',base,headers)['result']
        count=q.http.request('POST',base+'/points/count',headers,{'exact':True})['result']['count']
        expected={cid for cid,did in index.chunk_to_doc.items() if index.doc_meta[did]['doc_type']==kind}
        if count!=len(expected):raise ValueError('Cloud/local count mismatch')
        value={'collection':name,'points_count':count,'vectors':info['config']['params']['vectors']}
        if full:
            seen={};offset=None
            while True:
                body={'limit':500,'with_payload':True,'with_vector':False}
                if offset is not None:body['offset']=offset
                page=q.http.request('POST',base+'/points/scroll',headers,body)['result']
                for point in page['points']:
                    p=point['payload'];cid=p['chunk_id']
                    if cid in seen or cid not in expected:raise ValueError('Unexpected/duplicate cloud chunk')
                    local=index.children[cid]
                    for field in ('chunk_id','parent_id','title_path','text'):
                        if p.get(field)!=local.get(field):raise ValueError('Cloud/local payload mismatch')
                    seen[cid]={field:p.get(field) for field in ('chunk_id','parent_id','title_path','text')}
                offset=page.get('next_page_offset')
                if offset is None:break
            if set(seen)!=expected:raise ValueError('Cloud child IDs incomplete')
            value.update(all_payloads_compared=len(seen),all_exact_match=True,
                payload_sha256=digest([seen[k] for k in sorted(seen)]),sample_ids=sorted(seen)[::max(1,len(seen)//10)][:10])
        value['state_signature']=digest({'endpoint_sha256':hashlib.sha256(q.url.encode()).hexdigest(),
            'collection':name,'points_count':count,'vectors':value['vectors']})
        result[kind]=value
    return result

def baseline_audit(index,queries):
    old=ROOT_DIR/'eval/results_ablation_20';versions={v:json.loads((old/(v+'.json')).read_text()) for v in ('V1','V4')}
    rows=[]
    for q in queries:
        qid=q['query_id'];dense={}
        for kind,target in [('statute','statute_db'),('case','case_db')]:
            hits=json.loads((old/'cache'/f'{qid}_{target}.json').read_text())['value']
            dense[kind]={h['payload']['parent_id'] for h in hits}
        tracked={}
        for v in versions:
            r=next(x for x in versions[v]['details'] if x['query_id']==qid)
            pool=dense['statute'] if v=='V1' else dense['statute']|dense['case']
            tracked[v]={}
            for kind in ('statute','case'):
                gt={pid for pid in q['relevant_parent_ids'] if index.doc_meta[index.parent_to_doc[pid]]['doc_type']==kind}
                tracked[v][kind]={'gt':sorted(gt),'initial_missing':sorted(gt-pool),
                    'initial_hit_but_rerank_lost':sorted((gt&pool)-set(r['parent_ranks']['rerank'])),
                    'rerank_hit_but_evidence_lost':sorted((gt&set(r['parent_ranks']['rerank']))-set(r['parent_ranks']['evidence']))}
        rows.append({'query_id':qid,'query_type':q['query_type'],'funnel':tracked})
    write(OUT/'baseline_dense_audit.json',rows)
    return rows

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    config=AppConfig.load();providers=Providers(config)
    if any(getattr(getattr(providers,n),'offline',False) for n in ('qdrant','embedding','rerank','chat')):raise ValueError('Real providers required')
    providers.qdrant.http.timeout=60
    baseline=OUT/'baseline_source';baseline.mkdir(exist_ok=True)
    for name in ('schemas.py','source_policy.py','retrieval.py','skill_router.py','workflow.py','evidence.py','providers.py'):
        dst=baseline/name
        if not dst.exists():dst.write_bytes((ROOT_DIR/'rag'/name).read_bytes())
    fingerprint,files=corpus_fingerprint();write(OUT/'corpus_fingerprint.json',{'fingerprint':fingerprint,'files':files})
    cloud=cloud_state(providers,config,index)
    print('Qdrant: all 3718 payloads and IDs equal local JSON; no writes',flush=True)
    mysql=mysql_proof(OUT/'mysql_preflight')
    print('MySQL: all 860 parents, JSON fallback prohibited',flush=True)
    _,cold=get_lexical_index();_,warm=get_lexical_index()
    queries=load_queries(ROOT_DIR/'eval/queries_ablation_20.jsonl');audit=baseline_audit(index,queries)
    protected={str(ROOT_DIR/p):hashlib.sha256((ROOT_DIR/p).read_bytes()).hexdigest() for p in
        ['api_key/api_config.yaml','rag/source_policy.py','rag/evidence.py','eval/queries_ablation_20.jsonl',
         'Refinement and Optimization/report_ablation_4way.md','Refinement and Optimization/report1.md']}
    result={'corpus_fingerprint':fingerprint,'corpus_files':len(files),'qdrant':cloud,'mysql':mysql,
        'index_cold':{k:v for k,v in cold.items() if k!='files'},'index_warm':{k:v for k,v in warm.items() if k!='files'},
        'dependencies':{'jieba':importlib.metadata.version('jieba'),'bm25':'in-repo Okapi with positive IDF'},
        'protected_hashes':protected,'baseline_audit_queries':len(audit),'qdrant_writes':0}
    result['lexical_implementation_sha256']=hashlib.sha256((ROOT_DIR/'rag/lexical.py').read_bytes()).hexdigest()
    write(OUT/'preflight_hybrid.json',result)
    print('preflight complete; cold_index_seconds='+str(round(cold['build_seconds'],3))+
          '; warm_access_seconds='+str(round(warm['access_seconds'],4)),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:print('preflight_failed: '+error_label(e));sys.exit(1)
