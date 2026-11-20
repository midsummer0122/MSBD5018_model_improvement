"""Run the fixed A / B / B-Fair experiment with shared embedding and resumable checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval_retrieval import (ROOT_DIR, ApiMonitor, AppConfig, build_indexes, load_queries, validate_labels,
                            make_retriever, evaluate_one, save_result, error_label)

VERSIONS = {'A':{'targets':['statute_db'],'top_k':30},
            'B':{'targets':['statute_db','case_db'],'top_k':30},
            'B_fair':{'targets':['statute_db','case_db'],'top_k':15}}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries',type=Path,default=ROOT_DIR/'eval/queries_windtunnel_60.jsonl')
    parser.add_argument('--output-dir',type=Path,default=ROOT_DIR/'eval/results_windtunnel')
    parser.add_argument('--request-sleep',type=float,default=0.25)
    parser.add_argument('--limit',type=int,default=0,help='Diagnostic subset only; 0 runs all 60.')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    annotation=json.loads((ROOT_DIR/'eval/annotation_validation.json').read_text(encoding='utf-8'))
    if file_hash(args.queries)!=annotation['dataset_sha256']:raise ValueError('Dataset differs from frozen annotation')
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    rows=load_queries(args.queries);validate_labels(rows,index)
    if args.limit: rows=rows[:args.limit]
    config=AppConfig.load();monitor=ApiMonitor(delay=args.request_sleep)
    r=make_retriever(config,index,monitor,ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    protected={str(p):file_hash(p) for p in (ROOT_DIR/'api_key/api_config.yaml',
        ROOT_DIR/'Refinement and Optimization/案例库构建与双库接入优化记录.md')}
    # Read only: validate both collections and snapshot exact counts before/after the experiment.
    def remote_snapshot():
        info={}
        for target in ('statute_db','case_db'):
            collection=r._collection_for(target)
            base=r.qdrant.url+'/collections/'+r.qdrant.quote(collection)
            headers={'api-key':r.qdrant.api_key}
            value=monitor.call('qdrant_preflight',lambda:r.qdrant.http.request('GET',base,headers))['result']
            count=monitor.call('qdrant_count',lambda:r.qdrant.http.request('POST',base+'/points/count',headers,{'exact':True}))['result']['count']
            vectors=value['config']['params']['vectors']
            if vectors.get('size')!=1024 or vectors.get('distance')!='Cosine':raise ValueError('Unexpected vector config')
            expected=sum(index.doc_meta[index.chunk_to_doc[c]]['doc_type']==('statute' if target=='statute_db' else 'case') for c in index.children)
            if count!=expected:raise ValueError('Remote/local point counts differ; inspect before benchmarking')
            info[target]={'collection':collection,'count':count,'vectors':vectors}
        return info

    manifest_path=args.output_dir/'run_manifest.json'
    prior=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    signature={'dataset_sha256':file_hash(args.queries),'catalog_sha256':annotation['catalog_sha256'],
               'embedding_model':r.embedding.model,'dimension':r.embedding.dimension,'rerank_model':r.rerank_provider.provider.model,
               'collection_names':[r._collection_for(t) for t in ('statute_db','case_db')],
               'endpoint_fingerprint':hashlib.sha256((r.embedding.base_url+r.qdrant.url+r.rerank_provider.provider.url).encode()).hexdigest()}
    if prior and prior['signature']!=signature:raise ValueError('Prior run inputs differ; use a new output directory')
    remote_before=prior['remote_before'] if prior else remote_snapshot()
    manifest=prior or {'signature':signature,'versions':VERSIONS,'final_k':5,'rerank_keep':20,
                     'started_at_utc':datetime.now(timezone.utc).isoformat(),'remote_before':remote_before,
                     'protected_hashes':protected,'mock_count':0,'query_rewriting':False,
                     'query_order':'frozen q001..q060','version_order':'rotate A,B,B_fair by query index',
                     'timing_notes':'retrieval_seconds excludes shared embedding and initial local JSON index load; includes actual MySQL/local fallback attempts',
                     'data_leakage_control':'dataset and source quotation checks frozen before first remote search; no tuning on dev or test'}
    cache_path=args.output_dir/'query_embedding_cache.json'
    cache=json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    details,events={},{}
    for version in VERSIONS:
        path=args.output_dir/f'version_{version}.json'
        old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        details[version]=old.get('details',[]);events[version]=old.get('api_events',[])
    shared_events=prior.get('shared_api_events',[]) if prior else []
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    def save_manifest(status):
        manifest.update(status=status,shared_api_events=shared_events,
                        completed={v:len(details[v]) for v in VERSIONS},embedding_cache_entries=len(cache))
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        for number,item in enumerate(rows):
            qid=item['query_id'];monitor.context={'query_id':qid,'version':'shared'}
            cache_key=hashlib.sha256((json.dumps(signature,sort_keys=True)+item['query']).encode()).hexdigest()
            if cache_key not in cache:
                before=len(monitor.events);t=time.perf_counter()
                vector=monitor.call('embedding',lambda:r.embedding.embed(item['query']))
                if len(vector)!=1024 or not all(math.isfinite(v) for v in vector) or not any(vector):raise ValueError('Invalid query vector')
                cache[cache_key]={'query_id':qid,'vector':vector,'seconds':time.perf_counter()-t}
                cache_path.write_text(json.dumps(cache,ensure_ascii=False),encoding='utf-8')
                shared_events.extend(monitor.events[before:])
            vector=cache[cache_key]['vector']
            order=list(VERSIONS);order=order[number%3:]+order[:number%3]
            for version in order:
                if any(row['query_id']==qid for row in details[version]):continue
                monitor.context={'query_id':qid,'version':version};before=len(monitor.events)
                spec=VERSIONS[version]
                result=evaluate_one(item,r,spec['top_k'],5,index,spec['targets'],vector,monitor)
                result['shared_embedding_seconds']=cache[cache_key]['seconds']
                details[version].append(result);details[version].sort(key=lambda x:x['query_id'])
                events[version].extend(monitor.events[before:])
                metadata={**signature,'version':version,**spec,'final_k':5,'rerank_keep':20,
                          'mock_count':0,'shared_embedding':True}
                save_result(args.output_dir/f'version_{version}.json',details[version],metadata,events[version])
                save_manifest('running')
                m=result['scoped_metrics']['statute_gt']['parent']['evidence']
                print(f"[{number+1}/{len(rows)}] {qid} {version}: statute parent R@5={m.get('recall@5',0):.3f}, {result['timings']['retrieval_seconds']:.2f}s",flush=True)
                time.sleep(args.request_sleep)
        monitor.context={'version':'shared','phase':'postflight'}
        manifest['remote_after']=remote_snapshot()
        if manifest['remote_before']!=manifest['remote_after']:raise ValueError('Remote collection snapshot changed during run')
        if any(file_hash(Path(path))!=sha for path,sha in protected.items()):raise ValueError('Protected file changed')
        manifest.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),protected_files_unchanged=True,
                        remote_unchanged=True,embedding_successes=sum(e['phase']=='embedding' and e['ok'] for e in shared_events),
                        query_count=len(rows),rerank_fallback_count=sum(e['phase']=='rerank_fallback' for e in monitor.events))
        save_manifest('complete')
        print('Wind-tunnel complete:',manifest['completed'],flush=True)
    except Exception as exc:
        manifest['failure']={'type':error_label(exc),'query_id':monitor.context.get('query_id'),
                             'version':monitor.context.get('version')}
        manifest['failed_attempt_events']=monitor.events[-5:]
        save_manifest('failed')
        raise
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print('Wind-tunnel failed:',error_label(exc),flush=True)
        raise SystemExit(1)
