"""All-parent MySQL proof; missing rows may be incrementally imported after conflict checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

from eval_retrieval import ROOT_DIR, AppConfig, build_indexes

sys.path.insert(0,str(ROOT_DIR/'scripts'))
from import_parent_chunks_to_mysql import load_parent_rows,check_existing_rows,write_to_mysql
from verify_mysql_parent_store import connect
from rag.retrieval import ParentChunkStore


def run(output_dir, inserted_count=0):
    output_dir.mkdir(parents=True,exist_ok=True)
    config=AppConfig.load();index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (ROOT_DIR/'api_key/api_config.yaml',ROOT_DIR/'Refinement and Optimization/案例库构建与双库接入优化记录.md',
         ROOT_DIR/'Refinement and Optimization/report_windtunnel_ab.md')}
    snapshot_dir=output_dir/'baseline_source';snapshot_dir.mkdir(exist_ok=True)
    for name in ('schemas.py','skill_router.py','retrieval.py','evidence.py','workflow.py'):
        dst=snapshot_dir/name
        if not dst.exists():dst.write_bytes((ROOT_DIR/'rag'/name).read_bytes())
    conn=connect(config.section('mysql'))
    try:
        start=time.perf_counter()
        with conn.cursor() as cur:
            cur.execute('SELECT @@version AS version,@@port AS port,CURRENT_USER() AS account')
            server=cur.fetchone()
            cur.execute('''SELECT pc.parent_id,pc.text,pc.title_path,d.doc_id,d.title
                           FROM parent_chunks pc LEFT JOIN documents d ON d.id=pc.document_id''')
            actual={row['parent_id']:row for row in cur.fetchall()}
        direct_seconds=time.perf_counter()-start
        missing=set(index.parents)-set(actual)
        if missing:
            documents=[];parents=[]
            for dirname in ('json','json_cases'):
                d,p=load_parent_rows(ROOT_DIR/'knowledge_data'/dirname);documents+=d;parents+=p
            check_existing_rows(conn,documents,parents)
            needed_docs={index.parent_to_doc[pid] for pid in missing};section=config.section('mysql')
            args=argparse.Namespace(host=section['host'],port=section['port'],user=section['user'],
                password=section['password'],password_env=None,ask_password=False,database=section['database'],commit_every=20)
            write_to_mysql(args,[d for d in documents if d['doc_id'] in needed_docs],
                           [p for p in parents if p['parent_id'] in missing])
            return run(output_dir,inserted_count+len(missing))
        for pid,parent in index.parents.items():
            row=actual[pid];did=index.parent_to_doc[pid]
            if row['text']!=parent['text'] or row['title_path']!=parent['title_path'] or row['doc_id']!=did or row['title']!=index.doc_meta[did]['title']:
                raise ValueError(f'MySQL/source mismatch: {pid}')
    finally:conn.close()
    store=ParentChunkStore(config);timings={};counts={}
    with patch.object(store,'_get_many_local',side_effect=AssertionError('JSON fallback prohibited')) as local:
        for kind in ('statute','case'):
            ids=[pid for pid,did in index.parent_to_doc.items() if index.doc_meta[did]['doc_type']==kind]
            start=time.perf_counter();rows=store.get_many(ids);timings[kind]=time.perf_counter()-start
            assert set(rows)==set(ids)
            assert all(rows[pid]['text']==index.parents[pid]['text'] and rows[pid]['title_path']==index.parents[pid]['title_path'] for pid in ids)
            counts[kind]={'documents':sum(m['doc_type']==kind for m in index.doc_meta.values()),'parents':len(ids)}
        assert local.call_count==0 and store._local_index is None
    result={'server':server,'counts':counts,'source_type_basis':'JSON directory, never an ID prefix heuristic',
        'expected_parents':len(index.parents),'mysql_parents':len(actual),'all_expected_text_title_path_doc_links_match':True,
        'runtime_json_fallback_calls':0,'direct_read_seconds':direct_seconds,'runtime_mysql_seconds':timings,
        'protected_hashes':protected,'no_ddl_executed':True,'new_missing_parent_inserts':inserted_count}
    (output_dir/'preflight_mysql.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='protected_hashes'},ensure_ascii=False))
    return result


if __name__=='__main__':
    try:run(ROOT_DIR/'eval/results_ablation_20')
    except Exception as exc:
        print('MySQL preflight failed:',type(exc).__name__);raise SystemExit(1)
