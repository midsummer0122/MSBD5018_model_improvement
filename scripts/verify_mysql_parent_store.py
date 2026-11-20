"""Verify every parent against MySQL; optionally test read-only dual-collection retrieval.

Passwords come only from the local YAML and are never printed. The retrieval test
reuses an existing real embedding CSV vector, so it makes zero Embedding/LLM calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.config import AppConfig
from rag.providers import Providers
from rag.retrieval import ParentChunkStore, Retriever
from rag.schemas import RetrievalPlan
from import_parent_chunks_to_mysql import load_parent_rows, sha256_text
from embed_json_children_ali import build_embedding_text


def connect(section):
    import pymysql
    return pymysql.connect(host=section['host'],port=int(section['port']),user=section['user'],
        password=section['password'],database=section['database'],charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,connect_timeout=5,autocommit=False)


def expected_rows():
    docs,parents=[],[]
    for directory in ('json','json_cases'):
        d,p=load_parent_rows(ROOT/'knowledge_data'/directory);docs.extend(d);parents.extend(p)
    if len({d['doc_id'] for d in docs})!=len(docs) or len({p['parent_id'] for p in parents})!=len(parents):
        raise ValueError('Cross-directory ID collision')
    return docs,parents


def verify_all(config):
    documents,parents=expected_rows()
    expected={p['parent_id']:p for p in parents};doc_map={d['doc_id']:d for d in documents}
    conn=connect(config.section('mysql'))
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT VERSION() AS version,@@port AS port,@@bind_address AS bind_address,CURRENT_USER() AS account')
            server=cursor.fetchone()
            cursor.execute('SELECT COUNT(*) AS n FROM documents');total_documents=cursor.fetchone()['n']
            cursor.execute('SELECT COUNT(*) AS n FROM parent_chunks');total_parents=cursor.fetchone()['n']
            section=config.section('mysql')
            grantee="'"+section['user']+"'@'127.0.0.1'"
            cursor.execute('SELECT TABLE_SCHEMA,PRIVILEGE_TYPE FROM information_schema.SCHEMA_PRIVILEGES WHERE GRANTEE=%s',(grantee,))
            privileges=[dict(row) for row in cursor.fetchall()]
            cursor.execute('SELECT PRIVILEGE_TYPE FROM information_schema.USER_PRIVILEGES WHERE GRANTEE=%s',(grantee,))
            global_privileges=[row['PRIVILEGE_TYPE'] for row in cursor.fetchall()]
            if any(row['TABLE_SCHEMA']!='law_rag' for row in privileges) or any(p!='USAGE' for p in global_privileges):
                raise ValueError('Project account has unexpected privileges outside law_rag')
            cursor.execute("SELECT DEFAULT_CHARACTER_SET_NAME AS charset FROM information_schema.SCHEMATA WHERE SCHEMA_NAME='law_rag'")
            charset=cursor.fetchone()['charset']
            assert charset=='utf8mb4'
            cursor.execute('''SELECT pc.parent_id,pc.title_path,pc.text,pc.child_ids,pc.parent_index,pc.content_hash,
                                      d.doc_id,d.title AS document_title,d.source_path
                               FROM parent_chunks pc LEFT JOIN documents d ON d.id=pc.document_id''')
            actual={row['parent_id']:row for row in cursor.fetchall()}
        for pid,wanted in expected.items():
            if pid not in actual:raise ValueError(f'Missing MySQL parent: {pid}')
            found=actual[pid]
            for key in ('doc_id','title_path','text','parent_index','content_hash'):
                if found[key]!=wanted[key]:raise ValueError(f'MySQL mismatch: {pid}, {key}')
            if found['document_title']!=doc_map[wanted['doc_id']]['title']:raise ValueError(f'Title mismatch: {pid}')
            if found['source_path']!=doc_map[wanted['doc_id']]['source_path']:raise ValueError(f'Source mismatch: {pid}')
            if json.loads(found['child_ids'])!=json.loads(wanted['child_ids']):raise ValueError(f'Child IDs mismatch: {pid}')
            if sha256_text(found['text'])!=found['content_hash']:raise ValueError(f'Hash mismatch: {pid}')
    finally:conn.close()
    store=ParentChunkStore(config)
    # A connection error, incomplete MySQL query or JSON fallback must fail this check.
    with patch.object(store,'_get_many_local',side_effect=AssertionError('JSON fallback prohibited in this verification')) as local:
        restored=store.get_many(list(expected))
        assert all(restored[pid]['text']==row['text'] for pid,row in expected.items())
        assert local.call_count==0 and store._local_index is None
    return {'server':server,'database':config.section('mysql')['database'],
        'database_documents':total_documents,'database_parents':total_parents,
        'expected_documents':len(documents),'expected_parents':len(parents),
        'statute_documents':sum(not d['doc_id'].startswith('5') for d in documents),
        'case_documents':sum(d['doc_id'].startswith('5') for d in documents),
        'statute_parents':sum(not p.startswith('5') for p in expected),
        'case_parents':sum(p.startswith('5') for p in expected),
        'all_text_ids_titles_links_hashes_match':True,'runtime_mysql_parents':len(restored),'json_fallback_calls':0,
        'charset':charset,'project_privileges':privileges,'global_privileges':global_privileges}


def verify_retrieval(config):
    # Reuse a real existing law child vector. Its exact embedding input is the query.
    law_doc=next(json.loads(p.read_text(encoding='utf-8-sig'))
        for p in (ROOT/'knowledge_data/json').glob('*.json')
        if json.loads(p.read_text(encoding='utf-8-sig'))['doc_id']=='00002')
    child=next(c for c in law_doc['children'] if '及时足额支付劳动报酬' in c['text'])
    query=build_embedding_text(child)
    vector=None
    for path in (ROOT/'knowledge_data/embedding_csv').glob('*.csv'):
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if row['chunk_id']==child['chunk_id']:
                    vector=json.loads(row['embedding']);break
        if vector is not None:break
    if vector is None or len(vector)!=1024 or not all(math.isfinite(v) for v in vector):
        raise ValueError('Existing real embedding is unavailable or invalid')
    providers=Providers(config)
    assert all(not getattr(getattr(providers,n),'offline',False) for n in ('embedding','rerank','qdrant'))
    retriever=Retriever(config,providers.embedding,providers.qdrant,providers.rerank)
    def cached_embedding(actual_query):
        if actual_query!=query:raise AssertionError('Unexpected embedding input')
        return vector
    with patch.object(providers.embedding,'embed',side_effect=cached_embedding), \
         patch.object(retriever.parents,'_get_many_local',side_effect=AssertionError('JSON fallback prohibited')) as local, \
         patch.object(retriever.parents,'_get_many_mysql',wraps=retriever.parents._get_many_mysql) as mysql, \
         patch.object(providers.rerank,'rerank',wraps=providers.rerank.rerank) as rerank:
        chunks=retriever.retrieve(RetrievalPlan(query=query,retrieval_targets=['statute_db','case_db'],top_k=3,final_k=2))
        assert len(chunks)==6 and all(c.rerank_score is not None for c in chunks),'Actual Rerank failed or fell back'
        kinds=Counter('case' if c.parent_id.startswith('5') else 'statute' for c in chunks)
        assert kinds['statute'] and kinds['case'] and mysql.call_count==1 and local.call_count==0
        expected={p['parent_id']:p['text'] for p in expected_rows()[1]}
        assert all(c.parent_text==expected[c.parent_id] for c in chunks)
    return {'query_embedding_source':'existing real law CSV vector','embedding_api_calls':0,'llm_calls':0,
            'qdrant_read_searches':2,'rerank_real_calls':rerank.call_count,'retrieved_children':len(chunks),
            'retrieved_source_counts':dict(kinds),'mysql_parent_query_calls':mysql.call_count,
            'mysql_unique_parents':len({c.parent_id for c in chunks}),'json_fallback_calls':local.call_count,
            'all_parent_texts_match':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval',action='store_true');args=parser.parse_args()
    config=AppConfig.load();result=verify_all(config)
    if args.retrieval:result['retrieval']=verify_retrieval(config)
    out=ROOT/'data/mysql_setup';out.mkdir(parents=True,exist_ok=True)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('MySQL verification failed:',type(exc).__name__,'code:',exc.args[0] if exc.args and isinstance(exc.args[0],int) else 'not disclosed')
        raise SystemExit(1)
