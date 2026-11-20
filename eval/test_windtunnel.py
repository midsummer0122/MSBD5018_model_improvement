"""Offline checks for metric fairness, frozen labels and hidden-rerank fallback detection."""
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from eval_retrieval import (ROOT_DIR, ApiMonitor, AuditedReranker, build_indexes, compute_metrics,
                           load_queries, validate_labels, to_doc_rank, CorpusIndex, evaluate_one)
from rag.config import AppConfig
from rag.retrieval import Retriever
from rag.schemas import RetrievedChunk


class WindTunnelChecks(unittest.TestCase):
    def test_source_scopes_do_not_remove_competing_results(self):
        # A case ranked first must move the statute from rank 1 to 2.
        score=compute_metrics(['case','statute'],['statute'],[1,5])
        self.assertEqual(score['mrr'],0.5)
        self.assertEqual(score['recall@1'],0)
        self.assertEqual(score['recall@5'],1)
        self.assertEqual(score['precision@5'],0.2)

    def test_dedup_and_empty_case_subset(self):
        self.assertEqual(to_doc_rank(['c1','c2','c3'],{'c1':'d1','c2':'d1','c3':'d2'}),['d1','d2'])
        self.assertEqual(compute_metrics(['x'],[],[5]),{})
        self.assertEqual(compute_metrics(['x','x','yes'],['yes'],[5])['mrr'],0.5)

    def test_real_production_fallback_is_detectable(self):
        provider=Mock();provider.rerank.side_effect=RuntimeError('simulated service outage')
        monitor=ApiMonitor(retries=1)
        retriever=Retriever(AppConfig({}),None,None,AuditedReranker(provider,monitor))
        ranked=retriever.rerank('test',[RetrievedChunk('c','p','text',score=0.9)],5)
        self.assertIsNone(ranked[0].rerank_score)
        self.assertFalse(monitor.events[-1]['ok'])

    def test_all_frozen_labels_and_legacy_examples_are_real(self):
        index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
        rows=load_queries(ROOT_DIR/'eval/queries_windtunnel_60.jsonl')
        validate_labels(rows,index)
        self.assertEqual(len(rows),60)
        self.assertEqual(sum(bool(r['relevant_case_doc_ids']) for r in rows),24)
        self.assertEqual(len(index.doc_meta),302)
        self.assertEqual(len(index.parents),860)
        validate_labels(load_queries(ROOT_DIR/'eval/queries.jsonl'),index)
        cd,pd,all_ids=index
        self.assertEqual(len(all_ids),302)

    def test_eval_uses_same_collection_parent_and_rerank_path_as_production(self):
        config=AppConfig({'qdrant':{'collection_name':'laws','cases_collection_name':'cases'}})
        embedding=Mock();embedding.embed.return_value=[1.0]
        qdrant=Mock()
        def search(**kwargs):
            cid,pid=('c1','p1') if kwargs['collection']=='laws' else ('c2','p2')
            return [{'id':cid,'score':0.8,'payload':{'chunk_id':cid,'parent_id':pid,'text':cid,'title_path':pid}}]
        qdrant.search.side_effect=search
        rerank=Mock();rerank.rerank.return_value=[{'index':1,'relevance_score':0.9},{'index':0,'relevance_score':0.7}]
        r=Retriever(config,embedding,qdrant,rerank)
        r.parents=Mock();r.parents.get_many.return_value={p:{'text':'parent '+p,'document_title':p} for p in ('p1','p2')}
        from rag.schemas import RetrievalPlan
        production=r.retrieve(RetrievalPlan('query',retrieval_targets=['statute_db','case_db'],top_k=30,final_k=5))
        idx=CorpusIndex({'c1':'d1','c2':'d2'},{'p1':'d1','p2':'d2'},{'c1':'p1','c2':'p2'},
             {'d1':{'doc_type':'statute'},'d2':{'doc_type':'case'}},
             {p:{'text':'parent '+p} for p in ('p1','p2')},{c:{'text':c} for c in ('c1','c2')})
        item={'query_id':'test','query':'query','relevant_doc_ids':['d1','d2'],
              'relevant_statute_doc_ids':['d1'],'relevant_case_doc_ids':['d2'],'relevant_parent_ids':['p1','p2']}
        result=evaluate_one(item,r,30,5,idx,['statute_db','case_db'],[1.0],ApiMonitor())
        self.assertEqual([c.chunk_id for c in production],[c['chunk_id'] for c in result['rerank_chunks']])
        self.assertEqual(embedding.embed.call_count,1) # Eval reused the supplied vector.
        self.assertEqual([c.kwargs['collection'] for c in qdrant.search.call_args_list],['laws','cases','laws','cases'])
        self.assertTrue(all(c.kwargs['limit']==30 for c in qdrant.search.call_args_list))


if __name__=='__main__':unittest.main()
