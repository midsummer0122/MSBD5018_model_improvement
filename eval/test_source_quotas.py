"""Offline behavioral checks; API scores in these unit fixtures are synthetic."""
import math
import unittest
from unittest.mock import Mock

from eval_retrieval import ROOT_DIR
from rag.config import AppConfig
from rag.evidence import build_evidence
from rag.retrieval import Retriever
from rag.schemas import QueryUnderstandingResult, RetrievedChunk
from rag.skill_router import route_skills
from rag.source_policy import resolve_intent, source_quotas, select_by_source, chunk_source, local_source_index


def child(cid,pid,kind,score):
    return RetrievedChunk(cid,pid,'fixture child',rerank_score=score,
        payload={'doc_type':kind},parent_text='fixture parent')


class QuotaChecks(unittest.TestCase):
    def test_intent_priority_and_route_wiring(self):
        u=QueryUnderstandingResult(query_type='legal_consultation',intents=['程序咨询','案例检索'])
        self.assertEqual(resolve_intent(u),'案例检索')
        self.assertEqual(route_skills('q',u).rerank_quotas,{'statute':8,'case':12})
        self.assertEqual(resolve_intent(QueryUnderstandingResult(query_type='legal_consultation',intents=['程序咨询'])),'法律咨询')
        self.assertEqual(resolve_intent('unknown'),'法律咨询')

    def test_all_policies_scaling_and_single_library(self):
        for intent,expected in [('法条解释',(16,4)),('程序咨询',(16,4)),('法律咨询',(12,8)),('案例检索',(8,12))]:
            self.assertEqual(tuple(source_quotas(intent,20).values()),expected)
            for n in range(1,25):self.assertEqual(sum(source_quotas(intent,n).values()),n)
        self.assertEqual(source_quotas('案例检索',20,['statute_db']),{'statute':20,'case':0})
        plan=route_skills('q',QueryUnderstandingResult(),final_k=7,retrieval_targets=['case_db'])
        self.assertEqual(plan.rerank_quotas,{'statute':0,'case':28})
        self.assertEqual(plan.evidence_quotas,{'statute':0,'case':7})

    def test_shortage_fill_and_global_score_order(self):
        items=[child('a','p1','statute',.9),child('b','p2','case',.8),child('c','p3','case',.7)]
        result=select_by_source(items,3,{'statute':2,'case':1},chunk_source,lambda c:c.rerank_score)
        self.assertEqual([c.chunk_id for c in result],['a','b','c'])

    def test_full_score_request_once_and_quota_not_separate_scores(self):
        chunks=[child(str(i),str(i),'statute' if i<5 else 'case',0) for i in range(10)]
        provider=Mock();provider.rerank.return_value=[{'index':i,'relevance_score':1-i*.05} for i in range(10)]
        r=Retriever(AppConfig({}),None,None,provider)
        ranked=r.rerank('q',chunks,2,split_rerank=True,rerank_quotas={'statute':4,'case':4})
        self.assertEqual(provider.rerank.call_count,1)
        self.assertEqual(provider.rerank.call_args.kwargs['top_n'],10)
        self.assertEqual([c.chunk_id for c in ranked],['0','1','2','3','5','6','7','8'])

    def test_global_off_does_not_apply_quota(self):
        chunks=[child(str(i),str(i),'statute' if i<5 else 'case',0) for i in range(10)]
        provider=Mock();provider.rerank.return_value=[{'index':i,'relevance_score':1-i*.05} for i in range(8)]
        r=Retriever(AppConfig({}),None,None,provider)
        ranked=r.rerank('q',chunks,2,split_rerank=False,rerank_quotas={'statute':0,'case':8})
        self.assertEqual(provider.rerank.call_args.kwargs['top_n'],8)
        self.assertEqual([c.chunk_id for c in ranked],list(map(str,range(8))))

    def test_parent_formula_unchanged_and_evidence_quota_only_selection(self):
        chunks=[child('s1','s','statute',.8),child('s2','s','statute',.7),
            child('c1','c','case',.9),child('c2','c2','case',.85)]
        legacy=build_evidence(chunks,2)
        quota=build_evidence(chunks,2,{'statute':1,'case':1})
        self.assertEqual([e.parent_id for e in legacy.items],['c','c2'])
        self.assertEqual([e.parent_id for e in quota.items],['c','s'])
        self.assertAlmostEqual(quota.items[1].score,.8+.02*math.log1p(2))
        self.assertAlmostEqual(quota.items[0].score,.9+.02*math.log1p(1))
        self.assertEqual([e.evidence_id for e in quota.items],['ev_001','ev_002'])

    def test_directory_mapping_includes_nonzero_statute_ids(self):
        index=local_source_index()
        laws=[key for key,kind in index.items() if kind=='statute' and key[:1] in ('1','2','3')]
        self.assertTrue(laws)
        for key in laws[:5]:self.assertEqual(chunk_source(RetrievedChunk('',key,'')),'statute')
        self.assertEqual(chunk_source(RetrievedChunk('anything','anything','',payload={'_retrieval_target':'case_db'})),'case')


if __name__=='__main__':unittest.main()
