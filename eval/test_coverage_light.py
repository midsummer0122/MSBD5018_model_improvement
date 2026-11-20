"""Offline boundary tests; synthetic fixtures never enter the real experiment."""
import copy
import json
import unittest
from unittest.mock import Mock,patch

from eval_retrieval import ROOT_DIR
from rag.answer import format_evidence_text,_answer_messages
from rag.coverage import check_evidence_coverage,diagnosis_messages
from rag.schemas import (EvidenceItem,EvidencePack,RetrievedChunk,RetrievalPlan,
    QueryUnderstandingResult,ConversationState,FinalAnswer,to_dict)
from rag.skill_router import route_skills


def pack():
    return EvidencePack([EvidenceItem(f'ev_{i+1:03d}','parent body '+str(i),'source','title',
        chunk_id='old-child'+str(i),parent_id='old'+str(i),score=.9-i*.1,
        matched_children=[{'chunk_id':'old-child'+str(i),'text':'important snippet after the parent opening','rerank_score':.8}])
        for i in range(5)],status='sufficient')

def chunk(pid,kind='statute',number=0):
    return RetrievedChunk(pid+'-child'+str(number),pid,'new child',parent_text='new full parent',score=.8,
                          payload={'doc_type':kind},title_path='new title')


class CoverageChecks(unittest.TestCase):
    def setup(self,diagnostic=None,chunks=None,scores=None):
        p=pack();r=Mock();r.initial_candidates.return_value=chunks or [chunk('new')]
        r.enrich_parents.return_value={c.parent_id:{'text':c.parent_text} for c in (chunks or [chunk('new')])}
        scores=scores or [.9,.1,.3,.4,.5,.8]
        r.rerank_provider.rerank.return_value=[{'index':i,'relevance_score':s} for i,s in enumerate(scores)]
        chat=Mock();chat.chat.return_value=json.dumps(diagnostic or {'sufficient':False,'missing_aspect':'core issue',
                'target_source':'statute','supplementary_query':'neutral legal concept'})
        self.addCleanup(patch.stopall)
        patch('rag.coverage.local_source_index',return_value={**{f'old{i}':'statute' if i<3 else 'case' for i in range(5)},'new':'statute','new-case':'case'}).start()
        return p,r,chat

    def run_check(self,p,r,chat,plan=None,intent='法律咨询'):
        return check_evidence_coverage(QueryUnderstandingResult(intents=[intent]),plan or RetrievalPlan('original question',enable_coverage_check=True),p,r,chat)

    def test_gate_off_is_identity_and_no_calls(self):
        p,r,chat=self.setup();before=copy.deepcopy(to_dict(p))
        result,t=self.run_check(p,r,chat,RetrievalPlan('q',enable_coverage_check=False))
        self.assertIs(result,p);self.assertEqual(to_dict(result),before);self.assertEqual(t['status'],'gate_skipped')
        chat.chat.assert_not_called();r.initial_candidates.assert_not_called()

    def test_all_other_intents_and_empty_pack_skip(self):
        for intent in ['法条解释','程序咨询','案例检索']:
            p,r,chat=self.setup();result,t=self.run_check(p,r,chat,intent=intent)
            self.assertIs(result,p);self.assertEqual(t['diagnosis_calls'],0);chat.chat.assert_not_called()
        p,r,chat=self.setup();p.items=[];self.assertEqual(self.run_check(p,r,chat)[1]['status'],'gate_skipped')

    def test_parse_error_wrong_boolean_or_multiple_queries_fail(self):
        for raw in ['not JSON',json.dumps({'sufficient':'false','missing_aspect':'x','target_source':'both','supplementary_query':'q'}),
                    json.dumps({'sufficient':False,'missing_aspect':'x','target_source':'both','supplementary_query':['q1','q2']})]:
            p,r,chat=self.setup();chat.chat.return_value=raw
            result,t=self.run_check(p,r,chat);self.assertIs(result,p);self.assertEqual(t['status'],'check_failed')
            self.assertFalse(t['triggered_search']);r.initial_candidates.assert_not_called()

    def test_timeout_is_failed_not_sufficient(self):
        p,r,chat=self.setup();chat.chat.side_effect=TimeoutError()
        result,t=self.run_check(p,r,chat);self.assertIs(result,p);self.assertEqual(t['status'],'check_failed')
        self.assertEqual(chat.chat.call_count,1)

    def test_diagnosis_uses_the_exact_answer_visible_evidence(self):
        p,r,chat=self.setup();messages=diagnosis_messages('q',p)
        visible=format_evidence_text(p)
        self.assertTrue(messages[1]['content'].endswith(visible))
        prompt=_answer_messages('q',[],ConversationState('unit'),QueryUnderstandingResult(),RetrievalPlan('q'),p)[1]
        self.assertIn(visible,prompt);self.assertIn('important snippet after the parent opening',messages[1]['content'])
        self.run_check(p,r,chat);self.assertEqual(chat.chat.call_args.kwargs['temperature'],0)

    def test_sufficient_does_not_require_both_sources(self):
        p,r,chat=self.setup({'sufficient':True,'missing_aspect':'','target_source':'both','supplementary_query':''})
        p.items=p.items[:3]  # Only statutes; absence of cases does not force a search.
        result,t=self.run_check(p,r,chat);self.assertIs(result,p);self.assertEqual(t['status'],'sufficient')
        r.initial_candidates.assert_not_called();r.rerank_provider.rerank.assert_not_called()

    def test_both_caps_twenty_and_deduplicates_old_parents(self):
        cs=[chunk('old0','statute',i) for i in range(15)]+[chunk('old3','case',i) for i in range(15)]
        p,r,chat=self.setup({'sufficient':False,'missing_aspect':'x','target_source':'both','supplementary_query':'one query'},cs)
        result,t=self.run_check(p,r,chat);self.assertIs(result,p)
        self.assertTrue(t['triggered_search']);self.assertFalse(t['replaced'])
        self.assertEqual(t['supplementary_child_count'],20);self.assertEqual(t['supplementary_child_counts'],{'statute':10,'case':10})
        args=r.initial_candidates.call_args.args[0];self.assertEqual(args.top_k,10)
        self.assertEqual(args.retrieval_targets,['statute_db','case_db']);self.assertFalse(args.enable_coverage_check)
        self.assertEqual(t['new_candidate_parent_ids'],[]);r.rerank_provider.rerank.assert_not_called()

    def test_uniform_original_question_and_same_source_minimum(self):
        p,r,chat=self.setup(scores=[.9,.1,.3,.01,.5,.8]);result,t=self.run_check(p,r,chat)
        self.assertTrue(t['replaced']);self.assertEqual(t['comparison']['old_parent_id'],'old1')
        self.assertNotIn('old1',[e.parent_id for e in result.items]);self.assertIn('old3',[e.parent_id for e in result.items])
        self.assertEqual(len(result.items),5);self.assertEqual([e.evidence_id for e in result.items],[f'ev_{i:03d}' for i in range(1,6)])
        call=r.rerank_provider.rerank.call_args
        self.assertEqual(call.args[0],'original question');self.assertEqual(call.kwargs['top_n'],6)
        self.assertEqual(len(call.args[1]),6);self.assertNotIn('父块排序分',call.args[1][0])
        self.assertEqual(r.rerank_provider.rerank.call_count,1);self.assertFalse(t['gap_resolution_verified'])

    def test_equal_or_lower_new_score_keeps_original_exactly(self):
        for score in [.1,.09]:
            p,r,chat=self.setup(scores=[.9,.1,.3,.01,.5,score]);before=to_dict(p)
            result,t=self.run_check(p,r,chat);self.assertIs(result,p);self.assertEqual(to_dict(result),before)
            self.assertTrue(t['triggered_search']);self.assertFalse(t['replaced'])

    def test_global_minimum_only_when_same_source_absent(self):
        p,r,chat=self.setup({'sufficient':False,'missing_aspect':'x','target_source':'case','supplementary_query':'q'},
                            chunks=[chunk('new-case','case')],scores=[.9,.1,.3,.4,.5,.2])
        patch('rag.coverage.local_source_index',return_value={**{f'old{i}':'statute' for i in range(5)},'new-case':'case'}).start()
        result,t=self.run_check(p,r,chat);self.assertTrue(t['replaced']);self.assertFalse(t['comparison']['same_source'])
        self.assertEqual(t['comparison']['old_parent_id'],'old1')

    def test_rerank_failure_and_novel_parent_budget(self):
        cs=[chunk('n'+str(i),number=0) for i in range(12)]
        p,r,chat=self.setup(chunks=cs);r.rerank_provider.rerank.side_effect=RuntimeError('fixture outage')
        result,t=self.run_check(p,r,chat);self.assertIs(result,p);self.assertEqual(t['status'],'rerank_failed')
        self.assertLessEqual(len(t['new_candidates']),5);self.assertLessEqual(t['unified_rerank_input_count'],10)

    def test_online_default_dense_and_optional_modes_preserved(self):
        u=QueryUnderstandingResult(intents=['法律咨询'])
        plan=route_skills('q',u);self.assertEqual(plan.retrieval_mode,'dense');self.assertFalse(plan.enable_coverage_check)
        self.assertFalse(RetrievalPlan('q').enable_coverage_check)
        self.assertTrue(route_skills('q',u,enable_coverage_check=True).enable_coverage_check)
        self.assertEqual(route_skills('q',u,retrieval_mode='hybrid').retrieval_mode,'hybrid')
        self.assertEqual(route_skills('q',u,retrieval_mode='dense_bm25').retrieval_mode,'dense_bm25')

    def test_novel_children_aggregate_to_unique_parents(self):
        cs=[chunk('new','statute',i) for i in range(4)]
        p,r,chat=self.setup(chunks=cs)
        result,t=self.run_check(p,r,chat)
        self.assertEqual(t['new_candidate_parent_ids'],['new'])
        self.assertEqual(r.rerank_provider.rerank.call_args.kwargs['top_n'],6)
        self.assertEqual(len({e.parent_id for e in result.items}),5)

    def test_invalid_unified_results_preserve_pack_and_report_failure(self):
        p,r,chat=self.setup();r.rerank_provider.rerank.return_value=[{'index':0,'relevance_score':1.}]*6
        result,t=self.run_check(p,r,chat)
        self.assertIs(result,p);self.assertEqual(t['status'],'rerank_failed');self.assertFalse(t['replaced'])


class CoverageRequestSwitch(unittest.TestCase):
    def test_both_endpoints_default_off_and_explicit_opt_in(self):
        import app as web
        final=FinalAnswer('switch-test','controlled fixture')
        for endpoint in ('/api/chat','/api/chat/stream'):
            for supplied in ({},{'enable_coverage_check':False},{'enable_coverage_check':True}):
                with self.subTest(endpoint=endpoint,supplied=supplied), \
                     patch('app.run_agentic_rag',return_value=final) as direct, \
                     patch('app.AgenticRagWorkflow') as factory:
                    factory.return_value.run.return_value=final
                    response=web.app.test_client().post(endpoint,json={'message':'test',**supplied},buffered=True)
                    self.assertEqual(response.status_code,200)
                    call=direct if endpoint=='/api/chat' else factory.return_value.run
                    self.assertIs(call.call_args.kwargs['enable_coverage_check'],supplied.get('enable_coverage_check',False))

    def test_ambiguous_values_never_start_a_workflow(self):
        import app as web
        for endpoint in ('/api/chat','/api/chat/stream'):
            for value in ('false','true',1,None):
                with self.subTest(endpoint=endpoint,value=value), \
                     patch('app.run_agentic_rag') as direct,patch('app.AgenticRagWorkflow') as factory:
                    response=web.app.test_client().post(endpoint,json={'message':'test','enable_coverage_check':value},buffered=True)
                    self.assertEqual(response.status_code,400)
                    direct.assert_not_called();factory.assert_not_called()


if __name__=='__main__':unittest.main()
