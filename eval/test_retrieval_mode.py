"""Offline request and routing checks for the optional retrieval modes."""
import unittest
from unittest.mock import patch

from eval_retrieval import ROOT_DIR
from rag.schemas import FinalAnswer, QueryUnderstandingResult
from rag.skill_router import RETRIEVAL_MODES, route_skills


class RetrievalModeSelection(unittest.TestCase):
    def test_router_preserves_modes_quotas_and_independent_coverage_flag(self):
        u = QueryUnderstandingResult(intents=['法律咨询'])
        baseline = route_skills('q', u)
        self.assertEqual(baseline.retrieval_mode, 'dense')
        for mode in RETRIEVAL_MODES:
            for enabled in (False, True):
                plan = route_skills('q', u, retrieval_mode=mode, enable_coverage_check=enabled)
                self.assertEqual(plan.retrieval_mode, mode)
                self.assertIs(plan.enable_coverage_check, enabled)
                self.assertEqual(plan.evidence_quotas, baseline.evidence_quotas)
                self.assertEqual(plan.rerank_quotas, baseline.rerank_quotas)

    def test_both_endpoints_forward_mode_with_independent_coverage(self):
        import app as web
        final = FinalAnswer('mode-test', 'controlled fixture')
        for endpoint in ('/api/chat', '/api/chat/stream'):
            for mode in (None, *RETRIEVAL_MODES):
                for enabled in (False, True):
                    with self.subTest(endpoint=endpoint, mode=mode, coverage=enabled), \
                         patch('app.run_agentic_rag', return_value=final) as direct, \
                         patch('app.AgenticRagWorkflow') as factory:
                        factory.return_value.run.return_value = final
                        payload = {'message':'test', 'enable_coverage_check':enabled}
                        if mode is not None:payload['retrieval_mode'] = mode
                        response = web.app.test_client().post(endpoint, json=payload, buffered=True)
                        self.assertEqual(response.status_code, 200)
                        call = direct if endpoint == '/api/chat' else factory.return_value.run
                        self.assertEqual(call.call_args.kwargs['retrieval_mode'], mode or 'dense')
                        self.assertIs(call.call_args.kwargs['enable_coverage_check'], enabled)

    def test_invalid_modes_rejected_before_any_workflow(self):
        import app as web
        for mode in ('unknown', '', 'HYBRID', None, True, [], {}):
            with self.assertRaises(ValueError):route_skills('q', QueryUnderstandingResult(), retrieval_mode=mode)
            for endpoint in ('/api/chat', '/api/chat/stream'):
                with self.subTest(endpoint=endpoint, mode=mode), \
                     patch('app.run_agentic_rag') as direct, patch('app.AgenticRagWorkflow') as factory:
                    response = web.app.test_client().post(endpoint, json={'message':'test', 'retrieval_mode':mode}, buffered=True)
                    self.assertEqual(response.status_code, 400)
                    direct.assert_not_called()
                    factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
