"""Browser/SSE replay of saved real C1 responses, with live MySQL and no HTTP API calls.

Understanding, answer and safety are controlled fixtures, NOT a new answer test.
Optional dependency: playwright; uses installed Windows Edge.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval_retrieval import ROOT_DIR, AppConfig
from rag.providers import Providers
from rag.retrieval import Retriever
from rag.schemas import QueryUnderstandingResult, FollowupResult, SafetyReview
from rag.workflow import AgenticRagWorkflow
from rag.lexical import corpus_fingerprint
from run_hybrid_m123 import signature, rerank_payload
import app as web

OUT = ROOT_DIR / 'data/coverage_ui'
IMAGES = ROOT_DIR / 'Refinement and Optimization/images'
SAVED = ROOT_DIR / 'eval/results_coverage_20'
BASELINE = ROOT_DIR / 'eval/results_hybrid_20'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def cached(folder, payload):
    key = signature(payload)
    record = read(folder / (key + '.json'))
    assert record['signature'] == key
    return record['value']

def fixture(qid, enabled=True):
    config = AppConfig.load()
    actual = Providers(config)
    baseline = next(r for r in read(BASELINE/'M1.json')['details'] if r['query_id'] == qid)
    expected = next(r for r in read(SAVED/'C1.json')['details'] if r['query_id'] == qid)
    diagnosis_calls = []
    class Embedding:
        def embed(self, query):
            if query == baseline['query']:
                return read(ROOT_DIR / f'eval/results_ablation_20/cache/{qid}_embedding.json')['value']
            return cached(SAVED/'cache/embedding', {'query':query,'dimension':actual.embedding.dimension,
                'model':actual.embedding.model,'endpoint':actual.embedding.base_url+'/embeddings'})
    class Search:
        def search(self, **kwargs):
            if kwargs['limit'] == 30:
                target = 'case_db' if kwargs['collection'] == config.section('qdrant')['cases_collection_name'] else 'statute_db'
                return [{'id':c['chunk_id'],'score':c['dense_score'],'payload':c['payload']}
                    for c in baseline['recall_chunks'] if c['payload']['_retrieval_target'] == target]
            return cached(SAVED/'cache/qdrant_search', {'vector_sha256':signature(kwargs['vector']),
                'limit':kwargs['limit'],'collection':kwargs['collection'],
                'corpus_fingerprint':corpus_fingerprint()[0],'endpoint':actual.qdrant.url})
    class Rerank:
        def rerank(self, query, documents, top_n):
            folder = BASELINE/'cache/rerank' if top_n > 10 else SAVED/'cache/unified_rerank'
            return cached(folder, rerank_payload(actual.rerank,query,documents,top_n))
    class Chat:
        def chat(self, messages, temperature):
            diagnosis_calls.append(True)
            return cached(SAVED/'cache/diagnosis', {'messages':messages,'temperature':temperature,
                'model':actual.chat.model,'endpoint':actual.chat.base_url+'/chat/completions'})
    providers = SimpleNamespace(chat=Chat(),embedding=Embedding(),qdrant=Search(),rerank=Rerank())
    workflow = AgenticRagWorkflow(config=config, providers=providers)
    workflow.retriever = Retriever(config, providers.embedding, providers.qdrant, providers.rerank)
    understanding = QueryUnderstandingResult(intents=[baseline['query_type']],rewritten_query=baseline['query'])
    with patch('app.AgenticRagWorkflow',return_value=workflow), \
         patch('rag.workflow.understand_query',return_value=understanding), \
         patch('rag.workflow.decide_followup',return_value=FollowupResult()), \
         patch('rag.workflow.stream_generate_answer',return_value=(
             'UI test: saved real C1 retrieval and diagnosis replay. No new legal answer generated.','')), \
         patch('rag.workflow.llm_safety_check',return_value=SafetyReview()), \
         patch('rag.providers.HttpJsonClient.request',side_effect=AssertionError('External API forbidden')), \
         patch.object(workflow.retriever.parents,'_get_many_local',side_effect=AssertionError('MySQL required')) as fallback:
        request_payload = {'message':baseline['query']}
        if enabled:request_payload['enable_coverage_check'] = True
        response = web.app.test_client().post('/api/chat/stream',json=request_payload,buffered=True)
        body = response.get_data(as_text=True)
        assert response.status_code == 200 and fallback.call_count == 0
    events = []
    for block in body.strip().split('\n\n'):
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith('event: '))
        value = json.loads(next(line[6:] for line in lines if line.startswith('data: ')))
        assert name != 'error', 'SSE error during cached replay'
        events.append((name,value))
    final = next(value for name,value in events if name == 'final')
    assert final['debug']['evidence_pack'] == (expected if enabled else baseline)['evidence_pack']
    assert final['debug']['coverage_check']['status'] == (expected['coverage_check']['status'] if enabled else 'gate_skipped')
    if not enabled:
        assert not diagnosis_calls
        assert final['debug']['coverage_check']['reason'] == 'disabled'
        assert final['debug']['coverage_check']['triggered_search'] is False
    progress = [value['payload'] for name,value in events if name == 'step' and value['step'] == 'coverage_check']
    if qid == 'q024' and enabled:
        assert [v['status'] for v in progress] == ['checking','searching','reranking','searched_replaced']
        assert progress[-1]['child_count'] == 10 and progress[-1]['new_parent_count'] == 5
        assert progress[-1]['comparison_count'] == 10
    visible = [(name,value) for name,value in events if not (name == 'step' and value['step'] == 'step_status'
        and value['payload']['step'] in ('answer_generation','safety_review'))]
    return {'query':baseline['query'],'status':progress[-1]['status'],'coverage_progress':progress,'enabled':enabled,
        'stream':''.join(web.sse_event(name,value) for name,value in visible)}

def run():
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True,exist_ok=True)
    IMAGES.mkdir(parents=True,exist_ok=True)
    fixtures = {qid:fixture(qid) for qid in ('q024','q004','q060')}
    default_off = fixture('q024',enabled=False)
    errors = []
    checks = []
    files = {'q024':'coverage_replaced.png','q004':'coverage_sufficient.png','q060':'coverage_gate_skipped.png'}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1200},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://127.0.0.1:5000')
        assert not page.locator('#coverage-toggle').is_checked()
        def off_response(route, request):
            assert request.post_data_json['enable_coverage_check'] is False
            route.fulfill(status=200,content_type='text/event-stream',body=default_off['stream'])
        page.route('**/api/chat/stream',off_response)
        page.locator('#message-input').fill(default_off['query'])
        page.locator('#send-button').click()
        page.wait_for_selector('.coverage-card[data-status="gate_skipped"]')
        assert page.locator('[data-coverage-stage="gate"] .coverage-stage-value').inner_text() == 'Switch off'
        assert '0 diagnosis calls' in page.locator('.coverage-outcome').inner_text()
        page.locator('.coverage-card').scroll_into_view_if_needed()
        page.screenshot(path=str(IMAGES/'coverage_default_off.png'),full_page=True)
        page.unroute('**/api/chat/stream')
        checks.append('default unchecked, request false, real workflow retains exact C0 evidence with zero diagnosis calls')
        for qid, record in fixtures.items():
            page.goto('http://127.0.0.1:5000')
            assert not page.locator('#coverage-toggle').is_checked()
            page.locator('#coverage-toggle').check()
            def on_response(route, request, record=record):
                assert request.post_data_json['enable_coverage_check'] is True
                route.fulfill(status=200,content_type='text/event-stream',body=record['stream'])
            page.route('**/api/chat/stream',on_response)
            page.locator('#message-input').fill(record['query'])
            page.locator('#send-button').click()
            page.wait_for_selector(f'.coverage-card[data-status="{record["status"]}"]')
            assert page.locator('.coverage-card').count() == 1
            assert page.locator('[data-route-stage="bm25"] .route-value').inner_text() == 'Not enabled'
            if qid == 'q024':
                assert page.locator('.coverage-score-new strong').inner_text() == '0.7228'
                assert page.locator('.coverage-score-old strong').inner_text() == '0.6594'
                assert 'gap not rechecked' in page.locator('.coverage-outcome').inner_text()
            if qid == 'q060':assert '0 diagnosis calls' in page.locator('.coverage-outcome').inner_text()
            page.locator('.coverage-card').scroll_into_view_if_needed()
            page.screenshot(path=str(IMAGES/files[qid]),full_page=True)
            checks.append(qid+': actual workflow SSE, saved real C1 response, live MySQL, browser rendered '+record['status'])
            page.unroute('**/api/chat/stream')
        page.locator('#new-session').click()
        assert not page.locator('#coverage-toggle').is_checked()
        checks.append('reload and new session reset coverage switch to off')
        # Failure is a controlled boundary check, not a reported experimental case.
        page.evaluate("clearSteps(); appendStep('coverage_check','Evidence coverage',{status:'check_failed'})")
        assert page.locator('.coverage-badge').inner_text() == 'Check failed'
        assert 'initial evidence retained' in page.locator('.coverage-outcome').inner_text()
        checks.append('controlled failure remains explicit and retains original evidence')
        page.set_viewport_size({'width':390,'height':844})
        for progress in fixtures['q024']['coverage_progress']:
            page.evaluate("payload => updateCoverageCard(payload, 'Evidence coverage')", progress)
            assert page.locator('.coverage-card').get_attribute('data-status') == progress['status']
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        checks.append('actual q024 progress renders all four states; 390px viewport has no horizontal overflow')
        browser.close()
    assert not errors, errors
    result = {'checks':checks,'screenshots':['coverage_default_off.png',*files.values()],'browser_errors':errors,
        'new_model_calls':0,'qdrant_requests':0,'mysql_json_fallback_calls':0,
        'note':'Saved real C1 API responses replayed through production coverage/workflow/SSE and live MySQL. Understanding, answer, safety controlled; no new legal answer verified.'}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__ == '__main__':
    run()
