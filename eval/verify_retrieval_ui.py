"""Verify the real SSE wiring and browser UI using saved real retrieval scores.

No new model or Qdrant calls. MySQL is read with JSON fallback prohibited.
Question understanding/answer/safety are controlled fixtures ONLY in this test.
Optional browser dependency: pip install playwright; uses installed Windows Edge.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from unittest.mock import patch

from eval_retrieval import ROOT_DIR,AppConfig
from rag.providers import Providers
from rag.retrieval import Retriever
from rag.schemas import QueryUnderstandingResult,FollowupResult,SafetyReview
from rag.workflow import AgenticRagWorkflow
from run_hybrid_m123 import rerank_payload,signature
import app as web

OUT=ROOT_DIR/'data/ui_trace'
IMAGES=ROOT_DIR/'Refinement and Optimization/images'
SAVED=ROOT_DIR/'eval/results_hybrid_20'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def fixture(qid, mode='hybrid'):
    config=AppConfig.load();providers=Providers(config)
    m1=next(r for r in read(SAVED/'M1.json')['details'] if r['query_id']==qid)
    result_file={'dense':'M1.json','dense_bm25':'M2.json','hybrid':'M3.json'}[mode]
    expected=next(r for r in read(SAVED/result_file)['details'] if r['query_id']==qid)
    class SavedEmbedding:
        def embed(self,query):
            assert query==m1['query']
            return read(ROOT_DIR/f'eval/results_ablation_20/cache/{qid}_embedding.json')['value']
    class SavedSearch:
        def search(self,**kwargs):
            target='case_db' if kwargs['collection']==config.section('qdrant')['cases_collection_name'] else 'statute_db'
            return [{'id':c['chunk_id'],'score':c['dense_score'],'payload':c['payload']}
                    for c in m1['recall_chunks'] if c['payload']['_retrieval_target']==target]
    class SavedRerank:
        def rerank(self,query,documents,top_n):
            payload=rerank_payload(providers.rerank,query,documents,top_n)
            cached=read(SAVED/'cache/rerank'/(signature(payload)+'.json'))
            assert cached['signature']==signature(payload)
            return cached['value']
    workflow=AgenticRagWorkflow(config=config,providers=providers)
    workflow.retriever=Retriever(config,SavedEmbedding(),SavedSearch(),SavedRerank())
    understanding=QueryUnderstandingResult(intents=[m1['query_type']],query_type='knowledge_question',rewritten_query=m1['query'])
    # Real workflow -> app worker -> compact_payload -> SSE, with no external HTTP permitted.
    with patch('app.AgenticRagWorkflow',return_value=workflow), \
         patch('rag.workflow.understand_query',return_value=understanding), \
         patch('rag.workflow.decide_followup',return_value=FollowupResult()), \
         patch('rag.workflow.stream_generate_answer',return_value=('UI preview: saved retrieval replay. No new answer was generated.','')), \
         patch('rag.workflow.llm_safety_check',return_value=SafetyReview()), \
         patch('rag.providers.HttpJsonClient.request',side_effect=AssertionError('External HTTP forbidden in UI verification')), \
         patch.object(workflow.retriever.parents,'_get_many_local',side_effect=AssertionError('MySQL required')) as fallback:
        response=web.app.test_client().post('/api/chat/stream',json={'message':m1['query'],
            'retrieval_mode':mode,'enable_coverage_check':False},buffered=True)
        body=response.get_data(as_text=True)
        assert fallback.call_count==0
    events=[]
    for block in body.strip().split('\n\n'):
        lines=block.splitlines();name=next(line[7:] for line in lines if line.startswith('event: '))
        data=json.loads(next(line[6:] for line in lines if line.startswith('data: ')))
        assert name!='error'
        events.append((name,data))
    progress=[d['payload'] for name,d in events if name=='step' and d['step']=='retrieval_progress']
    assert progress[0]['stage']=='plan' and progress[-1]['stage']=='evidence'
    assert progress[-1]['count']==len(expected['evidence_pack']['items'])==5
    final=next(d for name,d in events if name=='final')
    assert final['debug']['retrieval_plan']['retrieval_mode']==mode
    assert final['debug']['coverage_check']['status']=='gate_skipped'
    assert final['debug']['evidence_pack']==expected['evidence_pack']
    # The preview ends at retrieval; omit the controlled answer/safety progress indicators.
    visible=[]
    for name,data in events:
        if name=='step' and data['step']=='step_status' and data['payload']['step'] in ('answer_generation','safety_review'):continue
        visible.append((name,data))
    stream=''.join(web.sse_event(name,data) for name,data in visible)
    return {'query':m1['query'],'mode':mode,'stream':stream,'progress':progress,'mysql_json_fallback_calls':0,'new_model_calls':0}

def run():
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    fixtures={qid:fixture(qid) for qid in ('q060','q004')}
    (OUT/'sse_fixtures.json').write_text(json.dumps(fixtures,ensure_ascii=False,indent=2),encoding='utf-8')
    edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    checks=[];errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(edge),headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.goto('http://127.0.0.1:5000');page.wait_for_selector('#retrieval-map')
        assert page.locator('#retrieval-state').inner_text()=='Not started'
        checks.append('initial diagram is planned, with no completed stages')
        page.screenshot(path=str(IMAGES/'hybrid_retrieval_idle.png'),full_page=True)
        for qid,record in fixtures.items():
            page.reload();page.wait_for_selector('#retrieval-map')
            page.locator('#retrieval-mode-select').select_option('hybrid')
            page.route('**/api/chat/stream',lambda route,request,record=record:route.fulfill(status=200,content_type='text/event-stream',body=record['stream']))
            page.locator('#message-input').fill(record['query']);page.locator('#send-button').click()
            page.wait_for_function("document.querySelector('#retrieval-state').textContent === 'Complete'")
            assert page.locator('[data-route-stage="dense"] .route-value').inner_text()=='Law 30 · Cases 30'
            assert page.locator('[data-route-stage="fusion"] .route-value').inner_text()=='Law 30 · Cases 30'
            if qid=='q060':
                assert 'high' in page.locator('[data-route-stage="exact"] .route-value').inner_text()
                page.screenshot(path=str(IMAGES/'hybrid_retrieval_complete.png'),full_page=True)
            else:assert 'No exact matches' in page.locator('#retrieval-note').inner_text()
            checks.append(qid+': production SSE counts displayed; cached evidence IDs unchanged')
            page.evaluate('interruptRetrievalMap()')
            assert page.locator('#retrieval-state').inner_text()=='Complete'
            page.unroute('**/api/chat/stream')
        page.evaluate("clearSteps(); appendStep('followup_result','Follow-up Decision',{need_followup:true,followup_questions:['Which city?']})")
        assert page.locator('#retrieval-state').inner_text()=='Awaiting details'
        assert page.locator('.route-node.is-done').count()==0
        checks.append('follow-up does not claim retrieval ran')
        page.evaluate("updateRetrievalMap({stage:'plan',mode:'hybrid',targets:['statute_db','case_db'],rerank_k:20,final_k:5});updateRetrievalMap({stage:'parents',status:'started'});interruptRetrievalMap()")
        assert page.locator('#retrieval-state').inner_text()=='Interrupted'
        assert page.locator('.route-node.is-running').count()==0
        checks.append('interrupted retrieval is marked; later answer errors preserve completed retrieval')
        page.evaluate("updateRetrievalMap({stage:'plan',status:'planned',mode:'dense',targets:['statute_db'],rerank_k:20,final_k:5})")
        assert page.locator('[data-route-stage="exact"]').get_attribute('class').find('is-disabled')>=0
        assert page.locator('[data-route-stage="fusion"] .route-value').inner_text()=='Not enabled'
        page.evaluate("updateRetrievalMap({stage:'rerank',status:'fallback',count:20,statute_count:20,case_count:0});updateRetrievalMap({stage:'evidence',status:'completed',count:5,statute_count:5,case_count:0})")
        assert page.locator('#retrieval-state').inner_text()=='Used fallback'
        checks.append('dense mode disables unused channels; fallback is visible')
        page.set_viewport_size({'width':390,'height':844});page.reload()
        page.locator('#retrieval-mode-select').select_option('hybrid')
        page.route('**/api/chat/stream',lambda route:route.fulfill(status=200,content_type='text/event-stream',body=fixtures['q060']['stream']))
        page.locator('#message-input').fill(fixtures['q060']['query']);page.locator('#send-button').click()
        page.wait_for_function("document.querySelector('#retrieval-state').textContent === 'Complete'")
        box=page.locator('#retrieval-map').bounding_box();assert box['x']>=0 and box['x']+box['width']<=391
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.locator('#retrieval-map').scroll_into_view_if_needed()
        page.screenshot(path=str(IMAGES/'hybrid_retrieval_mobile.png'),full_page=True)
        checks.append('mobile 390px has no horizontal overflow')
        browser.close()
    if errors:raise ValueError('Browser JavaScript errors: '+str(errors))
    result={'checks':checks,'browser_errors':errors,'new_model_calls':0,'qdrant_requests':0,
            'mysql_json_fallback_calls':0,'note':'Production SSE transport and live MySQL; saved real candidate/rerank records; no answer-generation validation.'}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    try:run()
    except Exception as exc:print('ui_verification_failed: '+type(exc).__name__);raise
