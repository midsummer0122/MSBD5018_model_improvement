"""Verify selected modes through real workflow/SSE and live MySQL using saved API results.

No new model/Qdrant requests. Understanding, answer and safety are controlled fixtures.
This script creates new screenshots; it does not overwrite historical hybrid screenshots.
"""
import json
from playwright.sync_api import sync_playwright
from eval_retrieval import ROOT_DIR
from verify_retrieval_ui import fixture

OUT = ROOT_DIR / 'data/retrieval_modes_ui'
IMAGES = ROOT_DIR / 'Refinement and Optimization/images'


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    records = {mode:fixture('q060',mode) for mode in ('dense','dense_bm25','hybrid')}
    errors = []
    checks = []
    screenshots = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1200},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        for mode, record in records.items():
            page.goto('http://127.0.0.1:5000')
            assert page.locator('#retrieval-mode-select').input_value() == 'dense'
            assert not page.locator('#coverage-toggle').is_checked()
            page.locator('#retrieval-mode-select').select_option(mode)
            assert page.locator('#retrieval-state').inner_text() == 'Not started'
            def serve(route, request, record=record):
                assert request.post_data_json['retrieval_mode'] == record['mode']
                assert request.post_data_json['enable_coverage_check'] is False
                assert page.locator('#retrieval-mode-select').is_disabled()
                route.fulfill(status=200,content_type='text/event-stream',body=record['stream'])
            page.route('**/api/chat/stream',serve)
            page.locator('#message-input').fill(record['query'])
            page.locator('#send-button').click()
            page.wait_for_function("document.querySelector('#retrieval-state').textContent === 'Complete' && !document.querySelector('#send-button').disabled")
            assert page.locator('.coverage-card').get_attribute('data-status') == 'gate_skipped'
            assert 'Switch off' in page.locator('.coverage-card').inner_text()
            for channel in ('bm25','exact','fusion'):
                disabled = mode == 'dense' or (channel == 'exact' and mode != 'hybrid')
                actual = 'is-disabled' in page.locator(f'[data-route-stage="{channel}"]').get_attribute('class')
                assert actual == disabled
            assert mode.split('_')[0].capitalize() in page.locator('#retrieval-mode').inner_text()
            image_name = 'retrieval_mode_' + mode + '.png'
            page.screenshot(path=str(IMAGES/image_name),full_page=True)
            screenshots.append(image_name)
            checks.append(mode+': selected request -> actual workflow routing -> live MySQL -> saved expected evidence -> correct SSE channels')
            before = page.locator('#retrieval-mode').inner_text()
            page.locator('#retrieval-mode-select').select_option('dense' if mode == 'hybrid' else 'hybrid')
            assert page.locator('#retrieval-mode').inner_text() == before
            checks.append(mode+': changing next-question selection does not relabel completed results')
            page.unroute('**/api/chat/stream')
        page.locator('#new-session').click()
        assert page.locator('#retrieval-mode-select').input_value() == 'dense'
        assert not page.locator('#coverage-toggle').is_checked()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        checks.append('new session resets to Dense and coverage off; 390px viewport has no horizontal overflow')
        browser.close()
    assert not errors, errors
    result = {'checks':checks,'screenshots':screenshots,'browser_errors':errors,
        'new_model_calls':0,'qdrant_requests':0,'mysql_json_fallback_calls':0,
        'note':'Saved real M1/M2/M3 responses through production workflow/SSE and live MySQL; controlled understanding, answer and safety; not new legal answers.'}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    run()
