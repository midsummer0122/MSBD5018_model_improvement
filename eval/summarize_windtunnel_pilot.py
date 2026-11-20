"""Summarize completed paired checkpoints after the user reduced the benchmark scope. No API calls."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from eval_retrieval import ROOT_DIR, build_indexes, save_result, summarize, load_queries, validate_labels
from analyze_windtunnel import metric, outcome, trace_missing, quota_replay


def main():
    out=ROOT_DIR/'eval/results_windtunnel'
    versions=('A','B','B_fair')
    raw={v:json.loads((out/f'version_{v}.json').read_text(encoding='utf-8')) for v in versions}
    ids=set.intersection(*({r['query_id'] for r in raw[v]['details']} for v in versions))
    rows={v:[r for r in raw[v]['details'] if r['query_id'] in ids] for v in versions}
    n=len(ids)
    assert n and all([r['query_id'] for r in rows[v]]==sorted(ids) for v in versions)
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    validate_labels(load_queries(ROOT_DIR/'eval/queries_windtunnel_60.jsonl'),index)
    manifest=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    for path,sha in manifest['protected_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    stats={v:summarize(rows[v]) for v in versions}
    quota={v:summarize([quota_replay(r,index) for r in rows[v]]) for v in ('B','B_fair')}
    paired=[]
    for a,b,fair in zip(rows['A'],rows['B'],rows['B_fair']):
        paired.append({'query_id':a['query_id'],'query':a['query'],'split':a['split'],
            'all_parent_outcome':outcome(metric(a),metric(b)),
            'statute_parent_outcome':outcome(metric(a,'statute_gt'),metric(b,'statute_gt')),
            'A_all_parent_recall5':metric(a),'B_all_parent_recall5':metric(b),'B_fair_all_parent_recall5':metric(fair),
            'A_statute_parent_recall5':metric(a,'statute_gt'),'B_statute_parent_recall5':metric(b,'statute_gt'),
            'B_missing_trace':trace_missing(b,index)})
    for v in versions:
        events=[e for e in raw[v]['api_events'] if e.get('query_id') in ids]
        save_result(out/f'pilot_version_{v}.json',rows[v],{**raw[v]['metadata'],
            'scope':'user-reduced exploratory pilot','paired_queries':n,'original_planned_queries':60},events)
    causes={v:dict(Counter(t['cause'] for r in rows[v] for t in trace_missing(r,index))) for v in versions}
    statute_causes={v:dict(Counter(t['cause'] for r in rows[v] for t in trace_missing(r,index)
                                  if t['doc_type']=='statute')) for v in versions}
    summary={'scope':'Incomplete full benchmark; completed common subset only',
        'paired_queries':n,'query_ids':sorted(ids),'query_types':dict(Counter(r['query_type'] for r in rows['A'])),
        'splits':dict(Counter(r['split'] for r in rows['A'])),
        'case_labeled_queries':sum(bool(r['relevant_case_doc_ids']) for r in rows['A']),
        'raw_completed_groups':{v:len(raw[v]['details']) for v in versions},'versions':stats,'quota_replay':quota,
        'paired_outcomes':{key:dict(Counter(r[key] for r in paired)) for key in ('all_parent_outcome','statute_parent_outcome')},
        'loss_causes':causes,'statute_loss_causes':statute_causes,'paired_details':paired,
        'protected_files_unchanged':True,'extra_model_calls_for_summary':0}
    (out/'pilot_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest.update(status='stopped_after_scope_reduction',stopped_at_utc=datetime.now(timezone.utc).isoformat(),
        reason='User requested a quick exploratory check instead of the full 60-query benchmark',
        completed={v:len(raw[v]['details']) for v in versions},paired_queries_for_pilot=n,
        protected_files_unchanged=True,full_60_query_benchmark_complete=False,
        postflight_remote_snapshot_performed=False)
    (out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    def m(v,scope='all_gt',grain='parent',name='recall@5'):
        return stats[v]['scoped_metrics'][scope][grain]['evidence'].get(name,0)
    lines=['# 单法条库 vs 法条＋案例双库：小样本风洞粗测报告','',
        '日期：2026-11-20。状态：按用户后续要求停止完整版，仅汇总已经完成的共同样本。','',
        '## 1. 先看结论','',
        '这批个案咨询题中，案例库能够补充相关依据，但当前“两个库一起召回、统一重排、只保留5条父块证据”的方案，会明显减少最终保留的法条。直接扩大候选量没有在这批题里带来更好的最终覆盖。','',
        f'**这不是完整60题的正式评测。** 三组共同完成 {n} 题，只比较这个交集；A 额外完成的题目不混入均值。以下比例表示对预先标注依据的覆盖率，不是法律回答准确率。','',
        '## 2. 本次实际测了什么','',
        '- 语料：128份中央法规类资料＋174份劳动案例；511＋349个父块，1,384＋2,334个子块。',
        '- 已制作60题、目录清单和原文校验记录，按完整方案启动；用户希望只做粗测后，停止剩余请求。',
        f"- 实际配对样本：{n}题，全部为法律咨询；其中{summary['case_labeled_queries']}题有案例真值，其余为纯法条核心题；split分布为{summary['splits']}。样本来自固定顺序的前段，并非均衡抽样，不能代表法条解释、程序咨询和案例检索四种意图的整体表现。",
        '- 标注先于检索：60题按真实原文设计，101处摘录及ID通过检查；没有用向量排名作为正确答案。',
        '- 三组共享同一题的一个Query Embedding，各自真实执行搜索、父块回补和Rerank。不调用DeepSeek或生成答案。',
        '- 运行时检测真实provider与重排分数，没有把Mock或Rerank异常回退计为有效结果。','',
        '| 组别 | 查询目标 | 每库候选 | 总候选 | 重排子块 | 最终父块证据 |',
        '|---|---|---:|---:|---:|---:|',
        '| A | 法条 | 30 | 30 | 20 | 5 |','| B | 法条＋案例 | 30 | 60 | 20 | 5 |',
        '| B-Fair | 法条＋案例 | 15 | 30 | 20 | 5 |','',
        'B-Fair与A的总候选数相同，但法条候选也从30减到15，不能把两者差异完全归因于案例内容。评测复用实际集合映射、父块回补、重排和证据归组逻辑，没有修改线上策略。','',
        '## 3. 最终证据覆盖结果','',
        '| Evidence@5 指标 | A | B | B-Fair |','|---|---:|---:|---:|']
    for label,scope,grain,name in [('法条文档 Recall@5','statute_gt','doc','recall@5'),
            ('法条父块 Recall@5','statute_gt','parent','recall@5'),
            ('法条父块 MRR','statute_gt','parent','mrr'),
            ('法条＋案例文档 Recall@5','all_gt','doc','recall@5'),
            ('法条＋案例父块 Recall@5','all_gt','parent','recall@5'),
            ('法条＋案例父块 MRR','all_gt','parent','mrr')]:
        lines.append('| '+label+' | '+' | '.join(f'{m(v,scope,grain,name):.4f}' for v in versions)+' |')
    lines+=['','“文档”看是否找到正确的法规或案例；“父块”进一步看是否找到标注规则所在的段落。计算法条指标时，案例仍占据结果位置，因而能反映挤占现象。空案例真值不计零分。','',
        '### 3.1 平均证据包结构与耗时','',
        '| 组别 | 平均法条父块 | 平均案例父块 | 平均检索耗时/秒 |','|---|---:|---:|---:|']
    for v in versions:
        s=stats[v];lines.append(f"| {v} | {s['avg_statute_evidence_count']:.2f} | {s['avg_case_evidence_count']:.2f} | {s['mean_retrieval_seconds']:.2f} |")
    lines+=['','耗时不含共享Embedding、初次本地索引读取及组间等待。每组约4秒用于尝试不可用MySQL后回退本地JSON，Rerank约7—8秒，是本次运行较慢的主要原因。完整方案需要180组运行；用户缩小范围后已停止，不再追加模型调用。','',
        '### 3.2 逐题胜负（B相对A）','',
        '| 口径 | Win | Tie | Loss |','|---|---:|---:|---:|']
    for key,label in [('all_parent_outcome','综合父块Recall@5'),('statute_parent_outcome','法条父块Recall@5')]:
        counts=summary['paired_outcomes'][key]
        lines.append('| '+label+' | '+' | '.join(str(counts.get(k,0)) for k in ('Win','Tie','Loss'))+' |')
    lines+=['','### 3.3 法条减少发生在哪一步','',
        '| 组别 | 真值父块未召回次数 | 已召回但未进入重排20条 | 进入重排但未进最终5条 |','|---|---:|---:|---:|']
    for v in versions:
        c=statute_causes[v];lines.append('| '+v+' | '+' | '.join(str(c.get(k,0)) for k in ('not_recalled','lost_rerank','lost_evidence'))+' |')
    lines+=['','上表只统计已标注法条父块的漏失次数，不把A无法查询案例算入法条退化。逐题对应的ID和阶段保存在pilot_summary.json。','',
        '## 4. 用已有结果回放一次法条保底（没有新增API调用）','',
        '对相同的20个真实重排子块重新归组，在最终5条证据中优先留出最多3个法条父块，其余按原父块排序补齐。政策取自原任务中的示例，没有搜索最优比例。它只能挽回仍在重排候选中的法条。','',
        '| 组别 | 原法条父块Recall@5 | 保底后 | 原综合父块Recall@5 | 保底后 |','|---|---:|---:|---:|---:|']
    for v in ('B','B_fair'):
        q=quota[v]['scoped_metrics'];lines.append(f"| {v} | {m(v,'statute_gt'):.4f} | {q['statute_gt']['parent']['evidence']['recall@5']:.4f} | {m(v):.4f} | {q['all_gt']['parent']['evidence']['recall@5']:.4f} |")
    lines+=['','## 5. 对项目工作的直接建议','',
        '1. 案例库有保留价值。这批需要类案帮助的咨询题中，综合依据覆盖高于单法条库；但不能据此声称所有问题都更准确。',
        '2. 下一步优先考虑按意图路由、限制案例占用证据名额，或给法条设置保底。先看配额回放能改善多少，再决定是否改线上代码；被Rerank提前淘汰的法条还需要前置保护。',
        '3. 当前不值得继续扩大评测耗时。B的60候选在本小样本中没有胜过30候选的B-Fair，暂时没有证据支持单纯增加候选量。精确条号匹配、BM25、RRF的最终收益本次未实际测定，不给出“能直接救回多少”的结论。','',
        '## 6. 文件、运行记录与范围限制','',
        '| 文件 | 内容 |','|---|---|',
        '| eval/kb_catalog.json | 302份真实资料的目录与父块摘要。 |',
        '| eval/queries_windtunnel_60.jsonl / annotation_validation.json | 已完成的60题和冻结校验；不代表60题实验已跑完。 |',
        '| eval/queries.jsonl | 原3条占位示例已替换真实ID和父块。 |',
        '| eval/eval_retrieval.py | 升级后的双目录、多目标、双粒度和分真值评测。 |',
        '| eval/run_windtunnel_ab.py | 原完整运行入口；已有缓存可供以后需要时继续。 |',
        '| eval/results_windtunnel/version_A.json / version_B.json / version_B_fair.json | 停止时的原始检查点，组间数量可能不相同。 |',
        '| eval/results_windtunnel/pilot_version_A.json / pilot_version_B.json / pilot_version_B_fair.json | 本报告使用的相同21题，各含完整轨迹、分组和所有指标。 |',
        '| eval/results_windtunnel/pilot_summary.json | 小样本汇总、逐题胜负、漏失阶段和配额回放。 |',
        '| eval/summarize_windtunnel_pilot.py | 仅从现有检查点生成粗测报告，无外部请求。 |','',
        '### 6.1 新增、修改的源代码及完整路径','',
        '项目根目录为 `G:\\MSBD5018_agentic-rag-law-main`。源代码保存在 `eval`，实验Markdown报告统一保存到 `Refinement and Optimization`；JSON结果继续保存在 `eval/results_windtunnel`。','',
        '| 状态 | 源代码完整路径 | 主要入口与作用 |','|---|---|---|',
        '| 新增 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\build_windtunnel_dataset.py` | `build_catalog()`、`build_queries()`、`validate_queries()`：整理真实目录、建立测试题并核验原文。 |',
        '| 修改 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\eval_retrieval.py` | `build_indexes()`、`evaluate_one()`、`metrics_for()`：执行实际检索链路，计算分口径指标，检查重排回退。 |',
        '| 新增 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\run_windtunnel_ab.py` | `main()`：运行A/B/B-Fair，共享Query Embedding并保存检查点；本次在中途停止。 |',
        '| 新增 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\test_windtunnel.py` | 离线检查指标口径、真实ID、生产链路一致性及回退检测，不调用外部服务。 |',
        '| 新增 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\summarize_windtunnel_pilot.py` | `main()`：按三组共同完成的题目生成本粗测报告，离线回放法条配额。 |',
        '| 新增，预备代码 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\analyze_windtunnel.py` | 完整60题报告及词法诊断入口；本次没有执行完整分析，不应当作已验证的实验结果。 |','',
        f"停止时已落盘组数：{summary['raw_completed_groups']}；三组比较只使用共同{n}题。Embedding缓存已保存，可供以后恢复；进程停止时可能有尚未完成的请求，本地记录不能据此保证服务端未计费。",
        '本次评测脚本只执行只读搜索和计数，没有创建、删除或更新Qdrant数据。停止后没有再执行完整版的远端后置核对，不声称已经完成完整60题的前后审计。',
        '配置文件和原《案例库构建与双库接入优化记录.md》哈希核对未变；本次没有更新该文档。','',
        '限制：只有21道个案咨询，且18道含案例真值，样本偏向案例发挥作用的场景；标注由同一整理者按原文完成，并不穷举所有同类案例和重复版本。父块命中也不代表截取后的900字正文一定保留目标规则。本次没有生成答案，所以这些数字不是答案准确率。','',
        '重建本报告只需：','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -X utf8 eval\\summarize_windtunnel_pilot.py','```','']
    report_path=ROOT_DIR/'Refinement and Optimization/report_windtunnel_ab.md'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'paired_queries':n,'query_types':summary['query_types'],'quota_replay':{
        v:{'statute_parent_R5':quota[v]['scoped_metrics']['statute_gt']['parent']['evidence']['recall@5'],
           'all_parent_R5':quota[v]['scoped_metrics']['all_gt']['parent']['evidence']['recall@5']} for v in quota},
        'paired_outcomes':summary['paired_outcomes'],'protected_files_unchanged':True},ensure_ascii=False))


if __name__=='__main__':main()
