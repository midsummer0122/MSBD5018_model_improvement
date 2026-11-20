"""Reconcile saved results and write the comprehensive Chinese report; no APIs."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

from eval_retrieval import ROOT_DIR, build_indexes, load_queries, validate_labels, metrics_for, summarize
from run_ablation_4way import OUT, QUERY_PATH, write_json

REPORT=ROOT_DIR/'Refinement and Optimization/report_ablation_4way.md'
VERSIONS=('V1','V2','V3','V4')


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def fmt(v):return f'{v:.4f}'
def link(relative,label=None,line=None):
    path=ROOT_DIR/relative
    return f'[{label or relative}](<{path.as_posix()}{":"+str(line) if line else ""}>)'


def table(headers,rows):
    clean=lambda v:str(v).replace('|','／').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(clean,headers))+' |',
        '| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(clean,row))+' |' for row in rows])+'\n'


def run():
    manifest=read('first_run_manifest.json') if (OUT/'first_run_manifest.json').exists() else read('run_manifest.json')
    if not manifest['complete']:raise ValueError('Incomplete experiment')
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    queries=load_queries(QUERY_PATH);validate_labels(queries,index)
    data={v:read(v+'.json') for v in VERSIONS};preflight=read('preflight_mysql.json')
    subsets=read('subset_manifest.json');checks=read('top_n_checks.json');routes=read('route_checks.json')
    events=read('api_events.json');times=read('query_timings.json')
    for v in VERSIONS:
        if len(data[v]['details'])!=20:raise ValueError('Wrong result count')
        for row in data[v]['details']:
            recalculated=metrics_for(row,row['doc_ranks'],row['parent_ranks'],index)
            if recalculated!=row['scoped_metrics']:raise ValueError('Stored metrics inconsistent with rankings')
        if summarize(data[v]['details'])!=data[v]['summary']:raise ValueError('Stored aggregate inconsistent')
    by={v:{r['query_id']:r for r in data[v]['details']} for v in VERSIONS}
    def scalar(v,scope='all_gt',grain='parent',stage='evidence',metric=None,rows=None):
        metric=metric or ('recall@20' if stage=='rerank' else 'recall@5')
        if rows is None:return data[v]['summary']['scoped_metrics'][scope][grain][stage][metric]
        values=[r['scoped_metrics'][scope][grain][stage][metric] for r in rows
                if r['scoped_metrics'][scope][grain][stage]]
        return mean(values) if values else None
    def kind(pid):return index.doc_meta[index.parent_to_doc[pid]]['doc_type']
    def ids_label(ids):
        return '；'.join(('法条:' if kind(pid)=='statute' else '案例:')+f'`{pid}`' for pid in sorted(ids)) or '—'
    def transitions(old,new,stage,qid):
        relevant=set(by[old][qid]['relevant_parent_ids'])
        a=set(by[old][qid]['parent_ranks'][stage]) & relevant
        b=set(by[new][qid]['parent_ranks'][stage]) & relevant
        return sorted(b-a),sorted(a-b)
    ledger=[]
    for q in queries:
        qid=q['query_id'];g3,l3=transitions('V2','V3','rerank',qid)
        e3,d3=transitions('V2','V3','evidence',qid);e4,d4=transitions('V3','V4','evidence',qid)
        r={'query_id':qid,'query_type':q['query_type'],'rerank_v3_v2_gains':g3,'rerank_v3_v2_losses':l3,
            'rerank_gains_reaching_v3_evidence':sorted(set(g3)&set(by['V3'][qid]['parent_ranks']['evidence'])),
            'rerank_losses_previously_in_v2_evidence':sorted(set(l3)&set(by['V2'][qid]['parent_ranks']['evidence'])),
            'evidence_v3_v2_gains':e3,'evidence_v3_v2_losses':d3,
            'evidence_v4_v3_gains':e4,'evidence_v4_v3_losses':d4}
        ledger.append(r)
        if [(c['chunk_id'],c['rerank_score']) for c in by['V3'][qid]['rerank_chunks']]!=[(c['chunk_id'],c['rerank_score']) for c in by['V4'][qid]['rerank_chunks']]:
            raise ValueError('V3/V4 child rankings differ')
    write_json(OUT/'gains_displacements.json',ledger)
    def total(field,source):return sum(sum(kind(p)==source for p in r[field]) for r in ledger)
    # End-to-end trace of each annotated parent; do not turn retrieval output into labels.
    traces=[]
    for q in queries:
        for pid in q['relevant_parent_ids']:
            traces.append({'query_id':q['query_id'],'parent_id':pid,'doc_type':kind(pid),
                **{v:{s:pid in by[v][q['query_id']]['parent_ranks'][s] for s in ('recall','rerank','evidence')} for v in VERSIONS}})
    write_json(OUT/'parent_trace.json',traces)
    # Immutable experiment summary suitable for later independent review.
    summary={'versions':{v:data[v]['summary'] for v in VERSIONS},
        'ledger_counts':{field:{s:total(field,s) for s in ('statute','case')} for field in ledger[0] if field not in ('query_id','query_type')},
        'recomputed_all_metrics_match':True,'v3_v4_rankings_equal':True}
    write_json(OUT/'analysis_summary.json',summary)
    lines=[]
    def add(text=''):lines.append(text.strip()+'\n')
    add('# 劳动法 Agentic RAG：案例建库、双库接入与两阶段来源配额消融报告')
    add('日期：2026-11-20。项目：`G:\\MSBD5018_agentic-rag-law-main`。本报告把“为什么补案例、怎样接入双库、为什么继续优化证据选择、实际测到了什么”串起来，方便组员理解和复现。历史记录保留，本文件独立成篇。')
    add('## 1. 本次完成了什么，结果怎样')
    add(f'原项目只有中央法规。本项目前期新增了 174 份劳动案例的提取、切分、向量化与入库，并打通法条＋案例双库检索。本轮在此基础上实现了**对双库候选统一重排打分，再按意图设置来源配额选择子块／父块**，实际完成 20 题 × 4 组的检索消融。')
    add(table(['Evidence@5 父块指标','V1 单法条','V2 双库全局','V3 子块配额','V4 子块＋父块配额'],
        [[label]+[fmt(scalar(v,scope)) for v in VERSIONS] for label,scope in [('法条 Recall','statute_gt'),('案例 Recall（9题）','case_gt'),('综合 Recall','all_gt')]]))
    add('上述数字是对预先标注依据的覆盖率，**不是法律回答准确率，也不是上线收益保证**。本轮不运行答案生成与安全审查，不评价措辞、引用正确性或法律结论。')
    add(f'MySQL 860 个父块全量一致；禁止 JSON 回退后，511 个法条父块和 349 个案例父块均由运行时 MySQL 分支完整取得。3 道 `top_n` 检查通过：前 20 排名及分数完全一致。4 道线上真实意图识别与配额映射均符合抽查预期。真实调用成功数：{manifest["api_success_counts"]}；失败尝试 {manifest["api_failed_attempts"]}。')
    add('## 2. 想解决的问题：从“只有规则”到“规则和实际处理思路一起看”')
    add('### 2.1 为什么补劳动案例')
    add('法条适合回答“规则是什么、期限多久、赔偿怎样计算”。个案咨询还常问“类似情况法院怎样处理、证据怎样认定”。因此补充案例，给生成环节提供实际争议和裁判思路。案例帮助理解规则，不能替代适用的法条。')
    add('原始案例是 PDF，不能直接交给当前子块向量检索流程。前期实现先提取 PDF 正文、清理重复页眉页脚，再切成父块和子块：子块用于语义搜索；父块保存较完整上下文，召回子块后凭 `parent_id` 回补。')
    add(table(['知识库','原始资料／JSON目录','文档','父块','子块／向量','Qdrant集合'],[
        ['中央法规','knowledge_data/中央法规 → knowledge_data/json',128,511,1384,'law_statutes'],
        ['劳动案例','knowledge_data/劳动案例 → knowledge_data/json_cases',174,349,2334,'law_cases'],
        ['合计','两个目录分别保留',302,860,3718,'两个独立集合']]))
    add('向量模型为 `text-embedding-v4`，1,024 维，Cosine；重排模型为 `qwen3-rerank`。中央法规复用已有向量，案例前期首次生成了向量；**本轮没有重新生成文档 Embedding，没有上传、删除或重建 Qdrant 数据**。仅为 20 个问题各生成一次查询向量。')
    add('案例提取沿用项目父子切分思路：父块偏长以保留上下文，子块偏短以便命中具体事项。脚本仍是文本切分，不是法律专家逐段标注；扫描件 OCR、案例有效性、裁判观点适用范围未在本轮新增人工审核。详细历史过程见 '+link('Refinement and Optimization/案例库构建与双库接入优化记录.md')+'。')
    add('### 2.2 双库已经能问答，为什么还需要这次试验')
    add('双库一起召回，不代表最后 5 条证据能合理兼顾两类来源。一个案例可能被多个子块命中，或在重排中普遍得分较高；全局截取时，法条会被挤掉。反过来，强行保留法条也可能挤掉真正相关的案例。我们需要同时记录两方面得失。')
    add('前一次粗测只比较了三组共同完成的 21 道法律咨询题，其中 18 题有案例真值。单法条 A 的最终法条父块 Recall 为 0.6667，双库 B 为 0.3413；综合父块 Recall 则从 0.4127 到 0.6032。它提示“综合覆盖上升但法条减少”，但不能代表四类意图。原表见 '+link('Refinement and Optimization/report_windtunnel_ab.md')+'。')
    add('本轮扩大意图覆盖，并把改动拆成递进四组。旧粗测与新试验题目组成不同，不能直接用均值相减当作改进收益；本报告中的主要比较只在同一批 20 题内进行。')
    add('## 3. MySQL 前置核验：排除 JSON 回退造成的假成功')
    add('### 3.1 实际实例和全量一致性')
    add(f'本机 MySQL {preflight["server"]["version"]}，`127.0.0.1:3306`，数据库 `law_rag`，项目账号 `law_rag_app@127.0.0.1`。账号用于查询／插入／更新，本轮没有执行 DDL，也没有修改账号或密码。服务名 `MySQL84LawRag`；程序位于 `G:\\MySQL\\mysql-8.4.13-winx64`。')
    add('核验脚本用 `utf-8-sig` 读取两个 JSON 目录，按目录建立来源类型表；再直接联查 `parent_chunks` 与 `documents`，比对所有预期 `parent_id` 的正文、`title_path`、关联 `doc_id` 和文档标题。860／860 一致，数据库恰好有 860 个父块，本次无需补导入。法条 ID 包含多个首位数字，分类没有使用 ID 前缀。')
    add(table(['核验','数量／耗时','结果'],[
        ['直接 MySQL 联表读回','860 个；'+fmt(preflight['direct_read_seconds'])+' 秒','全量正文、标题路径、文档关联一致'],
        ['运行时 get_many 法条','511 个；'+fmt(preflight['runtime_mysql_seconds']['statute'])+' 秒','JSON回退禁止，全部返回'],
        ['运行时 get_many 案例','349 个；'+fmt(preflight['runtime_mysql_seconds']['case'])+' 秒','JSON回退禁止，全部返回'],
        ['JSON回退调用',preflight['runtime_json_fallback_calls'],'0次；测试对象未加载本地父块索引']]))
    add('### 3.2 证明方法和日常容错')
    add('核验时临时把 `ParentChunkStore._get_many_local()` 替换成“一旦被调用就抛异常”，再走正常 `get_many()`。若 MySQL 连接失败或缺父块，测试会直接失败。20 题真实召回时也逐题禁止回退，并比对被回补的正文和标题路径。测试结束后替换自动撤销，日常 JSON 回退功能仍保留。')
    add('注意：读取本地 JSON 来统计预期集合、检查真值或建立来源映射，与“用 JSON 代替 MySQL 回补正文”是不同的事情；本轮正文回补只走 MySQL。证据见 '+link('eval/results_ablation_20/preflight_mysql.json')+'。')
    add('## 4. 已实现的优化：统一打分，两处选择配额')
    add('### 4.1 来源识别与意图映射')
    add('来源优先读取 `payload.doc_type`，其次用本次召回的 `statute_db`／`case_db` 标签；仍缺失时，用双目录建立的精确文档／父块／子块 ID 映射。运行时在本地 payload 副本里记录来源集合，不改远端 payload。评测同时把集合来源与本地映射交叉核对，发现冲突即停止。')
    add('真实结构在 `rag/schemas.py`，本地没有 `rag/types.py`。线上 `query_type` 有 `knowledge_question`、`legal_consultation`、`procedure_consultation`；`intents` 是中文多选列表。评测集的 `query_type` 则直接是四类中文标签。统一函数 `resolve_intent()` 同时支持这两种输入。')
    add('优先级为：明确案例检索 → 个案法律咨询 → 程序咨询 → 法条解释；未识别时使用法律咨询。例：同时有案例检索和法律咨询时选案例策略；具体咨询同时含程序意图时选法律咨询策略。这是明确的工程规则，不是对分类准确率的结论。')
    add(table(['策略','20个子块的法条／案例名额','5个父块的法条／案例名额'],[
        ['法条解释／程序咨询','16／4','4／1'],['法律咨询','12／8','3／2'],['案例检索','8／12','2／3']]))
    add('数量变化时按比例缩放，整数部分先分配，再按小数余数补齐，始终保证总数一致；单库路由时名额全部给该库。`route_skills()` 目前默认仍是每库召回 30 个、最终 5 个父块，默认生成子块和父块两阶段配额。')
    add('### 4.2 子块重排：同一把尺子打分，再分名额')
    add('`split_rerank=True` 的含义是：双库候选统一发出 **1 次** Rerank 请求，`top_n=len(documents)`，取得全部候选分数。随后分别取两类的配额，再合并按同一分数排序。没有分别调用两次模型给两库打分，避免把分数口径差异误当作来源效果。')
    add('一类候选不够时，另一类剩余候选按分数补齐。例如需要 12 条法条、8 条案例，但只有 5 条案例，则可用更多法条补到 20 条。若两类都充足，名额会被用满，即使被保留的某些候选分数很低；本轮不加最低分阈值。`split_rerank=False` 则按全局分数取前 20，不应用子块配额。')
    add('为兼容此前直接调用 `Retriever.rerank(query,chunks,final_k)` 的评测或工具，该方法新增关键字开关默认 `False`；线上 `Retriever.retrieve(plan)` 明确传入计划中的开关，`RetrievalPlan.split_rerank` 默认 `True`。因此旧直接调用仍保持全局策略，线上路由则使用新计划。')
    add('### 4.3 父块证据：保持原打分公式，只改变选择方式')
    add('同一父块对应的子块先聚合，父块分数仍是原公式：\n\n```text\nparent_score = 该父块最高子块分数 + 0.02 × log(1 + 命中的子块数量)\n```')
    add('原有 `0.02 × log1p(...)` 是多子块命中项，两类来源都适用。本轮**没有给案例另加权威分，没有额外 +0.02，没有改变父块分数**。这保证消融只改变来源名额。')
    add('`evidence_quotas=None` 时仍按父块分数全局取前 5；传入配额后分别选两类父块，不足则补齐，最后按父块分数排序。项目实际引用键叫 `evidence_id`，因此继续依次赋 `ev_001`、`ev_002` 等，没有另增不兼容的 `citation_id`。`workflow.py` 已把计划中的父块配额传给 `build_evidence()`。')
    add('### 4.4 线上真实意图映射抽查')
    add('仅执行问题理解和计划生成，没有执行检索、答案生成或审查。4 次 DeepSeek 响应均被记录并解析，缺少关键结构或发生规则兜底不能计作成功。')
    add(table(['抽查目标','LLM query_type','LLM intents','映射策略','子块／父块配额','符合预期'],[
        [r['expected'],r['raw_llm_query_type'],'、'.join(r['raw_llm_intents']),r['actual'],
         str(r['plan']['rerank_quotas'])+'／'+str(r['plan']['evidence_quotas']),r['matches_expected']] for r in routes]))
    add('这是 4 道轻量接线检查，不能解释为线上意图识别达到 100% 准确率。完整输入、输出和计划见 '+link('eval/results_ablation_20/route_checks.json')+'。')
    add('## 5. 实验怎样控制变量')
    add('### 5.1 20题及真值来源')
    add('本轮是**优化诊断用的探索性子集，部分题目已查看过上次结果**。正式盲测应由评测负责人独立准备和运行。没有把这 20 题说成独立测试集，也没有按新结果重写真值。')
    add(f'从原 60 题中按意图分组、保持原顺序，固定抽取等间距位置 `round(i*(N-1)/(n-1))`。共 8／5／4／3 题；有案例真值 {subsets["with_case_gt"]} 题，纯法条真值 {subsets["pure_statute"]} 题。共 {subsets["source_quote_checks"]} 处原文摘录校验通过。四类 ID 列表逐题保留；父块正文哈希、摘录和关联检查均通过。')
    add('真值来自此前按原文设计的题目及标注，不来自本轮向量排序。它们是选定的核心相关依据，不保证穷尽所有同样能回答问题的资料；召回一个未标注但实际有帮助的依据会被记为未命中。这一局限需要独立人工复核。')
    add('实验直接以人工中文 `query_type` 决定配额，回答的是“**意图判断正确时，两阶段配额有什么影响**”。不调用 LLM 改写这 20 个问题，不把分类误差混入本次检索比较；原始 dev/test 标记仍保留，但新子集整体按探索性诊断解释。')
    add(table(['题号','类型','问题','法条／案例真值文档','真值父块'],[
        [q['query_id'],q['query_type'],q['query'],','.join(q['relevant_statute_doc_ids'])+'／'+(','.join(q['relevant_case_doc_ids']) or '无'),','.join(q['relevant_parent_ids'])] for q in queries]))
    add('### 5.2 四组递进设计与固定请求安排')
    add(table(['组别','向量候选','保留20个子块','保留5个父块','隔离的变化'],[
        ['V1','法条Top30','全局重排前20','父块全局前5','单库法条参照'],
        ['V2','法条Top30＋案例Top30','统一打分，全局前20','父块全局前5','引入案例：V2−V1'],
        ['V3','同V2','同一批分数，按意图配额选20','父块全局前5','子块配额：V3−V2'],
        ['V4','同V2','与V3逐条ID和分数完全相同','按意图配额选5','叠加父块配额：V4−V3']]))
    add('每题只做 1 次 Query Embedding、两库各 1 次真实搜索、一次 MySQL 父块回补、1 次法条单独 Rerank、1 次双库全部分数 Rerank。V2／V3／V4 复用同一批真实双库分数，在本地经过实际 `Retriever.rerank()` 和 `build_evidence()` 逻辑回放。分数缓存器只重放真实响应，没有虚构或离线替代评分。V3 与 V4 的重排列表和分数在 20 题全部相同。')
    add('本设计没有单独测试“只加最终证据配额、不加重排配额”，因此不能据此声称两阶段都是必要条件。V2 的候选从 30 增至 60，也同时改变来源和候选池规模；它衡量的是当前双库接入方案的总体影响。')
    add('### 5.3 top_n 截断等价性前置检查')
    add('选定 3 题，在相同 60 个候选、相同顺序和文本上分别请求 `top_n=20` 和 `top_n=60`；后者取前 20 比较。完整分数请求被主实验复用，额外只需 3 次截断请求。先完成该检查，再汇总主实验。')
    add(table(['题号','前20索引一致','分数完全一致','最大分差','额外请求秒'],[
        [c['query_id'],c['same_top20_indices'],c['exact_scores_equal'],c['max_score_difference'],fmt(c['extra_request_seconds'])] for c in checks]))
    add('3 题结果完全一致，支持本轮使用全部分数回放 V2。它验证了这 3 个请求的行为，不构成对任意未来输入和服务版本的数学保证。若检查不一致，脚本会额外保留真实 `top_n=20` 响应作为 V2，不把不同结果偷偷视作等价。')
    add('### 5.4 指标读法与分母')
    add('`Rerank@20` 看保留的 20 个子块涉及哪些文档／父块；`Evidence@5` 看最终最多 5 个父块。文档级先按首次出现去重；父块级同样去重。因此它们反映的是该阶段列表覆盖的唯一资料，而不是 20 个独立文档。')
    add('Recall 是覆盖了多少已标注相关项；MRR 是第一条相关项排名的倒数；Hit 是是否至少命中一条相关项，均逐题计算后宏平均。MRR 在该阶段截断后的去重列表内计算。计算法条指标时**不从结果中删掉案例**，反之亦然，避免把挤占问题隐藏掉。')
    add('`statute_gt`：20 题的法条真值；`all_gt`：20 题的全部真值；`case_gt`：只平均 9 道有案例真值的题，不把另 11 题记作案例零分。')
    add('## 6. 全局完整指标')
    for scope,scope_name in [('statute_gt','法条真值 statute_gt（20题）'),('case_gt','案例真值 case_gt（9题）'),('all_gt','全部真值 all_gt（20题）')]:
        for grain,grain_name in [('doc','文档级'),('parent','父块级')]:
            add(f'### {scope_name} · {grain_name}')
            rows=[]
            for stage,k in [('rerank',20),('evidence',5)]:
                for metric,name in [(f'recall@{k}','Recall'),('mrr','MRR'),(f'hit@{k}','Hit')]:
                    rows.append([f'{"Rerank" if stage=="rerank" else "Evidence"}@{k}',name]+[
                        fmt(scalar(v,scope,grain,stage,metric)) for v in VERSIONS])
            add(table(['阶段','指标',*VERSIONS],rows))
    add('### 平均最终证据结构')
    add(table(['组别','平均法条父块','平均案例父块','平均总数'],[
        [v,fmt(data[v]['summary']['avg_statute_evidence_count']),fmt(data[v]['summary']['avg_case_evidence_count']),
        fmt(data[v]['summary']['avg_statute_evidence_count']+data[v]['summary']['avg_case_evidence_count'])] for v in VERSIONS]))
    add('## 7. 按意图看变化，以及硬配额的代价')
    add('### 7.1 四类问题的最终父块覆盖')
    rows=[]
    for intent in subsets['allocation']:
        for v in VERSIONS:
            subset=[r for r in data[v]['details'] if r['query_type']==intent]
            vals=[scalar(v,s,rows=subset) for s in ('statute_gt','case_gt','all_gt')]
            rows.append([intent,len(subset),v]+[fmt(x) if x is not None else '无案例真值，不计分' for x in vals]+
                [fmt(mean(r['evidence_counts']['statute'] for r in subset)),fmt(mean(r['evidence_counts']['case'] for r in subset))])
    add(table(['意图','题数','组别','法条Recall@5','案例Recall@5','综合Recall@5','平均法条数','平均案例数'],rows))
    add('分意图更完整的 Rerank／Evidence、文档／父块、Recall／MRR／Hit 保存在各版本 JSON 的 `groups` 中，避免只保留对改进有利的指标。每类仅 3—8 题，单题变化就能明显影响均值。')
    add('### 7.2 纯法条题：案例预留位是否有代价')
    pure_ids={q['query_id'] for q in queries if not q['relevant_case_doc_ids']}
    rows=[]
    for v in VERSIONS:
        subset=[r for r in data[v]['details'] if r['query_id'] in pure_ids]
        rows.append([v,len(subset),fmt(scalar(v,'statute_gt',rows=subset)),
            fmt(scalar(v,'statute_gt',stage='rerank',rows=subset)),fmt(mean(r['evidence_counts']['case'] for r in subset))])
    add(table(['组别','纯法条题数','法条父块Recall@5','法条父块Recall@20','平均案例占位'],rows))
    for old,new in [('V2','V3'),('V3','V4')]:
        deltas=[scalar(new,'statute_gt',rows=[by[new][qid]])-scalar(old,'statute_gt',rows=[by[old][qid]]) for qid in pure_ids]
        add(f'{new} 相对 {old}：纯法条题法条 Recall@5 改善 {sum(x>1e-12 for x in deltas)} 题、持平 {sum(abs(x)<=1e-12 for x in deltas)} 题、下降 {sum(x< -1e-12 for x in deltas)} 题；平均变化 {mean(deltas):+.4f}。')
    add('这里“纯法条”表示真值只标法条，不等于案例一定无用。但若这些题中给案例固定占位又漏掉已标注法条，就构成可观测的覆盖代价；没有漏掉标注法条的低分案例占位，也不能据此证明其生成价值。')
    add('### 7.3 有案例真值的题：法条保底是否挤掉必要案例')
    case_ids={q['query_id'] for q in queries if q['relevant_case_doc_ids']};rows=[]
    for v in VERSIONS:
        subset=[r for r in data[v]['details'] if r['query_id'] in case_ids]
        rows.append([v,len(subset),fmt(scalar(v,'case_gt',rows=subset)),
            fmt(scalar(v,'case_gt',stage='rerank',rows=subset)),fmt(scalar(v,'all_gt',rows=subset))])
    add(table(['组别','含案例真值题数','案例父块Recall@5','案例父块Recall@20','综合父块Recall@5'],rows))
    for old,new in [('V2','V3'),('V3','V4')]:
        deltas=[scalar(new,'case_gt',rows=[by[new][qid]])-scalar(old,'case_gt',rows=[by[old][qid]]) for qid in case_ids]
        add(f'{new} 相对 {old}：案例 Recall@5 改善 {sum(x>1e-12 for x in deltas)} 题、持平 {sum(abs(x)<=1e-12 for x in deltas)} 题、下降 {sum(x< -1e-12 for x in deltas)} 题；平均变化 {mean(deltas):+.4f}。')
    add('## 8. 双向父块变动账本：救回什么，也移出了什么')
    add('只记录**已标注相关父块**的进入／退出，未标注项不直接叫“错误”。“移出”指从保留集合移出，不表示数据库被删除。每题中的法条／案例类型均来自本地目录映射。计数单位是“题目—父块事件”，同一个父块出现在不同题中会分别计数。')
    count_rows=[]
    for name,gain,loss in [('V3−V2：Rerank20','rerank_v3_v2_gains','rerank_v3_v2_losses'),
        ('V3−V2：Evidence5','evidence_v3_v2_gains','evidence_v3_v2_losses'),
        ('V4−V3：Evidence5','evidence_v4_v3_gains','evidence_v4_v3_losses')]:
        count_rows.append([name,total(gain,'statute'),total(loss,'statute'),total(gain,'case'),total(loss,'case')])
    add(table(['变动阶段','救回法条','移出法条','救回案例','移出案例'],count_rows))
    add(f'V3 重排阶段新救回的相关父块中，真正出现在 V3 最终证据里的有法条 {total("rerank_gains_reaching_v3_evidence","statute")} 个、案例 {total("rerank_gains_reaching_v3_evidence","case")} 个。重排救回不等于最终得到引用机会；子块命中数变化也可能改变原公式中的多命中项，从而影响父块顺序。')
    add('### 8.1 V3 对 V2：子块配额及传导')
    add(table(['题号','类型','重排救回相关父块','重排移出相关父块','救回后进V3最终证据','V3最终证据新增','V3最终证据失去'],[
        [r['query_id'],r['query_type'],ids_label(r['rerank_v3_v2_gains']),ids_label(r['rerank_v3_v2_losses']),
            ids_label(r['rerank_gains_reaching_v3_evidence']),ids_label(r['evidence_v3_v2_gains']),ids_label(r['evidence_v3_v2_losses'])] for r in ledger]))
    add('### 8.2 V4 对 V3：父块配额的增量')
    add(table(['题号','类型','最终证据救回相关父块','最终证据移出相关父块'],[
        [r['query_id'],r['query_type'],ids_label(r['evidence_v4_v3_gains']),ids_label(r['evidence_v4_v3_losses'])] for r in ledger]))
    add('完整机器可读账本见 '+link('eval/results_ablation_20/gains_displacements.json')+'；每个真值父块在四组各阶段的位置存在性见 '+link('eval/results_ablation_20/parent_trace.json')+'。各组原始 `parent_ranks` 和子块分数也保留，可自行重新核算。')
    add('## 9. 如何解读这些发现，下一步选哪种方案')
    deltas={(a,b,s):scalar(b,s)-scalar(a,s) for a,b in [('V1','V2'),('V2','V3'),('V3','V4')] for s in ('statute_gt','case_gt','all_gt')}
    add(table(['递进变化','法条父块Recall@5变化','案例父块Recall@5变化','综合父块Recall@5变化'],[
        [f'{b}−{a}']+[f'{deltas[a,b,s]:+.4f}' for s in ('statute_gt','case_gt','all_gt')] for a,b in [('V1','V2'),('V2','V3'),('V3','V4')]]))
    add('### 9.1 子块保护和最终保护解决的是两个不同位置的问题')
    add('V3 能让另一来源的子块进入 20 条候选，但最终 5 个父块仍是全局排序。如果法条父块分数较低，重排阶段救回后仍可能排在第 6 名以后。因此是否需要 V4，应看“救回是否传导”的账本，不能只看子块比例已经好看。')
    add(f'本轮 V3 相对 V2 的综合父块 Recall@5 变化为 {deltas["V2","V3","all_gt"]:+.4f}，法条变化 {deltas["V2","V3","statute_gt"]:+.4f}；V4 叠加后的综合变化为 {deltas["V3","V4","all_gt"]:+.4f}，法条变化 {deltas["V3","V4","statute_gt"]:+.4f}。这些均是固定 20 题下的实测值。')
    if deltas['V3','V4','all_gt']>0 and deltas['V3','V4','statute_gt']>0:
        add('在本批样本内，V4 比只在重排阶段保护更能恢复最终法条覆盖，可作为下一轮重点比较方案。是否推广仍要结合案例损失、纯法条副作用和独立盲测，不应称为已经证明“两阶段必需”。')
    else:
        add('本批样本没有支持无条件采用 V4 的充分证据。应保留 V3 与 V4 作为可切换方案，结合不同来源的双向得失，再决定线上默认策略。')
    add('### 9.2 为什么值得继续测“软保底”')
    add('本轮的具体代价不能忽略：V4 的综合父块 Recall 虽从 V3 的 0.7917 到 0.8083，但综合 Hit 从 1.0000 降到 0.9500，MRR 从 0.9375 降到 0.9250。q004 中，相关案例父块 `50071_p001` 被最终法条名额挤出，且没有其他已标注依据进入最终5条，导致该题从有命中变成无命中。这是一次明确的负面结果。')
    add('同时，V4 的法条父块 Recall 0.7250 仍低于单法条 V1 的 0.8250；它恢复了一部分覆盖，没有完全恢复单库法条表现。11道纯法条题的标注覆盖在四组相同（0.9545），本批未观测到配额造成的额外下降；但其中仍有案例占位，不能据此保证扩展题集后没有代价。')
    add('当前硬配额会把名额用满，不能识别“本题其实没有足够相关案例”。下一轮可以先按来源预留名额，但只让达到最低相关性要求的候选占位，空出来的位再交给全局高分候选。这样有机会减少低分占位，但阈值需要独立开发集校准，不能看这 20 题逐题调到最好。')
    add('本轮未实现或测试软阈值，也没有单独跑“仅父块配额”组；因此它们是待检验方案，不是已获得的收益。建议下一轮最少比较 V3、V4、仅父块配额、软保底，并独立审核被移出的相关案例。')
    add('实际决策上，若优先避免把已命中的案例挤掉，V3 是本轮更保守的参照；若优先提高法条最终覆盖，V4值得继续验证，但不宜直接当作全面改进。综合考虑 q004 的失去全部标注命中，更合理的下一步是验证软保底和“仅最终配额”，而不是继续调大硬配额并宣布最优。')
    add('### 9.3 当前线上默认与实验结论的区别')
    add('按本次实现要求，`route_skills()` 默认同时填写两阶段配额，对应 V4。它是已接通、可切换的探索性策略，不代表已通过正式验收。要复现 V3，可保留 `split_rerank=True` 和 `rerank_quotas`，将 `plan.evidence_quotas=None`；V2 再令 `plan.split_rerank=False`。关闭这些开关不会删除任何知识库数据。')
    add('## 10. 实际耗时、API用量和缓存口径')
    timing_rows=[]
    for field,label in [('embedding_seconds','每题 Query Embedding'),('search_seconds','每题双库搜索合计'),('parent_seconds','每题 MySQL 回补'),
        ('statute_rerank_seconds','每题法条单独 Rerank'),('full_rerank_seconds','每题60候选统一 Rerank')]:
        values=[t['timings'][field] for t in times]
        timing_rows.append([label,fmt(mean(values)),fmt(min(values)),fmt(max(values)),fmt(sum(values))])
    add(table(['环节','平均秒／题','最短秒','最长秒','20题累计秒'],timing_rows))
    add(table(['组别','平均受控检索秒（双库共享搜索＋MySQL＋该组重排）','平均本地重放＋证据构建秒'],[
        [v,fmt(data[v]['summary']['mean_retrieval_seconds']),fmt(mean(r['timings']['local_replay_seconds'] for r in data[v]['details']))] for v in VERSIONS]))
    add(f'本轮实际首次运行用时记录为 {manifest["elapsed_this_invocation_seconds"]:.2f} 秒；其中包括 4 道意图抽查、3 次额外截断检查、逐题服务调用与保存结果。API 成功次数为 {manifest["api_success_counts"]}，没有为 V3／V4重复付费打分。API事件逐条记录调用耗时和成功状态，不保存请求头、令牌或完整异常文本。')
    failed=[e for e in events if not e['ok']]
    if failed:
        add(table(['失败尝试题号','环节','脱敏错误类型','等待秒','处理'],[
            [e.get('query_id','—'),e['phase'],e.get('error','unknown'),fmt(e['seconds']),'自动重试后成功；未采用降级分数'] for e in failed]))
        add(f'本次一次失败尝试累计等待 {sum(e["seconds"] for e in failed):.2f} 秒（约 {sum(e["seconds"] for e in failed)/60:.2f} 分钟），发生在 q025 的额外 `top_n=20` 检查请求，是约42分钟总耗时的主要来源。该检查缓存计时包含重试，不能当作常规单次 Rerank 延迟。其余成功调用合计约 {sum(e["seconds"] for e in events if e["ok"]):.2f} 秒。脱敏日志只保留 RuntimeError，未保留足以区分网络、服务端或本机等待的底层信息，因此不猜测根因。最终所有需要的真实响应均已取得，没有缺题，也没有将失败回退结果计入指标。')
    add('V1 复用双库共享搜索和 MySQL 批次中的法条候选，所以它的受控计时含双库共同开销，**不是独立单库线上延迟**。V2／V3／V4 共享同一次双库重排，表中的相同 API 成本是对照估算，不是三次实际调用。查询 Embedding、额外截断检查、意图识别不计入这一“受控检索秒”。')
    add('各请求缓存保存原始真实请求耗时；断点续跑命中缓存不会把这些耗时改成零。本地重放单独计时。`--replay` 不调用模型或 Qdrant，但仍读 MySQL 重新验证父块，并重建四组结果，因此其本地重放耗时会因机器状态略有变化。')
    if (OUT/'replay_manifest.json').exists():
        replay=read('replay_manifest.json')
        add(f'本次已实际验证缓存回放：共命中 {replay["cache_hits_this_invocation"]} 个真实请求缓存，约 {replay["elapsed_this_invocation_seconds"]:.2f} 秒完成，没有新增模型或Qdrant请求。上面模型／搜索耗时仍取原真实调用记录；MySQL和本地构建耗时取最近一次回放实测，因此不能用这几秒冒充首次真实服务总耗时。首次运行记录独立保存在 `first_run_manifest.json`，缓存回放记录在 `replay_manifest.json`。')
    add('旧粗测每组约 4 秒消耗在不可用 MySQL 后回退 JSON；本轮 MySQL 已可用，样本也换成四类分层 20 题，缓存及共用请求安排不同。不能把新旧耗时差都归功于配额优化。完整计时见 '+link('eval/results_ablation_20/query_timings.json')+'。')
    add('## 11. 文件变更与实现位置，方便小组整理')
    add('### 11.1 本轮新增／修改的生产代码')
    code_files=[('rag/source_policy.py','新增','resolve_intent / source_quotas / chunk_source / select_by_source','统一意图、精确来源识别、比例名额与补齐选择'),
        ('rag/schemas.py','修改','RetrievalPlan','新增 split_rerank、rerank_quotas、evidence_quotas，旧字段保留'),
        ('rag/skill_router.py','修改','route_skills','按真实意图生成两阶段配额；支持数量和单库目标变化'),
        ('rag/retrieval.py','修改','Retriever.retrieve / Retriever.rerank','记录本地来源标签；一次取全分数后按来源选；校验响应完整性'),
        ('rag/evidence.py','修改','build_evidence','原分数公式不变；可选父块配额及不足补齐'),
        ('rag/workflow.py','修改','build_evidence调用','把计划 evidence_quotas 传至证据构建')]
    add(table(['文件（完整路径）','状态','定位','作用'],[[link(p),status,where,what] for p,status,where,what in code_files]))
    add('本轮没有修改 `rag/query_understanding.py` 的模型提示词或意图输出，没有修改 `api_key/api_config.yaml`、依赖文件、父块导入脚本、答案生成或安全审查逻辑。配置文件与两份历史报告前后 SHA-256 完全相同。')
    add('### 11.2 本轮新增的评测代码、题集和结果')
    add(table(['文件／目录（完整路径）','作用'],[
        [link('eval/preflight_ablation.py'),'全量 MySQL 对账；运行时禁回退；只在缺父块且无冲突时允许增量导入'],
        [link('eval/run_ablation_4way.py'),'固定分层20题、真实调用、逐请求缓存、3题截断检查、4题路由抽查、四组重放'],
        [link('eval/test_source_quotas.py'),'来源配额、优先级、缩放、单库、缺额补齐、原分数公式等7项离线单元测试'],
        [link('eval/report_ablation_4way.py'),'从保存排名重新核算指标，生成双向账本及本报告；不调用外部API'],
        [link('eval/queries_ablation_20.jsonl'),'固定20题、原文校验及全部真值字段'],
        [link('eval/results_ablation_20'),'四组原始结果、缓存、调用事件、核验记录和分析结果'],
        [link('Refinement and Optimization/report_ablation_4way.md'),'当前综合报告；实验 Markdown 统一保存在此目录']]))
    add('结果目录重点文件：`V1.json`—`V4.json`（逐题排名、分数、证据及完整指标）、`cache/`（真实向量、候选、重排响应及签名）、`subset_manifest.json`、`preflight_mysql.json`、`route_checks.json`、`top_n_checks.json`、`api_events.json`、`query_timings.json`、`run_manifest.json`、`first_run_manifest.json`、`replay_manifest.json`、`gains_displacements.json`、`parent_trace.json`、`analysis_summary.json`、`test_results.json`、`web_start.json`、`web_check.json`、`qdrant_before.json`／`qdrant_after.json`。`baseline_source/` 保留本轮修改前的5份核心源文件，供追溯使用，不替代正常版本管理。')
    add('### 11.3 前期案例建库与MySQL接入文件（本轮复用）')
    add(table(['文件／目录（完整路径）','已实现用途'],[
        [link('scripts/chunk_case_pdfs.py'),'174份PDF提取正文，生成父子块与来源元数据'],
        [link('scripts/embed_json_children_ali.py'),'读取配置、每批最多10条、断点续跑生成向量CSV'],
        [link('scripts/upload_child_embeddings_to_qdrant.py'),'按 --collection 明确分库存储，支持 --dry-run'],
        [link('knowledge_data/json_cases'),'174份案例切分JSON；349父块、2334子块'],
        [link('knowledge_data/embedding_csv_cases'),'2334条案例子块的1024维向量'],
        [link('knowledge_data/json'),'128份法规JSON；511父块、1384子块'],
        [link('scripts/import_parent_chunks_to_mysql.py'),'utf-8-sig、关联检查、内容冲突检查、幂等导入'],
        [link('scripts/mysql_schema.sql'),'documents／parent_chunks兼容表结构，完整正文LONGTEXT，字符串ID'],
        [link('scripts/setup_mysql_local.py'),'前期本地安全输入root密码、建立服务及项目账号；本轮不再执行'],
        [link('scripts/verify_mysql_parent_store.py'),'前期MySQL与真实双库回补验证'],
        [link('Refinement and Optimization/MySQL本地配置与父块入库验证记录.md'),'MySQL安装、导入与验证历史说明']]))
    add('这些文件和知识库是前期已完成内容，本轮对它们的复用不表示再次生成向量或再次上传数据。`requirements.txt` 中的 `pymupdf` 等依赖来自前期案例接入。不要将包含任何凭据的配置或 MySQL 选项文件交到 Git。')
    add('## 12. 小组成员怎样复现')
    add('### 12.1 使用当前已准备好的本机环境')
    add('以下命令在 Windows PowerShell、项目根目录执行。先使用本机已有 `.venv`，不需要再次安装 MySQL、生成案例向量或上传 Qdrant。')
    add('```powershell\nSet-Location "G:\\MSBD5018_agentic-rag-law-main"\nGet-Service MySQL84LawRag\n# 如果服务未运行，在管理员PowerShell执行：\nStart-Service MySQL84LawRag\n```')
    add('检查依赖与离线单元测试：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s eval -p "test_*.py"\n.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s scripts -p test_mysql_setup.py\n```')
    add('验证当前 MySQL 的完整父块与禁止回退：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\preflight_ablation.py\n```\n\n正常应看到 `expected_parents=860`、`runtime_json_fallback_calls=0`。该命令会更新前置核验记录；要保留本次实测耗时，先复制结果目录留档。')
    add('无需再次付费重排的本地复现：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\run_ablation_4way.py --replay\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\report_ablation_4way.py\n```\n\n`--replay` 要求缓存完整且签名一致；缺缓存就停止，不会偷偷发新模型请求。仍需本机 MySQL 可连接，保证父块回补再次验证。若只重新生成文档，单独运行第二条即可。')
    add('真实请求和断点续跑：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\run_ablation_4way.py\n```\n\n脚本检查已有请求缓存；输入、候选、模型、top_n的签名一致时复用，不重复付费。若想完全重新测量，应先将整个 `eval/results_ablation_20` 目录另名留档，再先跑前置核验、后跑主脚本。不要只删其中几个结果却仍把它叫全新运行；更不要改真值后混用旧缓存。')
    add('### 12.2 其他机器从原始资料重建时')
    add('先准备 Python 3.10+、依赖和各自本地凭据，不共享本机密码。项目路径变化时按实际目录修改。案例切分与向量生成示例：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 scripts\\chunk_case_pdfs.py --input-dir knowledge_data\\劳动案例 --output-dir knowledge_data\\json_cases\n# 只有确实需要重建向量时再执行下条，会产生API费用；有现成CSV应复用。\n.\\.venv\\Scripts\\python.exe -X utf8 scripts\\embed_json_children_ali.py --json-dir knowledge_data\\json_cases --output-dir knowledge_data\\embedding_csv_cases --batch-size 10\n```')
    add('上传前先 dry-run，并明确集合名，防止两类数据进入同一个集合：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 scripts\\upload_child_embeddings_to_qdrant.py --json-dir knowledge_data\\json --csv-dir knowledge_data\\embedding_csv --collection law_statutes --dry-run\n.\\.venv\\Scripts\\python.exe -X utf8 scripts\\upload_child_embeddings_to_qdrant.py --json-dir knowledge_data\\json_cases --csv-dir knowledge_data\\embedding_csv_cases --collection law_cases --dry-run\n```\n\n上面只展示校验命令；本轮没有执行上传。其他机器正式入库应先核对各自集合状态与授权，不使用 `--recreate`。')
    add('父块 dry-run：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 scripts\\import_parent_chunks_to_mysql.py --json-dir knowledge_data\\json --dry-run\n.\\.venv\\Scripts\\python.exe -X utf8 scripts\\import_parent_chunks_to_mysql.py --json-dir knowledge_data\\json_cases --dry-run\n```\n\n建库需具有相应权限的本地管理员；日常项目账号不执行 DDL。实际导入可以用 `--ask-password` 在本地隐藏输入，或复用安全脚本从 YAML 在进程内读取项目密码；不要把密码写入命令行、报告或聊天。导入脚本默认不会自动读取 YAML，不能只填好 YAML 就假设 CLI 连接正确。已存在数据会先检查冲突，同ID不同内容不能静默覆盖。')
    add('### 12.3 网页启动与停止')
    add('```powershell\nSet-Location "G:\\MSBD5018_agentic-rag-law-main"\n.\\.venv\\Scripts\\python.exe -X utf8 app.py\n```\n\n浏览器打开 `http://127.0.0.1:5000`。修改生产代码后需重启 Flask，开发服务器不会自动加载本轮变动；原窗口按 `Ctrl+C` 停止，再运行命令。MySQL 是独立 Windows 服务，可继续运行。确需停止数据库时在管理员 PowerShell 执行 `Stop-Service MySQL84LawRag`，先确认没有本项目任务正在读库。')
    if (OUT/'web_check.json').exists():
        add('本次已将本项目原后台Flask进程重启以加载新代码，并验证首页 HTTP 200；没有为网页验证额外生成答案。当前后台运行记录在 `web_start.json`。当前实例没有前台窗口，可用下面命令停止端口5000的进程，但必须先核对它仍属于本项目：\n\n```powershell\n$ragListener = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction Stop\n$ragProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($ragListener.OwningProcess)"\nif ($ragProcess.CommandLine -like "*G:\\MSBD5018_agentic-rag-law-main\\app.py*") {\n    Stop-Process -Id $ragListener.OwningProcess\n} else {\n    Write-Host "端口进程不属于该项目，请先核对，不自动停止。"\n}\n```')
    add('## 13. 验证范围和未完成项')
    add('本轮完成了真实外部服务调用、20题80组结果、MySQL全量及禁回退核验、来源配额单元测试、历史配置保护、Qdrant前后只读状态核对。Qdrant两集合数量与向量配置前后一致，脚本只使用GET集合信息和POST搜索接口；并未依赖“数量不变”来宣称检查了每一个远端向量的内容哈希。')
    add('当前已运行的离线回归测试共15项：原风洞5项、新配额7项、MySQL设置3项，全部通过。测试中的合成分数只用于单元测试，主实验分数全部来自真实Rerank。保存指标已从逐题排名独立重新计算并与结果表核对一致。')
    add('仍未做：完整60题盲测、统计显著性检验、独立人工补齐相关真值、答案生成／审查／引用正确率评测、软保底阈值实验、仅最终证据配额消融。当前20题能定位具体损失和选择机制，不能证明普遍最优配额或完整法律问答质量提升。')
    content='\n'.join(lines)
    # Fail closed if any locally configured credential unexpectedly appears in the report.
    from rag.config import AppConfig
    raw=AppConfig.load().raw
    for section in raw.values():
        for key,value in section.items():
            if ('password' in key or 'api_key' in key) and value and len(str(value))>=6 and str(value) in content:
                raise ValueError('Credential leak detected before report write')
    REPORT.write_text(content,encoding='utf-8')
    print(f'Report written: {REPORT}; all stored metrics independently reconciled')


if __name__=='__main__':
    try:run()
    except Exception as exc:
        print('report_failed: '+type(exc).__name__);sys.exit(1)
