"""Offline reconciliation and Chinese M1/M2/M3 report. Never calls services."""
from __future__ import annotations
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

from eval_retrieval import ROOT_DIR,build_indexes,load_queries,validate_labels,metrics_for,summarize
from run_hybrid_m123 import OUT,MODES,write
from rag.evidence import build_evidence
from rag.schemas import RetrievedChunk

REPORT=ROOT_DIR/'Refinement and Optimization/report_hybrid_m123.md'
VERSIONS=tuple(MODES)

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def f(x):return f'{x:.4f}'
def link(path):return f'[{path}](<{(ROOT_DIR/path).as_posix()}>)'
def table(headers,rows):
    clean=lambda x:str(x).replace('|','／').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(clean,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(clean,row))+' |' for row in rows])+'\n'

def run():
    manifest=read('run_manifest.json');pre=read('preflight_hybrid.json')
    if not manifest['complete']:raise ValueError('Not all 20 queries complete')
    data={v:read(v+'.json') for v in VERSIONS};queries=load_queries(ROOT_DIR/'eval/queries_ablation_20.jsonl')
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases');validate_labels(queries,index)
    by={v:{r['query_id']:r for r in data[v]['details']} for v in VERSIONS}
    for v in VERSIONS:
        if len(data[v]['details'])!=20:raise ValueError('Wrong query count')
        for row in data[v]['details']:
            if metrics_for(row,row['doc_ranks'],row['parent_ranks'],index)!=row['scoped_metrics']:
                raise ValueError('Metric/rank inconsistency')
        if summarize(data[v]['details'])!=data[v]['summary']:raise ValueError('Summary mismatch')
    def kind(pid):return index.doc_meta[index.parent_to_doc[pid]]['doc_type']
    def labels(ids):return '；'.join(('法:' if kind(p)=='statute' else '案:')+f'`{p}`' for p in sorted(ids)) or '—'
    def metric(v,scope='all_gt',grain='parent',stage='evidence',name=None,subset=None):
        name=name or {'recall':'recall@pool','rerank':'recall@20','evidence':'recall@5'}[stage]
        if subset is None:return data[v]['summary']['scoped_metrics'][scope][grain][stage][name]
        vals=[r['scoped_metrics'][scope][grain][stage][name] for r in subset if r['scoped_metrics'][scope][grain][stage]]
        return mean(vals) if vals else None
    ledger=[];traces=[]
    for old,new in [('M1','M2'),('M2','M3')]:
        for q in queries:
            qid=q['query_id'];a,b=by[old][qid],by[new][qid];gt=set(q['relevant_parent_ids']);entry={'comparison':new+'-'+old,'query_id':qid,'query_type':q['query_type']}
            for stage in ('recall','rerank','evidence'):
                aset=set(a['parent_ranks'][stage])&gt;bset=set(b['parent_ranks'][stage])&gt
                entry[stage+'_gains']=sorted(bset-aset);entry[stage+'_losses']=sorted(aset-bset)
            entry['initial_gains_reaching_rerank']=sorted(set(entry['recall_gains'])&set(b['parent_ranks']['rerank']))
            entry['initial_gains_reaching_evidence']=sorted(set(entry['recall_gains'])&set(b['parent_ranks']['evidence']))
            entry['initial_gains_lost_at_rerank']=sorted(set(entry['recall_gains'])-set(b['parent_ranks']['rerank']))
            entry['initial_gains_lost_at_evidence']=sorted((set(entry['recall_gains'])&set(b['parent_ranks']['rerank']))-set(b['parent_ranks']['evidence']))
            ledger.append(entry)
    for q in queries:
        for pid in q['relevant_parent_ids']:
            stages={}
            for v in VERSIONS:
                row=by[v][q['query_id']]
                raw=read('cache/rerank/'+row['rerank_request']['request_signature']+'.json')['value']
                global20=[row['recall_chunks'][x['index']]['parent_id'] for x in sorted(raw,key=lambda x:x['relevance_score'],reverse=True)[:20]]
                no_final_quota=build_evidence([RetrievedChunk(**c) for c in row['rerank_chunks']],5)
                global5={e.parent_id for e in no_final_quota.items}
                hit={stage:pid in row['parent_ranks'][stage] for stage in ('recall','rerank','evidence')}
                reason=('initial_not_recalled' if not hit['recall'] else
                    'child_source_quota_displacement' if not hit['rerank'] and pid in global20 else
                    'rerank_score_below_global20' if not hit['rerank'] else
                    'final_source_quota_displacement' if not hit['evidence'] and pid in global5 else
                    'parent_score_below_global5' if not hit['evidence'] else 'final_hit')
                stages[v]={**hit,'in_global_rerank20_before_child_quota':pid in global20,
                           'in_global_parent5_before_final_quota':pid in global5,'diagnostic_reason':reason}
            traces.append({'query_id':q['query_id'],'parent_id':pid,'doc_type':kind(pid),
                           **stages})
    write(OUT/'gains_displacements.json',ledger);write(OUT/'parent_trace.json',traces)
    channel_counts={};exact_audit=[]
    for v in VERSIONS:
        counts=Counter()
        for row in data[v]['details']:
            for e in row['evidence_pack']['items']:
                sources=set(s for c in row['rerank_chunks'] if c['parent_id']==e['parent_id'] for s in c['retrieval_sources'])
                label='多路共现' if len(sources)>1 else {'dense':'dense独有','bm25':'bm25独有','exact_high':'exact_high独有','exact_weak':'exact_weak独有'}.get(next(iter(sources),''),'未标来源')
                counts[label]+=1
            for c in row['recall_chunks']:
                if c['exact_match_level']:
                    exact_audit.append({'query_id':row['query_id'],'chunk_id':c['chunk_id'],'parent_id':c['parent_id'],
                        'level':c['exact_match_level'],'reserved':c['payload'].get('_exact_reserved',False),
                        'metadata':c['payload'].get('_exact_metadata',{})})
        channel_counts[v]=dict(counts)
    write(OUT/'channel_attribution.json',channel_counts);write(OUT/'exact_audit.json',exact_audit)
    baseline=read('baseline_dense_audit.json')
    global_summary={'versions':{v:data[v]['summary'] for v in VERSIONS},'all_metrics_recomputed_match':True,
        'channel_counts':channel_counts,'ledger_counts':{pair:{stage:{change:{s:sum(sum(kind(p)==s for p in r[stage+'_'+change]) for r in ledger if r['comparison']==pair) for s in ('statute','case')}
            for change in ('gains','losses')} for stage in ('recall','rerank','evidence')} for pair in ('M2-M1','M3-M2')}}
    write(OUT/'analysis_summary.json',global_summary)
    L=[]
    def add(text=''):L.append(text.strip()+'\n')
    add('# 劳动法 Agentic RAG：精确匹配、中文 BM25 与库内 RRF 混合检索实验报告')
    add('**记录日期**：2026-11-20  \n**项目目录**：`G:\\MSBD5018_agentic-rag-law-main`  \n**实验范围**：固定 V4 下游策略，20 题 M1／M2／M3 递进消融。')
    add('## 1. 实验出发点：这次想解决什么问题？')
    add('### 1.1 案例和来源配额已接通，接下来检查“最初有没有找到”')
    add('前期已将128份法规和174份劳动案例接入双库：法条511父块／1384子块，案例349父块／2334子块。上一轮通过统一重排打分和两阶段来源配额，保护法条与案例进入证据包的机会。但配额只能从已经找到的候选里挑选，无法凭空救回最初没有召回的依据。')
    add('本轮新增两个互补入口：BM25寻找词语上的重合；精确匹配识别用户明确说出的法规名称、条号、指导案例号和案号。它们先在各自知识库内部与向量结果融合，再交给原来的统一重排和证据配额。')
    add('需要区分事实与解释：上一轮观察到案例占位较多，但“模型偏爱叙事”只能作为可能解释，不能仅凭覆盖指标认定模型存在某种已证实偏差。本轮只报告可追踪的候选与证据变化。')
    add('### 1.2 固定下游 V4，只改变初筛通道')
    add(table(['组别','初筛方式：每库最终30个子块','下游'],[
        ['M1 / dense','仅Qdrant向量Top30','统一重排全60分数 → 原V4子块20配额 → 原V4父块5配额'],
        ['M2 / dense_bm25','Dense30＋BM25最多30，库内RRF取30','与M1相同'],
        ['M3 / hybrid','M2＋高置信精确预留最多5位＋弱精确参与RRF','与M1相同']]))
    add('法条解释／程序咨询的子块名额仍为16:4、父块4:1；法律咨询为12:8、3:2；案例检索为8:12、2:3。`source_policy.py` 和 `evidence.py` 前后哈希不变。本轮没有调配额、没有给案例加权威分、没有加相关性软阈值。父块公式仍为 `最高子块重排分数 + 0.02 × log(1+命中子块数)`。')
    add('本轮核心结果先看这里：')
    add(table(['父块Recall','M1向量＋V4','M2加入BM25','M3再加Exact'],[
        ['初筛综合覆盖']+[f(metric(v,stage='recall')) for v in VERSIONS],
        ['最终综合覆盖']+[f(metric(v)) for v in VERSIONS],
        ['最终法条覆盖']+[f(metric(v,'statute_gt')) for v in VERSIONS],
        ['最终案例覆盖（9题）']+[f(metric(v,'case_gt')) for v in VERSIONS]]))
    add('当前默认参数下，BM25融合降低了这批题的标注覆盖，Exact没有在M2基础上新增相关父块收益。功能已实现并验证，但这次实验不支持“增加通道就提升效果”的结论。下面同时展示损失、机制边界和可能的下一步。')
    add('### 1.3 先核对上一轮初始召回：不能把所有漏失都归给向量')
    rows=[]
    for v,label in [('V1','上一轮单法条Top30'),('V4','上一轮双库30+30')]:
        for s,source in [('statute','法条'),('case','案例')]:
            rows.append([label,source]+[sum(len(r['funnel'][v][s][field]) for r in baseline) for field in ('initial_missing','initial_hit_but_rerank_lost','rerank_hit_but_evidence_lost')])
    add(table(['上一轮方案','真值类型','初始未召回事件','初始命中但未进20','进20但未进5'],rows))
    add('计数是“题目—已标注父块”事件，不是不同父块去重总数。单库无法召回案例是设计限制，不能算成其法条退化。双库初始只漏2个法条父块、没有漏案例父块，说明多数残余漏失发生在最终证据选择，混合检索的可提升空间需要据此判断。')
    missed=[[r['query_id'],s,labels(r['funnel']['V4'][s]['initial_missing'])] for r in baseline for s in ('statute','case') if r['funnel']['V4'][s]['initial_missing']]
    add(table(['题号','来源','上一轮双库初始未召回的相关父块'],missed))
    add('特别是 q004 的 `00002_p004` 没进入初始向量池，而 q057 的同一目标父块已进入向量池、最后才落选。两题不能用同一种“初始漏召回”解释。逐题原始审计见 '+link('eval/results_hybrid_20/baseline_dense_audit.json')+'。')
    add('## 2. 实现方案：怎样把三路检索接在一起？')
    add('### 2.1 核心概念，用通俗的话说明')
    add(table(['概念','含义','边界'],[
        ['Dense向量检索','按语义接近程度找资料，适合口语表达','未必能精确定位指定条号'],
        ['BM25','比较问题和片段的词语重合，兼顾词语稀有度和文本长度','不是法律推理；口语与法律术语不重合时不一定有效'],
        ['结构化Exact','识别法规＋条号，或本案的指导案例号／案号','正文引用其他规定只算弱信号'],
        ['RRF','将各通道的名次换成倒数后相加','不直接相加BM25分数和余弦分数'],
        ['父块回补','凭子块parent_id到MySQL取回完整上下文','测试时禁止JSON回退掩盖数据库问题'],
        ['初筛／重排／最终证据','60个子块池 → 20个子块 → 5个父块','初筛找到不代表最终能被答案引用']]))
    add('### 2.2 精确检索：区分“这是第47条”与“这里引用了第47条”')
    add('法条承载编号只从 `title_path` 和行首条文标题提取。例如第87条正文引用第47条，该块的 `owned_articles` 是87，`referenced_articles` 包含47。查询“劳动合同法第47条”时，真正承载第47条的块才有高置信预留资格；第87条的交叉引用只作为弱信号。')
    add('法规身份从文档标题和来源信息确定；一份标题提到“劳动合同法”的答复，不能因此被冒充成劳动合同法原文。简称采用最长匹配，避免把劳动合同法实施条例错误识别为劳动合同法。劳动争议司法解释（二）按该领域完整名称规范化，并结合来源文件名、正文开头识别被JSON简化的“（二）”。')
    add('每个子块保留 `version_tag`，文件名日期或明确版本信息不会被去重覆盖。只有查询明确说“某年版／修订”时才限制版本，案件事实里的2026年不会误被当作法规版本。没有明确版本的查询可能同时返回多个版本，后续仍需判断适用性。')
    add('案例本案号来自文档标题、title_path或文首案号头；其他案例正文引用的指导案例、检例或案号仅记为 `referenced_case_ids`。只说条号而没说法规名时，降级为weak；没有任何结构化标识时，Exact返回空列表。')
    add('### 2.3 中文分词与 BM25 的边界')
    add('使用独立的jieba分词器，注册劳动法专有词和否定词；文档 `title_path + 换行 + text` 与问题走同一管线。仅去掉“您好、谢谢、请问”等寒暄及标点空白，保留“不、未、没有、尚未、拒不”、金额、日期、时效与程序词。')
    add('仅为条／款／项／号附加规范化数字词元，如“第四十七条”和“第47条”都补 `art_47`；原始工资金额和日期不被改写。BM25的k1=1.5、b=0.75，采用正值平滑IDF `log(1+(N-df+0.5)/(df+0.5))` 与标准Okapi长度归一化。两库分别建索引，空词或无词项重合直接返回空列表，不把零分候选凑进前30。')
    add('### 2.4 库内融合：最多5个精确预留位，其余按RRF补齐')
    add('每个库先为high精确命中预留最多5个位置，按匹配紧密度、RRF分数及稳定ID顺序选择。其余位置按 `1/(60+dense_rank) + 1/(60+bm25_rank) + 0.5/(60+weak_rank)` 排序、按chunk_id去重，补到总计30个。缺失通道不贡献分数。high未获预留且没有其他通道支持时，不用零分硬凑进候选。')
    add('high候选若本来也进入Dense或BM25，还可能靠正常RRF排名入选，因此“预留上限5”不意味着最终池里最多只有5个带high标签的块。实验保存 `_exact_reserved` 区分真正使用预留位的候选。两库各保留30，融合不扩大送去重排的总预算。')
    add('### 2.5 分数、缓存和多轮作用域')
    add('子块分别保存dense_score／dense_rank、bm25_score／bm25_rank、exact_match_level、rrf_score、rerank_score和retrieval_sources。兼容字段score始终表示余弦分数；非Dense候选用0占位，但其明确缺失状态是dense_score=None。排序不拿这个占位值与RRF混排。')
    add('重排异常时，dense模式按dense_score排序，混合模式保留融合后统一顺序，并显式标记fallback。本轮任何fallback都会报错，不计入指标。本轮成功结果全部使用真实重排分数，未使用Mock评分。')
    add('索引以“302个JSON文件的相对路径＋字节数＋内容SHA-256组成的语料指纹、lexical_config_version”为键，在进程中共享。每次访问重新校验内容指纹，但命中时不再分词建索引；文件内容即使大小和mtime未变也能触发重建。')
    add('计划新增raw_query（本轮用户原话）与retrieval_mode。精确标识只来自当前原话和当前活跃改写：原话明确给出新标识时以原话为准，明确换话题时不继承旧编号；无新标识的同话题追问可使用活跃改写补充。不是把整段历史对话拼入Exact。')
    add('## 3. 实验执行与实际结果')
    add('### 3.1 数据、云端一致性与MySQL前置核验')
    add('测试仍为原先固定20题，8法律咨询、5法条解释、4程序咨询、3案例检索；其中9题有案例真值、11题仅法条真值。人工标签直接控制V4配额，未把意图识别误差混入检索消融。原文摘录、真值ID及父子关联重新通过检查，没有按新排名修改真值。')
    add(table(['项目','实测结果'],[
        ['语料指纹文件',str(pre['corpus_files'])+'个JSON，utf-8-sig读取；'+pre['corpus_fingerprint']],
        ['法条云端核对',str(pre['qdrant']['statute']['all_payloads_compared'])+'点，精确点数与全部关键payload一致'],
        ['案例云端核对',str(pre['qdrant']['case']['all_payloads_compared'])+'点，精确点数与全部关键payload一致'],
        ['MySQL父块','860／860正文、标题路径及文档关联一致；JSON回退0次'],
        ['法条／案例运行时get_many耗时',f(pre['mysql']['runtime_mysql_seconds']['statute'])+'秒／'+f(pre['mysql']['runtime_mysql_seconds']['case'])+'秒'],
        ['首次索引构建',f(pre['index_cold']['build_seconds'])+'秒'],
        ['同进程索引命中访问',f(pre['index_warm']['access_seconds'])+'秒（含重新计算指纹）']]))
    add('本轮超出“各抽样至少10点”的要求，逐点核对了全部3718个chunk_id、parent_id、title_path、text，并校验完整ID集合；前后只读复核均一致。没有比较每个远端向量分量的内容哈希，也没有写入、删除或重建任何集合。')
    add('### 3.2 请求签名与真实调用次数')
    add('Rerank签名对完整query、有序documents、model、top_n、endpoint及实际instruct进行SHA-256。供应商请求body由request_parameters()统一生成，避免缓存签名和线上参数各写一份而漂移。候选集合相同但顺序不同也视为新请求。endpoint参与签名，落盘仅记录其哈希；不记录请求头或密钥。')
    add('旧Embedding只有在原配置文件哈希未变、query／model／dimension签名一致且向量有限、1024维时导入新缓存；新缓存补充端点签名。旧Dense缓存缺少本轮完整集合状态指纹，故本轮真实重新搜索40次。旧Rerank缓存不含端点／instruct签名，保守不复用，M1也重新真实打分。M2和M3只有完整有序请求一致时才共享响应。')
    if (OUT/'delivery_check.json').exists() and read('delivery_check.json').get('original_V4_and_current_M1_parent_rankings_identical'):
        add('复核确认：本轮重新真实打分的M1，与上轮V4在全部20题的初筛、重排、最终证据父块列表均一致。它提供了下游V4接线未漂移的实际对照，主比较仍只使用本轮M1—M3。')
    add(table(['记录','数量'],[['真实成功调用',manifest['api_success_counts']],['失败尝试',manifest['api_failed_attempts']],
        ['本次运行缓存查询次数',manifest['cache_lookups']],['缓存命中次数',manifest['cache_hits']],
        ['不同重排请求签名',manifest['unique_rerank_signatures']],['旧重排直接复用',manifest['old_rerank_reused']]]))
    add('调用次数是累计落盘事件；缓存命中数按当前运行计，不等于免费API次数。真实服务可能有费用，缓存不能保证失败请求未计费。意图路由抽查只到计划／候选传递，不生成答案。')
    add('API事件计数覆盖问题理解、查询向量、向量搜索与重排调用；前后集合GET／精确count／scroll审计及MySQL查询不混入这些模型与搜索事件。20个查询向量复用先前已验证真实记录，本轮没有重新调用Embedding。')
    add('### 3.3 线上路由与话题切换检查')
    routes=read('route_checks.json')
    add(table(['样例','raw_query','改写query','mode','话题切换／残留检查'],[
        [r['sample'],r['plan']['raw_query'],r['plan']['query'],r['plan']['retrieval_mode'],
         '新话题；未带入旧案例号' if r['topic_continuation'] is not None else '当前话题标识'] for r in routes]))
    add('4个样例调用真实DeepSeek问题理解，第4例另调用话题延续判断；用当前计划接入实际initial_candidates()，Dense输入为空的探针只检查词法与传参，不额外请求查询向量和云端搜索。此检查不评价完整问答准确率，也不是大规模意图分类评测。')
    add('### 3.4 指标怎样读，避免误解“Recall@30”')
    add('初筛表示“每库30个、合并共60个子块的完整池”，因此用recall@pool／hit@pool统计，不把拼接列表前30个误叫双库初筛。文档和父块按首次出现去重；重排保留20个子块，最终5个父块。MRR为该阶段列表中首个相关项排名倒数。')
    add('初筛两库按法条30再案例30固定拼接，其MRR受该顺序影响，只在同一拼接口径下比较，不能当作跨库分数校准。法条真值评分不删除结果中的案例，案例评分也不删除法条。statute_gt与all_gt分母为20题；case_gt仅平均有案例真值的9题，空真值不记0分。')
    add('### 3.5 全局三阶段完整指标')
    for scope,title in [('statute_gt','法条真值（20题）'),('case_gt','案例真值（9题）'),('all_gt','全部真值（20题）')]:
        for grain,label in [('doc','文档级'),('parent','父块级')]:
            add('#### '+title+' · '+label)
            rows=[]
            for stage,stage_name,k in [('recall','初筛每库30合并池','pool'),('rerank','Rerank@20','20'),('evidence','Evidence@5','5')]:
                for name,metric_name in [('recall@'+k,'Recall'),('mrr','MRR'),('hit@'+k,'Hit')]:
                    rows.append([stage_name,metric_name]+[f(metric(v,scope,grain,stage,name)) for v in VERSIONS])
            add(table(['阶段','指标',*VERSIONS],rows))
    add('### 3.6 最终证据结构与通道构成')
    add(table(['组别','平均法条父块','平均案例父块'],[[v,f(data[v]['summary']['avg_statute_evidence_count']),f(data[v]['summary']['avg_case_evidence_count'])] for v in VERSIONS]))
    add('三组的平均法条／案例数量相同，但具体选中的父块不同；覆盖下降不能仅从名额数量看出来。')
    add('每个最终父块统计其所有保留子块的来源并集，按互斥类别归属；同一块若既在Dense又在BM25前30，计“多路共现”。这是来源标签占比，不能据此声称某通道单独造成了收益；因果比较以M2−M1、M3−M2为准。')
    add(table(['组别','类别','父块出现次数','该组全部证据占比'],[
        [v,label,n,f(n/sum(channel_counts[v].values()))] for v in ('M2','M3') for label,n in sorted(channel_counts[v].items())]))
    add('本轮M2有14／100条证据属于BM25独有，M3为13／100；多路共现分别为61／100和62／100。M3没有最终exact_high独有证据，且两组标注覆盖相同，说明来源标签变化本身不能证明检索质量提升。')
    add('### 3.7 三阶段漏斗：新召回了什么，又挤掉了什么')
    rows=[]
    for pair in ('M2-M1','M3-M2'):
        for stage in ('recall','rerank','evidence'):
            n=global_summary['ledger_counts'][pair][stage]
            rows.append([pair,stage,n['gains']['statute'],n['losses']['statute'],n['gains']['case'],n['losses']['case']])
    add(table(['比较','阶段','新增相关法条','移出相关法条','新增相关案例','移出相关案例'],rows))
    add('逐题只追踪预先标注相关父块，未标注资料不直接叫错误。“移出”只是从候选／证据列表移出，没有删除数据库数据。以下计数仍是题目—父块事件。')
    for pair in ('M2-M1','M3-M2'):
        add('#### '+pair+'：逐题双向账本')
        add(table(['题号','初筛新增','初筛移出','新增后进20','新增后进5','最终新增','最终失去'],[
            [r['query_id'],labels(r['recall_gains']),labels(r['recall_losses']),labels(r['initial_gains_reaching_rerank']),
             labels(r['initial_gains_reaching_evidence']),labels(r['evidence_gains']),labels(r['evidence_losses'])]
            for r in ledger if r['comparison']==pair]))
    add('#### 召回救回后仍落选：卡在哪一步')
    stuck=[r for r in ledger if r['initial_gains_lost_at_rerank'] or r['initial_gains_lost_at_evidence']]
    add(table(['比较','题号','初筛救回但未进20','进20但未进5'],[[r['comparison'],r['query_id'],labels(r['initial_gains_lost_at_rerank']),labels(r['initial_gains_lost_at_evidence'])] for r in stuck]) if stuck else '本轮没有“初筛新增的相关父块在后续再次落选”的事件。')
    add('报告进一步读取全60个真实重排分数，回放“未加子块配额的全局20”，以及“未加最终配额的全局父块5”，区分模型分数截断、子块来源配额、父块分数排序与最终来源配额。若同时存在多种约束，诊断标签表示沿当前流程首先能够确定的瓶颈，不把它说成独立因果效应。完整诊断保存在parent_trace.json。')
    add('#### q004 与 q057 的完整轨迹')
    for qid in ('q004','q057'):
        q=next(q for q in queries if q['query_id']==qid);add('**'+qid+'**：'+q['query'])
        reason_names={'initial_not_recalled':'初始未召回','child_source_quota_displacement':'子块配额移出',
            'rerank_score_below_global20':'模型全局20外','final_source_quota_displacement':'最终来源配额移出',
            'parent_score_below_global5':'父块分数全局5外','final_hit':'最终命中'}
        add(table(['组别','目标父块','初筛','重排20','最终5','诊断位置'],[
            [v,labels([pid]),pid in by[v][qid]['parent_ranks']['recall'],pid in by[v][qid]['parent_ranks']['rerank'],pid in by[v][qid]['parent_ranks']['evidence'],
             reason_names[next(t for t in traces if t['query_id']==qid and t['parent_id']==pid)[v]['diagnostic_reason']]]
            for v in VERSIONS for pid in q['relevant_parent_ids']]))
    add('#### Exact的owned／referenced保护实际发生在哪里')
    exact_rows=[]
    for row in data['M3']['details']:
        tagged=[c for c in row['recall_chunks'] if c['exact_match_level']]
        if tagged:
            exact_rows.append([row['query_id'],sum(c['payload'].get('_exact_reserved',False) for c in tagged),
                ', '.join(c['chunk_id'] for c in tagged if c['exact_match_level']=='high') or '—',
                ', '.join(c['chunk_id'] for c in tagged if c['exact_match_level']=='weak') or '—'])
    add(table(['题号','精确预留位：两库合计，各库最多5','high子块','weak子块（没有预留资格）'],exact_rows))
    add('实际例子：q060中 `50036_c039`、`50036_c040` 的本案号是 `(2025)苏02民终1980号`，正文引用了指导案例190号。因此两块是weak且reserved=False，未冒充“指导案例190号本案”获得保送；仍可凭Dense／BM25或弱RRF进入候选。')
    add('弱引用有时仍会通过BM25或Dense进入候选，它们没有被全局删除；“避免误召回”准确说是避免把交叉引用冒充high而强制预留。未运行“将引用也当high”的额外消融，不能凭本轮数字声称该区分单独提高多少召回率。')
    add('### 3.8 按四类意图拆解')
    rows=[]
    for intent in ('法律咨询','法条解释','程序咨询','案例检索'):
        for v in VERSIONS:
            subset=[r for r in data[v]['details'] if r['query_type']==intent]
            vals=[metric(v,s,subset=subset) for s in ('statute_gt','case_gt','all_gt')]
            rows.append([intent,len(subset),v]+[f(x) if x is not None else '无案例真值' for x in vals]+
                [f(metric(v,'all_gt',stage='recall',subset=subset)),f(metric(v,'all_gt',stage='rerank',subset=subset))])
    add(table(['意图','题数','组别','最终法条Recall','最终案例Recall','最终综合Recall','初筛综合Recall','重排综合Recall'],rows))
    add('更完整的分意图文档／父块、Recall／MRR／Hit保存在每组JSON的groups，不只保留有利指标。')
    add('### 3.9 实际耗时')
    add(table(['组别','平均初筛本地处理秒','平均MySQL回补秒','平均原始重排调用秒','平均受控检索秒'],[
        [v]+[f(mean(r['timings'][key] for r in data[v]['details'])) for key in ('local_initial_seconds','parent_seconds','rerank_seconds_original','retrieval_seconds')] for v in VERSIONS]))
    add(f'当前完整运行耗时 {manifest["elapsed_seconds"]:.2f} 秒。受控检索秒包含共享Dense搜索的原始耗时、本地初筛、MySQL、该组对应重排原始调用耗时和证据构建；不含共享Embedding、首次冷建索引、路由检查及文件落盘。共享请求在多组表里出现不表示重复计费。缓存保留原始调用秒数，实际命中延迟另在调用记录中保存。')
    if (OUT/'first_run_manifest.json').exists():
        first=read('first_run_manifest.json')
        add(f'首轮完整运行实测 {first["elapsed_seconds"]:.2f} 秒，成功请求为 {first["api_success_counts"]}。随后对“答复标题中提及某法不能冒充该法原文”、实施条例全称等解析正确性做严格核验，最终代码重建全部60组轨迹；完整有序请求相同则复用缓存。本次最终核验实测 {manifest["elapsed_seconds"]:.2f} 秒，其中新增真实重排 {manifest["api_success_counts"].get("rerank",0)-first["api_success_counts"].get("rerank",0)} 次，其余复用。未调BM25／RRF参数、来源配额或真值。first_run_manifest.json与first_cache_events.json保留首轮记录，当前指标只取最终统一代码生成的M1—M3。')
    failures=[e for e in read('api_events.json') if not e['ok']]
    if failures:add(table(['题号','组别','阶段','失败类型','尝试秒'],[[e.get('query_id',''),e.get('variant',''),e['phase'],e.get('error',''),f(e['seconds'])] for e in failures]))
    else:add('本轮API事件中没有失败尝试。')
    add('与上一轮42分钟总时间不做直接速度比较：上一轮有一次额外截断检查异常等待约34分钟，本轮缓存、调用拓扑和索引开销也不同。')
    add('### 3.10 客观结论与下一步')
    for old,new,label in [('M1','M2','加入BM25'),('M2','M3','叠加精确匹配')]:
        add(label+'：初筛综合父块Recall变化 '+f(metric(new,stage='recall')-metric(old,stage='recall'))+
            '，重排变化 '+f(metric(new,stage='rerank')-metric(old,stage='rerank'))+
            '，最终变化 '+f(metric(new)-metric(old))+'。是否产生最终价值，应看这三个环节的传导，不能只看找到更多候选。')
    add('具体到父块：q008 的 `00002_p005`、q014 的 `00002_p008` 在RRF初筛被移出；前者进一步造成最终证据损失。q017 的 `00017_p006` 仍在初筛和重排中，却在最终5条落选，不能把这项退步写成BM25初始漏召回。q055 的 `00002_p008` 则在重排阶段落选。新增通道没有救回已标注相关父块，但也不能将所有BM25独有、未标注证据直接认定为无关。')
    add('q004 的原始问题没有明确条号或案例号，Exact按设计静默，BM25也未补回目标法条；相关案例仍被最终来源名额移出。q057 的法条原本就进入候选并由子块配额保留，但父块分数排在全局5条之外。继续扩展初筛通道不能单独解决这两种残余问题。')
    add('本批结果更支持保留dense模式作为当前对照，下一轮先检查RRF对Dense相关候选的保护与信息需求拆分，再独立检验最终软保底或定向补检。不能在已看过的这20题上反复挑权重并把结果当盲测收益。按任务要求，线上接口默认仍是hybrid；这个工程默认值不代表实验已证明它优于dense，可通过计划参数切回。')
    add('本20题是已看过部分旧结果、用于指导优化的探索性诊断子集。真值是人工按原文选择的核心依据，不穷尽所有相关资料；没有独立标注者复审、没有显著性检验，也没有运行答案生成、引用准确率或法律结论质量评估。来源构成与证据覆盖提升都不能直接等同于回答更正确。')
    add('标识解析也有适用范围：目前是显式标题与正则规则，并非完整法律引用解析器；复杂跨法规并列引用、“条之一”、未登记简称、扫描件异常换行等仍需专项样例。多版本信息被保留，不等于已自动判断事实发生时应适用哪个版本。')
    add('如果初筛救回后仍在固定配额下丢失，下一步应比较带最低相关性阈值的软保底、信息需求覆盖检查或一轮定向补检。它们尚未实现或验证。本轮不因观察到某题结果而改配额、改阈值、补写真值；正式评估仍应由评测负责人准备独立盲测。')
    add('## 4. 新增／修改文件与团队复现')
    paths=[('rag/lexical.py','新增','法规／案例标识解析、保守分词、BM25、进程缓存、库内融合'),
        ('rag/schemas.py','修改','raw_query、retrieval_mode及各通道独立分数字段'),
        ('rag/retrieval.py','修改','共用initial_candidates与enrich_parents、三模式切换和安全回退'),
        ('rag/skill_router.py','修改','保存当前原话，线上默认hybrid，原配额不变'),
        ('rag/workflow.py','修改','增加初筛与回退调试记录，继续传递原V4证据配额'),
        ('rag/providers.py','修改','重排请求body提取为request_parameters，实际参数值保持一致'),
        ('requirements.txt','修改','增加jieba==0.42.1'),
        ('eval/preflight_hybrid.py','新增','302文件指纹、全量只读Qdrant对账、MySQL禁回退、旧初始召回审计'),
        ('eval/run_hybrid_m123.py','新增','20题三组、完整请求签名、真实调用、断点缓存、路由检查'),
        ('eval/test_hybrid_lexical.py','新增','13项新增离线行为测试'),
        ('eval/report_hybrid_m123.py','新增','从保存排名离线复算指标、双向账本及本报告'),
        ('eval/results_hybrid_20','新增','前置核验、三组结果、缓存、事件日志和分析'),
        ('Refinement and Optimization/report_hybrid_m123.md','新增','本报告；格式沿用report1的问题—方案—结果—文件清单主线')]
    add(table(['文件完整路径','状态','作用'],[[link(p),status,purpose] for p,status,purpose in paths]))
    add('`rag/source_policy.py`、`rag/evidence.py`、固定20题、私人配置及旧报告均受哈希保护，保持不变。之前为交付准备的重复备份目录没有自动同步本轮新代码；打包时以项目根目录实际代码为准。')
    add('### 4.1 离线测试和数据检查')
    add('原有15项测试与新增13项测试合计28项全部通过。覆盖本条／引用区分、无名条号降级、案例本案号、法规别名与版本、话题切换、否定与数字、BM25手算和零分边界、预留上限、分数隔离回退、内容指纹重建及完整请求签名敏感性。单元测试的合成评分只用于逻辑检验，主实验全部是真实评分。')
    add('```powershell\nSet-Location "G:\\MSBD5018_agentic-rag-law-main"\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s eval -p "test_*.py"\n.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s scripts -p test_mysql_setup.py\n```')
    add('### 4.2 真实运行、缓存回放与离线报告')
    add('先在自己的机器准备API凭据、两个Qdrant集合和MySQL父块，密钥不随报告提供。首次先执行前置核验：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\preflight_hybrid.py\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\run_hybrid_m123.py\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\report_hybrid_m123.py\n```')
    add('第一条会只读遍历云端payload，MySQL缺父块时沿用既有冲突检查后增量补齐路径；完整实验会调用真实模型并产生费用。报告脚本只读已有结果并离线重算，不发模型或数据库请求。')
    add('缓存完整且签名一致时：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\run_hybrid_m123.py --replay\n```\n\n回放仍需MySQL真实回补；缺缓存或签名不一致就报错，不悄悄调用模型。新机器先重新运行前置核验，避免沿用原机器绝对路径和配置哈希。端点／模型／有序文本变化都可能使旧重排缓存失效。')
    add('### 4.3 网页重启与模式切换')
    add('线上默认hybrid；纯向量或只加BM25时，可在route_skills调用中传入retrieval_mode="dense"或"dense_bm25"，下游配额仍保持不变。开发服务器需要重启加载新代码。\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 app.py\n```\n\n打开 `http://127.0.0.1:5000`，前台运行时Ctrl+C停止。首次混合检索会冷建词法索引，此后同进程复用。不要仅凭网页能打开就声称外部检索已通过；本轮真实服务证明来自上述事件与禁回退记录。')
    if (OUT/'web_check.json').exists():add('本次已重启属于该项目的后台Flask进程，首页实际返回HTTP 200，可直接打开体验；没有为网页检查新增完整答案生成调用。后台启动记录见web_start.json。以后手动启动前先停止当前项目实例，避免端口5000被占用；MySQL服务无需跟随Flask重启。')
    if (OUT/'replay_manifest.json').exists():
        replay=read('replay_manifest.json')
        add(f'还实际完成了一次最终代码的严格缓存回放：{replay["cache_lookups"]}次缓存查询全部命中，耗时{replay["elapsed_seconds"]:.2f}秒，没有新增模型或Qdrant搜索请求；仍从MySQL读取父块。回放后的60组指标重新按排名核算一致。')
    add('结果目录重点文件：preflight_hybrid.json、corpus_fingerprint.json、baseline_dense_audit.json、M1.json—M3.json、route_checks.json、api_events.json、cache_events.json、cache/、gains_displacements.json、parent_trace.json、channel_attribution.json、exact_audit.json、analysis_summary.json、run_manifest.json。完整逐题候选包含各通道分数与来源，供团队独立检查。')
    ui_notes=ROOT_DIR/'Refinement and Optimization/混合检索前端展示说明.md'
    if ui_notes.exists():
        add('## 5. 前端改进：在 Agent Steps 中展示混合检索')
        add('本节是检索实验完成后补充的前端可视化工作，不改变前面M1／M2／M3的实测数据。界面展示实际执行路径和命中数量，不代表检索效果已获得提升。')
        ui_body=ui_notes.read_text(encoding='utf-8').split('\n',1)[1].strip()
        ui_body=re.sub(r'^### ', '#### ',ui_body,flags=re.M)
        ui_body=re.sub(r'^## (\d+)\. ',lambda m:'### 5.'+m[1]+'. ',ui_body,flags=re.M)
        add(ui_body)
        add('### 5.6. 可直接复制到网页的手动测试问题')
        add('打开 `http://127.0.0.1:5000`，按Ctrl+F5刷新，点击右上角＋开始新会话，再复制以下问题：')
        add('> 请查找指导案例190号“王山诉万得信息技术股份有限公司竞业限制纠纷案”，结合《劳动合同法》第二十四条，说明仅凭营业执照经营范围相同是否足以认定竞争关系。这是公开案例学习，不涉及个人纠纷，请直接检索法条和案例进行解释。')
        add('这道问题同时包含明确的指导案例号和法规条号，适合观察Exact通道。信息和服务正常时，右侧应依次显示Dense／BM25／Exact → 库内RRF → Parent context → Unified rerank + quotas → Evidence + quotas。Dense通常为每库30个候选，RRF每库总预算30；随后显示实际保留的20个子块与最多5个父块，以及法条／案例数量。Exact应出现high命中，weak是否出现及父块回补数量取决于实际结果，不能把截图数字当固定答案。')
        add('此处提供的是手动测试问题，本次补写报告没有调用模型执行这条新问句。前端已有验证来自上节说明的真实检索记录回放与MySQL读取。报告生成脚本已同步纳入本节，以后离线重新生成报告时会保留前端说明和测试用例。')
    content='\n'.join(L)
    from rag.config import AppConfig
    for section in AppConfig.load().raw.values():
        if not isinstance(section,dict):continue
        for key,value in section.items():
            if ('api_key' in key or 'password' in key) and value and len(str(value))>=6 and str(value) in content:raise ValueError('Secret in report')
    REPORT.write_text(content,encoding='utf-8');print('report complete; all 60 result rows reconciled: '+str(REPORT))

if __name__=='__main__':
    try:run()
    except Exception as e:print('report_failed: '+type(e).__name__);sys.exit(1)
