"""Compact report3 from locked C0/C1 records; no external API or database calls."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

from eval_retrieval import ROOT_DIR,build_indexes
from run_coverage_light import OUT,aggregate,evidence_metrics,hydrate,write

REPORT=ROOT_DIR/'Refinement and Optimization/report3.md'

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def table(headers,rows):
    esc=lambda v:str(v).replace('|','／').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(esc,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(esc,row))+' |' for row in rows])+'\n'
def f(v):return f'{v:.4f}'
def link(p):return f'[{p}](<{(ROOT_DIR/p).as_posix()}>)'

def run():
    manifest=read('run_manifest.json')
    if not manifest['complete']:raise ValueError('Experiment incomplete')
    baseline,current=read('C0.json'),read('C1.json');c0,c1=baseline['details'],current['details']
    if len(c0)!=20 or len(c1)!=20:raise ValueError('Missing queries')
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    for rows in (c0,c1):
        for r in rows:
            if evidence_metrics(r,hydrate(r['evidence_pack']),index)!=r['metrics']:raise ValueError('Metric mismatch')
    changes=[]
    for before,after in zip(c0,c1):
        assert before['query_id']==after['query_id'] and before['evidence_pack']==after['initial_evidence_pack']
        delta=after['metrics']['all_gt']['parent']['recall@5']-before['metrics']['all_gt']['parent']['recall@5']
        mrr=after['metrics']['all_gt']['parent']['mrr']-before['metrics']['all_gt']['parent']['mrr']
        changes.append({'query_id':after['query_id'],'query_type':after['query_type'],'recall_delta':delta,'mrr_delta':mrr,
            'before_parent_ids':[e['parent_id'] for e in before['evidence_pack']['items']],
            'after_parent_ids':[e['parent_id'] for e in after['evidence_pack']['items']]})
    def counts(rows):return [sum(r['recall_delta']>1e-12 for r in rows),sum(abs(r['recall_delta'])<=1e-12 for r in rows),sum(r['recall_delta']< -1e-12 for r in rows)]
    consulting0=[r for r in c0 if r['query_type']=='法律咨询'];consulting1=[r for r in c1 if r['query_type']=='法律咨询']
    assert len(consulting1)==8
    summaries=[aggregate(rows) for rows in (c0,c1,consulting0,consulting1)]
    get=lambda s,scope,key:s['metrics'][scope]['parent'][key]
    query_time=sum(r['incremental_seconds'] for r in c1);api=manifest['api_success_counts']
    model_calls=sum(api.get(k,0) for k in ('diagnosis','embedding','unified_rerank'))
    ledger={'all_recall_improved_tied_worse':counts(changes),
        'consultation_recall_improved_tied_worse':counts([r for r in changes if r['query_type']=='法律咨询']),
        'mrr_declined_ids':[r['query_id'] for r in changes if r['mrr_delta']< -1e-12],
        'searched_query_ids':manifest['searched_query_ids'],'replaced_query_ids':manifest['replaced_query_ids'],
        'all_metrics_recomputed_match':True,'initial_evidence_packs_identical':True,'per_query':changes}
    write(OUT/'analysis_summary.json',ledger)
    L=[]
    def add(text=''):L.append(text.strip()+'\n')
    add('# 劳动法 Agentic RAG：轻量证据缺口诊断与单轮补检收官报告')
    add('**记录日期**：2026-11-20  \n**项目目录**：`G:\\MSBD5018_agentic-rag-law-main`  \n**范围**：固定20题，复用Dense＋V4首轮证据，仅比较覆盖检查开关C0／C1。\n\n**当前使用方式：证据覆盖检查默认关闭，需要时在输入框上方手动开启。下表保留原C0／C1实测结果，没有为这次开关改动重新付费跑实验。**')
    add('## 1. 实验出发点与轻量化取舍')
    add('前两阶段分别完成了法条＋案例双库与来源配额（report1），以及BM25／精确匹配／RRF的混合检索诊断（report2）。在这批20题中，Dense＋V4的综合父块Recall@5为0.8083，高于混合检索的0.7667。因此本阶段把线上默认模式切回dense，保留dense_bm25和hybrid可选，检验另一个问题：**首轮证据已拿到后，能否先发现明显缺口，再用很小的成本补一次？**')
    add('本阶段实现的是“意图门禁 → 一次整体缺口诊断 → 一条受限补检查询 → 基于原问题统一重排 → 最多替换一条”的轻量闭环。**尚未实现Proposal中的完整需求—片段矩阵，也没有逐需求的充分／部分／不支持三档评估。** 不把这次实现描述为完整的证据覆盖验证系统。')
    add('诊断不会要求每题同时有法条和案例。若现有成文法依据足以回答核心问题，缺少案例本身不构成缺口；仅在用户明确要求类案或裁判尺度确需案例时才考虑案例不足。这样可以减少为了“凑齐两种来源”而补检，但也可能漏掉较隐蔽的子问题，必须以实测结果判断。')
    add('另一项关键控制是分数口径：新父块不是先用补检词打分，再拿它与原问题的旧分数比。新旧父块使用同一种去除旧排序分的代表文本，**一起针对plan.query重排一次**，保证留用比较是在同一问题、同一模型、同一次请求下进行。初筛分数只用于缩小新候选范围，不参与新旧最终大小比较。')
    add('本次没有修改历史报告、密钥配置、Qdrant数据或固定题目。报告中的补检词、条号及分数来自真实响应；没有把标注答案、真值ID或某道题的标准检索词塞进诊断提示词。证据展示中原有的引用编号与实际命中片段编号继续保留，它们是生成模块本来可见的信息，不是另行提供的正确答案标签。')
    add('## 2. 核心概念与C0／C1实验设计')
    add(table(['机制','通俗含义','本轮边界'],[
        ['意图门禁','先决定这题是否值得花一次诊断成本','仅法律咨询＋证据非空＋开关开启；其他直接gate_skipped，0模型调用'],
        ['整体缺口诊断','判断回答核心问题是否明显缺少必需依据','一次DeepSeek、temperature=0，只输出一个JSON；解析／接口失败记check_failed'],
        ['生成输入对齐','检查回答模型真正能看到的证据','复用answer.format_evidence_text，含来源、标题、父块内容和命中子块摘录，不另截前240字'],
        ['单轮受限补检','把最关键缺口改写成一条短查询','单库最多10子块；both各10，总上限20；不递归，不额外做补检词重排'],
        ['父块去重与候选收缩','避免新增证据与原包重复','MySQL回补后排除原父块，聚合出至多5个新父块，与原5条合并不超过10条'],
        ['统一原问题替换','新旧用同一把尺子比较','选最高分新父块，优先与同来源最低分旧父块比较；严格大于才换，最多1条'],
        ['异常与结果隔离','执行完成不代表缺口消失','失败、无新候选或未胜出均保留原包；分别记录triggered_search与replaced，不做二次覆盖成功声明']]))
    add('**C0**直接读取已核验的M1首轮结果，0次新API请求；**C1**逐题复用完全相同的首轮证据，只增加覆盖检查。读取前锁定M1文件SHA-256和20题哈希，并逐题核对问题、意图、真值、原文内容及原指标。本次12道非法律咨询题全部免检，占固定20题的60%；线上省下多少诊断调用，取决于实际意图分布，不能把60%推广成固定线上节省比例。')
    add('咨询门禁通过现有resolve_intent统一映射，沿用多意图优先级；实验直接使用人工中文意图标签，衡量的是意图已判断正确后的补检行为，不是线上意图识别准确率。案例真值只在非空题上平均：全部20题里有9题，8道法律咨询里有6题；计算法条指标时不删除竞争的案例结果，反之亦然。')
    add('初筛和父块聚合仍复用现有代码。补检得到的原始子块按来源再次检查数量上限，按parent_id排除旧证据，保留至多5个新父块；新旧代表文本均包含来源、结构标题、父块正文和命中摘录，去掉旧排序分，避免模型受到两种旧分数量纲的暗示。同来源替换通常保留原来源数量；若原包完全没有该来源，才退回与全局最低分旧项比较。')
    add('不替换时返回原证据包对象，内容、顺序、分数和引用ID全部不变。替换时只更新一条成员，再按这次统一分数排序、重赋ev_001…ev_005；因此即使已标注依据的集合不变，首条相关依据的名次也可能变化，Recall和MRR需要分别检查。')
    add('## 3. 实测结果、得失与代表样例')
    rows=[]
    for label,scope,key in [('法条父块Recall@5','statute_gt','recall@5'),('案例父块Recall@5（非空真值题）','case_gt','recall@5'),
        ('综合父块Recall@5','all_gt','recall@5'),('综合父块MRR','all_gt','mrr'),('综合父块Hit@5','all_gt','hit@5')]:
        rows.append([label]+[f(get(s,scope,key)) for s in summaries])
    rows.extend([['触发诊断题数',0,8,0,8],['门禁跳过题数','—',12,'—',0],
        ['诊断失败题数',0,manifest['diagnosis_failures'],0,manifest['diagnosis_failures']],
        ['触发补检题数',0,len(manifest['searched_query_ids']),0,len(manifest['searched_query_ids'])],
        ['实际替换题数',0,len(manifest['replaced_query_ids']),0,len(manifest['replaced_query_ids'])],
        ['综合Recall改善／持平／退步','—','／'.join(map(str,counts(changes))),'—','／'.join(map(str,ledger['consultation_recall_improved_tied_worse']))],
        ['平均增量耗时（秒／题）','0.00',f(query_time/20),'0.00',f(query_time/8)],
        ['新增模型调用（诊断＋Embedding＋Rerank）',0,model_calls,0,model_calls]])
    add(table(['指标／行为','C0 全20题','C1 全20题','C0 咨询8题','C1 咨询8题'],rows))
    add(f'**实际补检题号：{", ".join(manifest["searched_query_ids"]) or "无"}；实际替换题号：{", ".join(manifest["replaced_query_ids"]) or "无"}。** 8次诊断中7次判充分、1次判缺口，失败0次；只有q024执行了1次Embedding、1次法条库Top10搜索和1次新旧共10父块的统一重排。合计新增模型调用10次，另有Qdrant搜索1次，没有生成最终法律答案或调用安全审查。集合审计的GET／count请求不混入模型调用数。')
    add(f'首轮查询级增量耗时合计{query_time:.2f}秒，整个实验进程约{manifest["elapsed_this_invocation_seconds"]:.2f}秒，后者还包括输入检查、MySQL全量核验、集合快照和落盘。C0不重新执行首轮检索，因此不能把它的增量0秒当作完整问答延迟。C1的平均值分别以全部20题和实际检查8题为分母，避免门禁跳过把单次检查成本掩盖掉。')
    add('全量MySQL核验读回860个父块，正文和标题路径一致；诊断后实际补检也强制禁止JSON回退，尝试回退即使被模块捕获，评测脚本仍会中止，不把它算成数据库成功。Qdrant仅调用只读接口，前后集合数量和向量配置一致。API事件没有失败尝试，且诊断、补检、替换三个计数分别核算，未把“诊断了8题”写成“补检了8题”。')
    add('### 3.1 q024：真实补检与替换发生了，但Recall没有增长')
    sample=next(r for r in c1 if r['query_id']=='q024');trace=sample['coverage_check'];comparison=trace['comparison']
    add('这道题咨询的是孕期员工因客观情况变化拟被解除，以及合同年底到期如何处理。模型认为，现有材料缺少“孕期劳动合同期满后的顺延规则”，自主给出的补检词为：')
    add('> '+trace['diagnosis']['supplementary_query'])
    add('补检只查法条库，最多10个子块，MySQL回补后剔除原包中的00002_p005和00003_p004，形成5个新候选：20008_p009、20007_p007、20008_p011、20008_p012、20007_p006。相关条号是模型本次自己输出的，不是脚本按q024预置的搜索词。')
    add(f'统一针对原问题重排后，最高分新父块为《女职工劳动保护特别规定》的 **20007_p006，{comparison["new_score"]:.4f}**；同来源最低分旧项为《劳动法》的 **00004_p004，{comparison["old_score"]:.4f}**。前者严格更高，因此按规则替换1条，原5条中的3个法条＋2个案例数量没有变化。')
    add('这次替换既没有新增也没有移出标注的目标父块00002_p005，所以q024的综合Recall仍为1.0000。但统一重排后，新父块排在目标父块前面，使目标从第1位降至第2位，单题MRR从1.0000降到0.5000；因此全20题MRR下降0.0250，咨询子集下降0.0625。这里是**排序退步1题**，不是Recall退步1题，二者不能混为一谈。')
    add('还需注意输入层的细节：00002_p005虽然已被父块指标记作命中，但生成模型看到的是原证据构建截取的正文和前3个命中子块摘录，显示文本中没有第四十五条。诊断复用了同一展示文本，没有额外截成240字，所以它看到的缺口与“父块ID已命中”并不矛盾。按要求，补检命中这个已有父块的子块会被去重；本轮没有刷新原父块的命中摘录，不能靠同父块的新片段自动补上展示内容。')
    add('被保留的新父块主要谈孕期不得降薪、辞退等；另一新候选20008_p009的正文直接涉及合同期限依法续延，但统一原问题得分较低，未被选中。这说明“对整题更相关”未必等同于“补齐刚才识别出的那个缺口”。本次没有二次检查，故只能说发生替换，不能宣布期满顺延问题已经得到证据补全。')
    add('### 3.2 q004：判定充分，不代表标注依据已经覆盖')
    add('该题涉及奖金审批与劳动报酬。诊断返回sufficient=True，未触发搜索，C1原样保留C0。按既有真值，目标法条父块00002_p004和案例父块50071_p001均未进入原5条证据，故两组该题综合Recall@5仍为0，Hit也仍为0。')
    add('这暴露了整体诊断可能漏报缺口的风险：模型可能把一般工资保护或相关案件材料视为足够，未识别奖金问题所需的更具体依据。但真值只标注了选定的核心资料，并非穷尽所有可用证据，不能仅凭Recall为0断定原材料全部无用；是否确实漏检，需要独立人工检查。全程没有为了让该题触发补检而改提示词、指定目标条号或重跑诊断挑结果。')
    add('本批没有“补检后未替换”的真实题目，也没有Recall改善或下降的题目，因此不编造这类成功／失败样例。同口径得分相等时不换、没有新父块时不换、解析失败保留原包等边界由离线单元测试覆盖，不能替代它们在更多真实问题上的检验。')
    add('### 3.3 本阶段能说明什么，还不能说明什么')
    add('本次确认了门禁、一次诊断、一次受限真实搜索、MySQL回补和统一原问题替换可以串通；但是**没有获得标注覆盖收益，反而付出了额外调用成本，并出现MRR下降**。这不是完整法律回答质量实验，也没有验证重写后的答案、引用正确率、处理建议或法律结论。20题仍是已查看过旧结果的探索性诊断集，不是独立盲测。')
    add('下一步可单独检验同父块snippets刷新、更直接的缺口匹配与诊断漏报人工复审；本轮均未实现。结合这批题没有覆盖收益、MRR下降且增加调用成本的结果，收尾时将最初的默认开启改为**默认关闭、按需开启**，保留演示与后续研究能力。关闭覆盖检查仍正常执行原有双库检索、证据构建、回答和安全审查。')
    add('## 4. 文件清单与最小复现方式')
    paths=[('rag/coverage.py','新增：整体诊断、受限补检、统一重排及单条替换'),
        ('rag/answer.py','修改：公开共享证据格式化函数，原私有别名仍兼容'),
        ('rag/schemas.py','修改：RetrievalPlan.enable_coverage_check默认False'),
        ('rag/skill_router.py','修改：默认dense；覆盖开关默认False，仍可显式开启'),
        ('rag/workflow.py','修改：run及run_agentic_rag接收默认关闭的开关；保留实时SSE和debug'),
        ('app.py','修改：普通／流式接口传递布尔开关，省略默认False，非法值返回400'),
        ('static/app.js','修改：四阶段卡片；随请求发送勾选值，执行中锁定开关，新会话恢复关闭'),
        ('static/style.css','修改：覆盖卡片、分数对照及开关样式'),
        ('templates/index.html','修改：输入框上方增加默认关闭的可选开关、费用提示及资源版本'),
        ('eval/test_coverage_light.py','新增：16项测试，含两API默认关闭、主动开启与非法参数拒绝'),
        ('eval/run_coverage_light.py','新增：锁定M1／题目、C0零调用、真实C1、缓存与断点恢复'),
        ('eval/report_coverage_light.py','新增：按保存证据离线重算本报告与得失账本'),
        ('eval/test_hybrid_lexical.py','修改：跟随线上默认dense，继续验证hybrid可选'),
        ('eval/verify_retrieval_ui.py','修改：原混合检索UI回放显式选hybrid并关闭本轮检查'),
        ('eval/verify_coverage_ui.py','新增：缓存＋SSE＋MySQL回放，验证默认关闭／主动开启并生成4张截图'),
        ('eval/results_coverage_20','新增：C0／C1、逐题检查点、请求缓存、事件、输入哈希及分析'),
        ('Refinement and Optimization/report3.md','新增：本收官报告')]
    add(table(['完整路径','改动与用途'],[[link(p),purpose] for p,purpose in paths]))
    add('首轮父块构建公式与来源配额保持不变，未修改rag/evidence.py、rag/source_policy.py、历史M1结果、固定题集或私人YAML。接口／解析异常记check_failed；补检／统一重排失败分别单独记录，原证据原样保留。SSE显示的是处理状态，替换成功后也明确标注gap not rechecked。默认dense时，BM25／Exact在右侧显示未启用，属于模式切换的预期表现。')
    add('原有28项单元测试与本次16项合计44项回归通过；测试中的模拟接口只用于离线边界，原C1实验使用真实DeepSeek、Embedding、Qdrant与Rerank。历史报告、固定题集、原始结果与密钥配置保持不变。请求缓存以完整查询、证据／有序文档、模型、端点和参数签名，端点仅以哈希导出，不记录请求头、API Key或数据库密码。')
    add('最小复现步骤（在项目根目录）：\n\n```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s eval -p "test_*.py"\n.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s scripts -p test_mysql_setup.py\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\run_coverage_light.py\n.\\.venv\\Scripts\\python.exe -X utf8 eval\\report_coverage_light.py\n```')
    add('已有检查点时恢复会跳过完成的题，不重复诊断付费。单独运行报告脚本即可离线重算，不调用模型或数据库。--replay模式只允许已有请求缓存，不允许缺缓存时发新请求；仍会核验本机MySQL。换机器需使用自己的服务凭据，不要把本机绝对路径和配置哈希当作队友环境已经通过核验。全新复测时，先将eval/results_coverage_20另名留档，再运行脚本；空目录下会自动建立当前环境的输入锁，避免混用旧检查点。')
    add('网页按原方式运行 `.\\.venv\\Scripts\\python.exe -X utf8 app.py`，地址 `http://127.0.0.1:5000`；前台Ctrl+C停止。修改Python代码后需重启，再Ctrl+F5刷新前端。本次界面／服务检查不增加完整法律问答调用。')
    web_check=ROOT_DIR/'data/coverage_ui/verification.json'
    if web_check.exists():
        ui=json.loads(web_check.read_text(encoding='utf-8'))
        assert ui['new_model_calls']==ui['qdrant_requests']==ui['mysql_json_fallback_calls']==0 and not ui['browser_errors']
        add('### 4.1 前端展示与测试截图')
        add('右侧Agent Steps新增四阶段卡片：**意图门禁 → 整体缺口诊断 → 单轮受限补检 → 原问题统一重排**。阶段进行中会实时更新；结束后显示实际补检词、子块／新增父块数量、新旧来源标题与同次分数。免检显示0次诊断，异常明确显示失败并保留原证据；替换后标注“gap not rechecked”，不把替换等同于缺口解决。默认dense下BM25、Exact及RRF显示未启用。')
        add('以下是浏览器测试截图：**使用已保存的真实C1接口响应，重走当前coverage模块、工作流及SSE，父块由本机MySQL读取并禁止JSON回退；问题理解、回答和安全审查使用受控测试输入，没有新生成法律答案。** 本次新增模型调用及Qdrant请求均为0，浏览器无脚本错误，390像素宽度无横向溢出。历史实验结果及旧截图未改。')
        add('**图1：手动开启后，q024补检并替换一条。** 10个子块形成5个新父块，与原5条统一比较；新0.7228大于旧0.6594，提示尚未复核缺口。\n\n![q024补检、统一重排与单条替换](images/coverage_replaced.png)')
        add('**图2：q004诊断充分，保留原包。** 补检和重排没有执行；这只是模型判断，不代表真值已全部命中。\n\n![q004诊断充分并保留原证据](images/coverage_sufficient.png)')
        add('**图3：q060意图门禁跳过。** 非法律咨询，不执行诊断、补检或再次重排。\n\n![q060免检与零诊断调用](images/coverage_gate_skipped.png)')
        add('复现截图：启动网页后运行 `.\\.venv\\Scripts\\python.exe -X utf8 eval\\verify_coverage_ui.py`（需Playwright及Edge），然后Ctrl+F5刷新。截图位于同级images；验证记录在data/coverage_ui/verification.json。前端增补不重跑C0／C1，保留原实验哈希；按当前代码重新实验前，先另名留档旧结果。')
        add('### 4.2 收尾调整：常态关闭，需要时开启')
        add('在输入框上方勾选**“证据覆盖检查（可选）”**，再发送问题；取消勾选即可关闭。刷新页面或新建会话恢复关闭；同一页面会话内保留当前选择，发送期间锁定开关，只影响下一次提交，不改变已生成答案。')
        add('前后端均默认False：普通和流式API仅在JSON参数enable_coverage_check为布尔true时开启，省略为关闭，字符串“false”等非法值返回400。代码调用可用workflow.run(..., enable_coverage_check=True)或run_agentic_rag(..., enable_coverage_check=True)。开关开启后仍受法律咨询意图门禁约束。')
        add('关闭时记录gate_skipped／disabled，保留首轮证据，覆盖阶段新增诊断、Embedding、Qdrant补检和Rerank调用均为0；原有问答费用仍按使用计费。浏览器回放确认q024默认关闭时与C0证据完全一致，手动开启时重现C1替换；两API开关与非法值测试均通过。原20题C1显式开启，不受线上默认值改变影响，实验数据未重写。\n\n![默认关闭，零诊断调用并保留首轮证据](images/coverage_default_off.png)')
    content='\n'.join(L)
    from rag.config import AppConfig
    for section in AppConfig.load().raw.values():
        if not isinstance(section,dict):continue
        for key,value in section.items():
            if ('api_key' in key or 'password' in key) and value and len(str(value))>=6 and str(value) in content:raise ValueError('Sensitive value in report')
    REPORT.write_text(content,encoding='utf-8')
    ratio=len(content)/len((ROOT_DIR/'Refinement and Optimization/report1.md').read_text(encoding='utf-8'))
    print('report3 written; all metrics reconciled; character_length_ratio_to_report1='+f(ratio))

if __name__=='__main__':
    try:run()
    except Exception as exc:print('coverage_report_failed: '+type(exc).__name__);sys.exit(1)
