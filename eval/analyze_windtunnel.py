"""Offline analysis: frozen metrics, source trace, quota replay and lexical candidate probe.

No API calls. Lexical settings and the three-statute replay are fixed, not fitted
to dev/test scores. Probe gains at candidate recall are NOT claimed as evidence gains.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from eval_retrieval import (ROOT_DIR, STAGES, build_indexes, build_evidence, compute_metrics,
                            load_queries, validate_labels, metrics_for, summarize, grouped, unique)
from rag.schemas import RetrievedChunk

OUT=ROOT_DIR/'eval/results_windtunnel'
VERSIONS=('A','B','B_fair')


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def metric(row,scope='all_gt',grain='parent',stage='evidence',name='recall@5'):
    return row['scoped_metrics'][scope][grain][stage].get(name,0.0)


def outcome(a,b):
    return 'Win' if b>a+1e-12 else ('Loss' if b<a-1e-12 else 'Tie')


def trace_missing(row,index):
    result=[]
    for pid in row['relevant_parent_ids']:
        if pid in row['parent_ranks']['evidence']:continue
        stage=('not_recalled' if pid not in row['parent_ranks']['recall'] else
               'lost_rerank' if pid not in row['parent_ranks']['rerank'] else 'lost_evidence')
        result.append({'parent_id':pid,'doc_id':index.parent_to_doc[pid],
                       'doc_type':index.doc_meta[index.parent_to_doc[pid]]['doc_type'],'cause':stage})
    return result


def quota_replay(row,index):
    children=[RetrievedChunk(chunk_id=c['chunk_id'],parent_id=c['parent_id'],text=c['text'],
               title_path=c['title_path'],score=c['score'],rerank_score=c['rerank_score'],
               source_name=index.doc_meta[c['doc_id']]['title'],parent_text=row['parent_contexts'][c['parent_id']])
              for c in row['rerank_chunks']]
    ranked=build_evidence(children,final_k=20).items
    if row['query_type']=='案例检索':selected=ranked[:5]
    else:
        statutes=[e for e in ranked if index.doc_meta[index.parent_to_doc[e.parent_id]]['doc_type']=='statute']
        chosen={e.parent_id for e in statutes[:3]}
        for e in ranked:
            if len(chosen)>=5:break
            chosen.add(e.parent_id)
        selected=[e for e in ranked if e.parent_id in chosen][:5]
    parents=[e.parent_id for e in selected]
    docs=unique([index.parent_to_doc[p] for p in parents])
    result=dict(row)
    result['parent_ranks']={**row['parent_ranks'],'evidence':parents}
    result['doc_ranks']={**row['doc_ranks'],'evidence':docs}
    result['scoped_metrics']=metrics_for(row,result['doc_ranks'],result['parent_ranks'],index)
    result['metrics']=result['scoped_metrics']['all_gt']['doc']
    result['evidence_counts']={kind:sum(index.doc_meta[index.parent_to_doc[p]]['doc_type']==kind for p in parents)
                               for kind in ('statute','case')}
    return result


def tokens(text):
    pieces=re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+',text.lower())
    out=[]
    for part in pieces:
        if re.match(r'[a-z0-9]',part):out.append(part)
        else:
            out.extend(part)
            out.extend(part[i:i+2] for i in range(len(part)-1))
    return out


def chinese_number(s):
    if s.isdigit():return int(s)
    digits=dict(zip('零一二三四五六七八九',range(10)));value=0;current=0
    for c in s:
        if c in digits:current=digits[c]
        elif c in '十百千':value+=(current or 1)*{'十':10,'百':100,'千':1000}[c];current=0
    return value+current


class LexicalProbe:
    def __init__(self,index):
        self.index=index;self.cids=sorted(index.children);self.postings=defaultdict(list);self.lengths=[]
        for i,cid in enumerate(self.cids):
            c=index.children[cid];counts=Counter(tokens(c['title_path']+' '+c['text']))
            self.lengths.append(sum(counts.values()))
            for token,tf in counts.items():self.postings[token].append((i,tf))
        self.avg=sum(self.lengths)/len(self.lengths)
        self.aliases={did:[m['title'],m['title'].replace('中华人民共和国','')] for did,m in index.doc_meta.items()
                      if m['doc_type']=='statute'}
        self.aliases['20008']+=['劳动争议司法解释（二）','劳动争议司法解释(二)']

    def bm25(self,query,k=60):
        scores=defaultdict(float);N=len(self.cids)
        for token in set(tokens(query)):
            posting=self.postings.get(token,[])
            if not posting:continue
            idf=math.log(1+(N-len(posting)+0.5)/(len(posting)+0.5))
            for i,tf in posting:
                scores[i]+=idf*(tf*2.2)/(tf+1.2*(0.25+0.75*self.lengths[i]/self.avg))
        return [self.cids[i] for i,_ in sorted(scores.items(),key=lambda t:(-t[1],self.cids[t[0]]))[:k]]

    def exact(self,query):
        found=[]
        for match in re.finditer(r'指导案例\s*(\d+)\s*号',query):
            needle='指导案例'+match.group(1)+'号'
            for cid in self.cids:
                did=self.index.chunk_to_doc[cid]
                if needle in re.sub(r'\s+','',self.index.doc_meta[did]['title']):found.append(cid)
        mentions=[]
        for did,aliases in self.aliases.items():
            for alias in set(aliases):
                if len(alias)<4:continue
                for m in re.finditer(re.escape(alias),query):mentions.append((m.start(),m.end(),did))
        # Each article is assigned to the closest preceding explicitly named statute.
        for match in re.finditer(r'第([零一二三四五六七八九十百千0-9]+)(?:条|(?=、))',query):
            previous=[m for m in mentions if m[1]<=match.start()]
            if not previous:continue
            latest=max(previous,key=lambda m:(m[1],m[1]-m[0]));did=latest[2]
            number=chinese_number(match.group(1))
            for cid in self.cids:
                if self.index.chunk_to_doc[cid]!=did:continue
                title=self.index.children[cid]['title_path']
                if any(chinese_number(n)==number for n in re.findall(r'第([零一二三四五六七八九十百千0-9]+)条',title)):
                    found.append(cid)
        return unique(found)

    def hybrid(self,row):
        dense=[c['chunk_id'] for c in row['recall_chunks']]
        lexical=self.bm25(row['query']);exact=self.exact(row['query'])
        score=defaultdict(float)
        for ranking in (dense,lexical):
            for rank,cid in enumerate(ranking,1):score[cid]+=1/(60+rank)
        fused=sorted(score,key=lambda cid:(-score[cid],cid))
        proposed=unique(exact+fused)[:len(dense)]
        return proposed,exact,lexical


def lexical_analysis(details,index):
    lexical=LexicalProbe(index);results={}
    for version,rows in details.items():
        queries=[]
        for row in rows:
            proposed,exact,bm25=lexical.hybrid(row)
            proposed_parents={index.chunk_to_parent[c] for c in proposed}
            original=set(row['parent_ranks']['recall']);evidence=set(row['parent_ranks']['evidence'])
            gt=set(row['relevant_parent_ids']);missing=gt-evidence;recall_missing=missing-original
            recovered=recall_missing & proposed_parents
            dropped=gt & original-proposed_parents
            q={'query_id':row['query_id'],'split':row['split'],'query_type':row['query_type'],
               'evidence_missing_parents':sorted(missing),'recall_missing_parents':sorted(recall_missing),
               'hybrid_candidate_recovered':sorted(recovered),'hybrid_candidate_dropped':sorted(dropped),
               'exact_recovered':sorted(recall_missing & {index.chunk_to_parent[c] for c in exact}),
               'bm25_top30_recovered':sorted(recall_missing & {index.chunk_to_parent[c] for c in bm25[:30]}),
               'proposed_candidate_chunk_ids':proposed,
               'original_pool_recall':len(gt & original)/len(gt),
               'hybrid_pool_recall':len(gt & proposed_parents)/len(gt)}
            queries.append(q)
        groups={}
        for split in ('all','dev','test'):
            sub=[q for q in queries if split=='all' or q['split']==split]
            counts={key:sum(len(q[key]) for q in sub) for key in ('evidence_missing_parents','recall_missing_parents',
                    'hybrid_candidate_recovered','hybrid_candidate_dropped','exact_recovered','bm25_top30_recovered')}
            groups[split]={**counts,'n_queries':len(sub),'queries_with_recovery':sum(bool(q['hybrid_candidate_recovered']) for q in sub),
                           'original_pool_recall':sum(q['original_pool_recall'] for q in sub)/len(sub),
                           'hybrid_pool_recall':sum(q['hybrid_pool_recall'] for q in sub)/len(sub)}
        results[version]={'groups':groups,'details':queries}
    result={'method':{'bm25_k1':1.2,'bm25_b':0.75,'tokenization':'Chinese character unigrams+bigrams, ASCII alphanumeric terms',
                     'rrf_constant':60,'fusion':'exact chunk matches pinned, then RRF(dense concat rank, BM25), same candidate count',
                     'inputs':'query text + frozen corpus only; no GT used to form candidate ranks',
                     'no_rerank_or_final_evidence_test':True,
                     'interpretation':'candidate-stage recovery ceiling, NOT a measured final evidence improvement'},'versions':results}
    write_json(OUT/'lexical_candidate_probe.json',result)
    return result


def paired_ci(a,b,seed=20261120):
    differences=[y-x for x,y in zip(a,b)];rng=random.Random(seed);n=len(a)
    samples=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(4000))
    return {'delta':sum(differences)/n,'ci_low':samples[100],'ci_high':samples[3899]}


def source_sensitivity(details,index):
    """Do not change official labels; separately inspect rule duplicates and truncation."""
    norm=lambda text:re.sub(r'\s+','',text)
    normalized={pid:norm(p['text']) for pid,p in index.parents.items()}
    records={}
    for version,rows in details.items():
        scored=[]
        for row in rows:
            alternatives={pid:{pid} for pid in row['relevant_parent_ids']}
            if row['query_type']!='案例检索':
                for check in row['source_checks']:
                    pid=check['parent_id'];quote=norm(check['quote'])
                    if index.doc_meta[index.parent_to_doc[pid]]['doc_type']=='case' and len(quote)>=20:
                        alternatives[pid]|={candidate for candidate,text in normalized.items()
                            if index.doc_meta[index.parent_to_doc[candidate]]['doc_type']=='case' and quote in text}
            ev=set(row['parent_ranks']['evidence'])
            score=sum(bool(ev & allowed) for allowed in alternatives.values())/len(alternatives)
            visible=[]
            content=''.join(norm(e['content'])+''.join(norm(c['text']) for c in e['matched_children'])
                            for e in row['evidence_pack']['items'])
            for check in row['source_checks']:
                quote=norm(check['quote'])
                if len(quote)>=20:visible.append(quote in content)
            scored.append({'query_id':row['query_id'],'split':row['split'],'strict_parent_recall':metric(row),
                'exact_rule_alternative_recall':score,'alternative_parent_ids':{p:sorted(v) for p,v in alternatives.items() if len(v)>1},
                'long_source_quotes':len(visible),'visible_long_source_quotes':sum(visible)})
        test=[r for r in scored if r['split']=='test']
        records[version]={'test_strict_parent_recall':sum(r['strict_parent_recall'] for r in test)/len(test),
            'test_exact_rule_alternative_recall':sum(r['exact_rule_alternative_recall'] for r in test)/len(test),
            'test_long_source_quotes':sum(r['long_source_quotes'] for r in test),
            'test_visible_long_source_quotes':sum(r['visible_long_source_quotes'] for r in test),'details':scored}
    write_json(OUT/'source_sensitivity.json',{'method':'Qrels unchanged. For consultation case parents, allow other case parents containing the same >=20-character frozen quotation. Named-case questions retain exact IDs. Also inspect long quotations in actual evidence.',
        'limitations':'Literal duplicate sensitivity is not independent relevance adjudication; quotation visibility is not answer correctness.',
        'versions':records})
    return records


def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete' or manifest.get('query_count')!=60:
        raise ValueError('A full completed 60-query run is required')
    data={v:json.loads((OUT/f'version_{v}.json').read_text(encoding='utf-8')) for v in VERSIONS}
    details={v:data[v]['details'] for v in VERSIONS}
    index=build_indexes(ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    frozen=load_queries(ROOT_DIR/'eval/queries_windtunnel_60.jsonl')
    validate_labels(frozen,index)
    assert hashlib.sha256((ROOT_DIR/'eval/queries_windtunnel_60.jsonl').read_bytes()).hexdigest()==manifest['signature']['dataset_sha256']
    catalog=json.loads((ROOT_DIR/'eval/kb_catalog.json').read_text(encoding='utf-8'))
    for doc in catalog['documents']:
        for parent in doc['parents']:
            assert hashlib.sha256(index.parents[parent['parent_id']]['text'].encode()).hexdigest()==parent['text_sha256']
    assert all([r['query_id'] for r in details[v]]==[r['query_id'] for r in frozen] for v in VERSIONS)
    assert all(r['query']==q['query'] for v in VERSIONS for r,q in zip(details[v],frozen))
    assert all(not e.get('phase')=='rerank_fallback' for v in VERSIONS for e in data[v]['api_events'])
    for v in VERSIONS:
        assert len([e for e in data[v]['api_events'] if e['phase']=='rerank' and e['ok']])==60
    for path,sha in manifest['protected_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    compare=[]
    for a,b,fair in zip(details['A'],details['B'],details['B_fair']):
        gt=set(a['relevant_parent_ids']);a_ev=set(a['parent_ranks']['evidence']);b_ev=set(b['parent_ranks']['evidence'])
        item={'query_id':a['query_id'],'split':a['split'],'query_type':a['query_type'],'query':a['query'],
              'all_parent_outcome':outcome(metric(a),metric(b)),
              'statute_parent_outcome':outcome(metric(a,'statute_gt'),metric(b,'statute_gt')),
              'A_all_parent_recall5':metric(a),'B_all_parent_recall5':metric(b),
              'B_fair_all_parent_recall5':metric(fair),
              'A_statute_parent_recall5':metric(a,'statute_gt'),'B_statute_parent_recall5':metric(b,'statute_gt'),
              'gained_gt_parents':sorted(gt & b_ev-a_ev),'lost_gt_parents':sorted(gt & a_ev-b_ev),
              'B_loss_trace':trace_missing(b,index),'A_loss_trace':trace_missing(a,index),
              'both_miss':metric(a,name='hit@5')==0 and metric(b,name='hit@5')==0}
        compare.append(item)
    write_json(OUT/'paired_query_comparison.json',compare)
    with (OUT/'paired_query_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['query_id','split','query_type','all_parent_outcome','statute_parent_outcome',
                'A_all_parent_recall5','B_all_parent_recall5','B_fair_all_parent_recall5',
                'A_statute_parent_recall5','B_statute_parent_recall5','query']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(compare)
    quota={v:[quota_replay(r,index) for r in details[v]] for v in ('B','B_fair')}
    write_json(OUT/'quota_replay.json',{'policy':'non-case intent: reserve up to 3 statute parents, fill remaining by original parent rank; case intent unchanged',
        'groups':{v:grouped(rows) for v,rows in quota.items()},
        'details':{v:[{'query_id':r['query_id'],'evidence_parents':r['parent_ranks']['evidence'],
                      'scoped_metrics':r['scoped_metrics']} for r in rows] for v,rows in quota.items()}})
    lex=lexical_analysis(details,index)
    sensitivity=source_sensitivity(details,index)
    def group(v,split='test',kind='all'):
        return data[v]['groups'][split][kind]
    def g(v,scope='all_gt',grain='parent',stage='evidence',name='recall@5',split='test',kind='all'):
        return group(v,split,kind)['scoped_metrics'][scope][grain][stage].get(name,0)
    lines=['# 单法条库与法条＋案例双库检索风洞评测报告','',
        '日期：2026-11-20。报告由冻结测试集与真实接口结果生成，再核对逐题轨迹。',
        '本次为检索评测，不调用 DeepSeek、不生成法律答案、不修改线上路由或远端知识库。',
        '按用户后续要求，本报告单独保存，原《案例库构建与双库接入优化记录.md》保持不变。','',
        '## 1. 实验设置与公平性','',
        '- 知识库：128 份法条、174 份案例；511＋349＝860 个父块，1,384＋2,334＝3,718 个子块。法条实际编号包括 0/1/2/3 四类，案例以 5 开头。',
        '- 60 题：12 dev＋48 test；法律咨询 24、法条解释 18、程序咨询 12、案例检索 6。24 题标注案例，36 题只标注法条。',
        '- 先读目录和原文、撰写自然语言问题、给出 101 处原文摘录并核验 ID 和源文件哈希，再冻结测试集，之后才进行向量检索。没有用检索排名生成真值。',
        '- dev、test 都没有用于调参；正文以 48 题 test 为主要读数，同时给出 all 60 和 dev 12 的结果。指定案例文档在 dev/test 间不重叠；同一法规和相近法律主题仍可能跨 split。',
        f"- 冻结数据 SHA-256：`{manifest['signature']['dataset_sha256']}`；报告生成时再次核对全部 860 个父块正文哈希与目录快照一致。",
        '- 题目按完整事实设计；检索评测直接输入原始 query，统一跳过问题理解、追问及查询改写。没有为 60 题另行调用 LLM 证明其绝不会追问。',
        '- 未找到独立 Proposal 原件；实施依据为本次任务要求、现有测试集指南和实际代码，不能声称重新核验了 Proposal 全文。','',
        '| 组别 | 查询目标 | 每库候选 | 总候选 | 重排保留子块 | 最终父块 |',
        '|---|---|---:|---:|---:|---:|',
        '| A | 法条 | 30 | 30 | 20 | 5 |','| B | 法条＋案例 | 30 | 60 | 20 | 5 |',
        '| B-Fair | 法条＋案例 | 15 | 30 | 20 | 5 |','',
        'B-Fair 与 A 的总候选量相同，但同时把法条召回名额从 30 降到 15，因此该差异反映“固定预算下改为双库”的综合影响，不能解释成只改变案例内容的纯因果效应。B 与 B-Fair 同时扩大两个库的名额。',
        '查询目标按生产代码映射，分别搜索后按目标顺序合并，调用真实父块回补、Retriever.rerank 和 build_evidence。召回阶段保留生产代码的目标拼接顺序，没有擅自按跨库相似度重新排序。每阶段按首次出现去重文档或父块。Recall@pool 表示该阶段整个返回列表的覆盖率，用于避免 30/60 候选量及重复子块造成的误读。','',
        '### 1.1 指标口径','',
        '- `statute_gt`：只把标注法条视为相关；结果列表仍保留案例的位置，能测到案例挤占。',
        '- `all_gt`：使用标注法条＋案例。A 无法命中案例是其能力范围限制，不单独据此评价法条能力。',
        '- `case_gt`：仅在有案例标注的题目上统计案例覆盖，空真值不计作零分。',
        '- Doc-level 按 doc_id；Parent-level 按 parent_id。MRR 是阶段内第一个相关结果排名的倒数，R@k 为相关真值覆盖率，P@k 分母固定为 k，Hit@k 为是否至少命中一个。采用逐题宏平均。',
        '- 父块命中不等于所有目标条文均已展示：生产证据仍截取父块正文 900 字、每个命中子块 260 字。本次主指标评估父块身份，另在局限中说明文本可见性。','',
        '### 1.2 真实接口与运行核验','',
        f"共享 Embedding 成功 {manifest['embedding_successes']} 次，缓存 {manifest['embedding_cache_entries']} 个向量；三组复用同一题的向量，没有重复生成三份。无 Mock。前后 collection 精确数量与配置一致，脚本没有发送写入或删除请求；配置和受保护的原优化文档哈希不变。",'',
        '| 组别 | Rerank 成功/题数 | 接口成功/尝试 | 静默回退 | 平均检索秒数 |',
        '|---|---:|---:|---:|---:|']
    for v in VERSIONS:
        events=[e for e in data[v]['api_events'] if e['phase']=='rerank'];success=sum(e['ok'] for e in events)
        lines.append(f"| {v} | {success}/60 | {success}/{len(events)} | {data[v]['rerank_fallback_count']} | {group(v,'all')['mean_retrieval_seconds']:.2f} |")
    lines+=['','耗时包含实际搜索、MySQL 失败后本地回补、Rerank、证据聚合；不含共享 Query Embedding、初次 JSON 载入、组间休眠。串行运行，组别顺序按题轮换以减少固定先后顺序影响。','']
    for split,title in [('test','48 题 test 主结果'),('all','60 题全部结果')]:
        lines += [f'## 2{ ".1" if split=="test" else ".2" }. {title}','']
        for scope,scope_title in [('statute_gt','表 1：法条保底能力'),('all_gt','表 2：完整知识覆盖')]:
            lines += [f'### {scope_title}（{scope}）','',
                '| 粒度 | 阶段 | A R@5 | A MRR | B R@5 | B MRR | B-Fair R@5 | B-Fair MRR |',
                '|---|---|---:|---:|---:|---:|---:|---:|']
            for grain in ('doc','parent'):
                for stage in STAGES:
                    vals=[]
                    for v in VERSIONS:vals += [g(v,scope,grain,stage,'recall@5',split),g(v,scope,grain,stage,'mrr',split)]
                    lines.append('| '+grain+' | '+stage+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
            lines+=['','R@5 看前五个去重结果；召回完整候选池的 Recall@pool 以及所有 P@k、Hit@k、各 k 指标保存在逐组 JSON。','']
    lines+=['## 3. 案例覆盖、证据结构与分组表现','',
            '### 3.1 含案例真值子集（test）','',
            '| 粒度 | 阶段 | A R@pool | B R@pool | B-Fair R@pool |',
            '|---|---|---:|---:|---:|']
    for grain in ('doc','parent'):
        for stage in STAGES:
            lines.append(f'| {grain} | {stage} | '+' | '.join(f"{g(v,'case_gt',grain,stage,'recall@pool'):.4f}" for v in VERSIONS)+' |')
    lines+=['','### 3.2 表 3：最终证据包结构（全部 60 题）','',
            '| 组别 | 平均法条父块 | 平均案例父块 | 平均合计 | 法条/案例占比 |','|---|---:|---:|---:|---|']
    for v in VERSIONS:
        s=group(v,'all');a=s['avg_statute_evidence_count'];b=s['avg_case_evidence_count']
        lines.append(f'| {v} | {a:.3f} | {b:.3f} | {a+b:.3f} | {a/(a+b):.1%} / {b/(a+b):.1%} |')
    lines+=['','### 3.3 分意图类型（test；Evidence Parent Recall@5）','',
            '| 意图 | n | A all_gt | B all_gt | Fair all_gt | A statute_gt | B statute_gt | Fair statute_gt |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind in ('法律咨询','法条解释','程序咨询','案例检索'):
        vals=[g(v,scope='all_gt',kind=kind) for v in VERSIONS]+[g(v,scope='statute_gt',kind=kind) for v in VERSIONS]
        lines.append(f"| {kind} | {group('A',kind=kind)['n_queries']} | "+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','### 3.4 dev 与 test 分开读数','',
            '| split | n | 组别 | statute_gt Parent R@5 | all_gt Parent R@5 |','|---|---:|---|---:|---:|']
    for split in ('dev','test','all'):
        for v in VERSIONS:
            lines.append(f"| {split} | {group(v,split)['n_queries']} | {v} | {g(v,'statute_gt',split=split):.4f} | {g(v,split=split):.4f} |")
    cis={}
    lines+=['','### 3.5 test 逐题配对差值的不确定性','',
            '| 对照 | 口径 | Δ Parent R@5 | 逐题重采样 95% 区间 |','|---|---|---:|---|']
    for pair in [('A','B'),('A','B_fair'),('B_fair','B')]:
        for scope in ('statute_gt','all_gt'):
            a=[metric(r,scope) for r in details[pair[0]] if r['split']=='test']
            b=[metric(r,scope) for r in details[pair[1]] if r['split']=='test']
            ci=paired_ci(a,b);cis['/'.join(pair)+'/'+scope]=ci
            lines.append(f"| {pair[1]} − {pair[0]} | {scope} | {ci['delta']:+.4f} | [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] |")
    write_json(OUT/'paired_confidence_intervals.json',cis)
    lines+=['','固定种子，4,000 次逐题配对 bootstrap；不同题可能共享主题和法规，样本也不是随机抽取的用户流量，所以该区间仅描述本测试集的不稳定性，不能证明真实业务泛化。','',
            '### 3.6 重复案例与证据截断的敏感性核对','',
            '正式真值保持冻结。另对咨询题中的案例要求，允许同样包含冻结原文摘录（至少20字）的其他案例父块作为规则相同的备选；明确要求指导案例编号的题不放宽。这是文本重复敏感性检查，不是事后重新标注正式真值。','',
            '| 组别 | test 正式 Parent R@5 | 允许同规则案例备选 | 长原文摘录在最终文本可见数/总数 |','|---|---:|---:|---|']
    for v in VERSIONS:
        ss=sensitivity[v]
        lines.append(f"| {v} | {ss['test_strict_parent_recall']:.4f} | {ss['test_exact_rule_alternative_recall']:.4f} | {ss['test_visible_long_source_quotes']}/{ss['test_long_source_quotes']} |")
    lines+=['','摘录可见性直接检查最终截断的正文和命中子块，只统计至少20字的摘录以减少短关键词误匹配；它仍不等于完整答案正确性。','',
            '## 4. 逐题胜负与失败发生位置','',
            'Win/Tie/Loss 主判据是 A→B 的 Evidence Parent Recall@5 差值；同时单列 statute_gt，避免“补进案例但丢了法条”在综合分中被掩盖。MRR 仍在逐题结果内，未用于打破 Recall 的平局。','',
            '| split | all_gt Win/Tie/Loss | statute_gt Win/Tie/Loss | 双方均无相关父块 |','|---|---|---|---:|']
    for split in ('all','dev','test'):
        sub=[r for r in compare if split=='all' or r['split']==split]
        def wtl(key):
            counts=Counter(r[key] for r in sub);return '/'.join(str(counts[k]) for k in ('Win','Tie','Loss'))
        lines.append(f"| {split} | {wtl('all_parent_outcome')} | {wtl('statute_parent_outcome')} | {sum(r['both_miss'] for r in sub)} |")
    for category,title in [('Win','4.1 综合覆盖变好的题目'),('Loss','4.2 综合覆盖变差的题目')]:
        lines += [f'### {title}（全部 60 题）','', '| ID / split | 问题摘要 | A→B | 新增/丢失的真值父块 |','|---|---|---|---|']
        selected=[r for r in compare if r['all_parent_outcome']==category]
        for r in selected:
            ids=r['gained_gt_parents'] if category=='Win' else r['lost_gt_parents']
            names='；'.join(pid+' '+index.doc_meta[index.parent_to_doc[pid]]['title'] for pid in ids)
            lines.append(f"| {r['query_id']} / {r['split']} | {r['query'][:82]}… | {r['A_all_parent_recall5']:.2f}→{r['B_all_parent_recall5']:.2f} | {names} |")
        if not selected:lines.append('| 无 | — | — | — |')
        lines.append('')
    lines+=['### 4.3 法条退化及同题交换（包括综合分持平的题目）','',
            '| ID | statute_gt R@5 A→B | B 丢失法条的阶段 | 同时得到的案例父块 |','|---|---|---|---|']
    for r in compare:
        if r['statute_parent_outcome']!='Loss':continue
        lost=set(r['lost_gt_parents'])
        traces=[t['parent_id']+':'+t['cause'] for t in r['B_loss_trace'] if t['parent_id'] in lost and t['doc_type']=='statute']
        gained=[p for p in r['gained_gt_parents'] if index.doc_meta[index.parent_to_doc[p]]['doc_type']=='case']
        lines.append(f"| {r['query_id']} | {r['A_statute_parent_recall5']:.2f}→{r['B_statute_parent_recall5']:.2f} | {'；'.join(traces)} | {', '.join(gained) or '无'} |")
    lines+=['','`not_recalled`＝目标父块没有进入候选；`lost_rerank`＝已召回但没有进入 20 个重排子块；`lost_evidence`＝进入重排但在父块归组后没有进入最终 5 条。阶段归因可以直接由 ID 轨迹验证；它本身不能证明模型为何给某段文字更高语义分。','',
            '### 4.4 共同失败与全部失分分解','',
            '| ID | 问题摘要 | A 失分阶段 | B 失分阶段 |','|---|---|---|---|']
    both=[r for r in compare if r['both_miss']]
    for r in both:
        describe=lambda ts:'；'.join(t['parent_id']+':'+t['cause'] for t in ts)
        lines.append(f"| {r['query_id']} | {r['query'][:70]}… | {describe(r['A_loss_trace'])} | {describe(r['B_loss_trace'])} |")
    if not both:lines.append('| 无完全共同漏命中题 | 双方仍可能只覆盖部分需求，见下面的失分分解 | — | — |')
    lines+=['','| 组别 | test 未召回真值父块次数 | test 重排丢失 | test 最终证据丢失 |','|---|---:|---:|---:|']
    causes={}
    for v in VERSIONS:
        counts=Counter(t['cause'] for r in details[v] if r['split']=='test' for t in trace_missing(r,index));causes[v]=dict(counts)
        lines.append('| '+v+' | '+' | '.join(str(counts[c]) for c in ('not_recalled','lost_rerank','lost_evidence'))+' |')
    lines+=['','这里所有标注父块都已从真实原文确认在库，因此这些失分不能归因于“知识库根本没有这项真值”。本集没有用缺失地方规范或缺失司法解释作正例；无法据此评价库外问题的拒答能力。原始父块较粗、重复版本及未穷举的同类案例，也会影响严格 ID 指标。','',
            '### 4.5 代表性退化的具体证据','']
    by_q={v:{r['query_id']:r for r in rows} for v,rows in details.items()}
    selected=[]
    for cause in ('lost_evidence','lost_rerank','not_recalled'):
        candidates=[r for r in compare if r['split']=='test' and r['statute_parent_outcome']=='Loss' and
                    any(t['cause']==cause and t['parent_id'] in r['lost_gt_parents'] and t['doc_type']=='statute' for t in r['B_loss_trace'])]
        if candidates:selected.append((cause,candidates[0]))
    for cause,comparison in selected:
        qid=comparison['query_id'];a=by_q['A'][qid];b=by_q['B'][qid]
        lines += [f"**{qid}：{cause}。** {comparison['query']}",'',
            f"法条 Parent Recall@5 从 {metric(a,'statute_gt'):.3f} 变为 {metric(b,'statute_gt'):.3f}；新增真值父块为 `{comparison['gained_gt_parents']}`，丢失真值父块为 `{comparison['lost_gt_parents']}`。",'',
            '| B 最终位置 | 来源类型 | 父块 | 聚合分 | 来源标题 |','|---:|---|---|---:|---|']
        for position,e in enumerate(b['evidence_pack']['items'],1):
            kind=index.doc_meta[index.parent_to_doc[e['parent_id']]]['doc_type']
            lines.append(f"| {position} | {kind} | {e['parent_id']} | {e['score']:.5f} | {e['source_name']} |")
        if cause=='lost_evidence':
            lines+=['','该法条已经进入真实重排结果，但父块聚合后未进入五条证据；最终名额保护可以在不追加搜索的情况下检验是否能恢复。','']
        elif cause=='lost_rerank':
            lines+=['','该法条的子块进入了候选，但真实 Rerank 返回的前20条不含它；只改最后五条证据的配额无法恢复，需要把保护前移到重排候选保留阶段。不能仅凭这一轨迹断言是案例“太长”导致。','']
        else:
            lines+=['','该目标父块没有进入 B 的候选；应先查看精确/词法探针。A 与 B 对法条都请求30条，若只在一组出现，还需考虑近似检索或服务波动，而不能简单归为最终名额挤占。','']
    if not selected:lines+=['没有符合上述定义的 test 法条退化样本。','']
    lines+=['## 5. 下一步优化的证据','',
            '### 5.1 三条法条保底的离线回放','',
            '使用同一次真实 Rerank 的 20 个子块，重新做父块归组；非“案例检索”题优先保留最多 3 条法条父块，其余名额按原排序补齐。纯案例检索维持原选择。此政策直接取自任务给出的示例，没有按 test 搜索比例，也没有再次请求 Rerank。','',
            '| 组别 | test 原法条 Parent R@5 | 配额回放法条 R@5 | 原 all_gt Parent R@5 | 配额回放 all_gt R@5 |','|---|---:|---:|---:|---:|']
    for v in ('B','B_fair'):
        replay=summarize([r for r in quota[v] if r['split']=='test'])['scoped_metrics']
        lines.append(f"| {v} | {g(v,'statute_gt'):.4f} | {replay['statute_gt']['parent']['evidence']['recall@5']:.4f} | {g(v):.4f} | {replay['all_gt']['parent']['evidence']['recall@5']:.4f} |")
    lines+=['','这个回放只能恢复仍在 20 个重排候选中的法条；被召回或 Rerank 阶段淘汰的法条，不能靠最终证据配额救回。是否上线还应结合纯案例题的需求和新的保留测试集验证。','',
            '### 5.2 精确匹配＋BM25＋RRF 能解决多少','',
            '另外执行了纯本地候选探针：仅使用 query 和全库子块建立 BM25（中文单字＋双字片段，k1=1.2、b=0.75），识别 query 中明确法规条号/指导案例号，再以 RRF（常数60）融合稠密与词法排名。固定保持原组候选总数。候选生成不读取真值，生成后才对照真值计分；不调用额外模型。','',
            '| 组别 | test 最终漏失父块次数 | 其中召回阶段已漏失 | 混合候选补回其中多少 | 同时丢掉原已召回真值 | 原→混合候选 R@pool |','|---|---:|---:|---:|---:|---|']
    for v in VERSIONS:
        q=lex['versions'][v]['groups']['test']
        lines.append(f"| {v} | {q['evidence_missing_parents']} | {q['recall_missing_parents']} | {q['hybrid_candidate_recovered']} | {q['hybrid_candidate_dropped']} | {q['original_pool_recall']:.4f}→{q['hybrid_pool_recall']:.4f} |")
    b=lex['versions']['B']['groups']['test'];den=b['evidence_missing_parents'];miss=b['recall_missing_parents'];rec=b['hybrid_candidate_recovered']
    lines+=[f"",f"以 B 的 test 为例，探针补回 {rec}/{miss} 个召回阶段漏失父块，占原最终漏失 {rec}/{den}（{rec/den:.1%}）"+('。' if den else '。'),
        '这只是候选阶段的潜在可恢复比例，且需要同时看被挤掉的原真值。未让混合候选经过真实 Rerank 和最终证据选择，不能声称这些比例已被“直接救回”为 Evidence@5。RRF 的跨源顺序及中文分词方案也需要专门验证。','',
        '## 6. 复现、产物和解释边界','',
        '在项目根目录依次运行（实际实验需配置自己的只读查询权限及 Embedding/Rerank 凭据）：','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -X utf8 eval\\build_windtunnel_dataset.py',
        '.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s eval -p test_windtunnel.py -v',
        '.\\.venv\\Scripts\\python.exe -X utf8 -u eval\\run_windtunnel_ab.py --request-sleep 0.25',
        '.\\.venv\\Scripts\\python.exe -X utf8 eval\\analyze_windtunnel.py','```','',
        '已有同签名结果时，运行脚本按题按组跳过完成记录，并复用向量缓存；不要为了复现报告重复请求 API。更换数据、模型、接口或 collection 后应使用新的输出目录。第一次完整运行预计有 60 次成功 Embedding、180 次成功 Rerank、300 次搜索，失败重试另计。','',
        '标注数据是单一整理者依据原文设计的定向测试集，没有第二位独立人工标注者；正例不是对全库所有相关材料的穷举。类似案例或同一案例的另一版本可能有用却未标为正例，因此 Precision 应理解为“命中已标注正例比例”，不能等同于全部法律相关性。小样本同主题问题也会限制泛化。','',
        '父块正文较长而生成证据会截断；本报告的父块身份命中不自动证明答案所需的整条规则可见。未测回答正确性、追问率或法律意见可靠性，也没有重新审定全库法规的时效状态。','',
        '| 文件 | 用途 |','|---|---|',
        '| eval/kb_catalog.json | 302 份真实文档、父块目录、标题摘要与内容哈希。 |',
        '| eval/queries.jsonl | 三条原占位示例已修复真实 ID 与父块。 |',
        '| eval/queries_windtunnel_60.jsonl | 冻结的 60 题、split、三类真值、需求、原文摘录。 |',
        '| eval/annotation_validation.json | 数量分布、来源校验和冻结哈希。 |',
        '| eval/eval_retrieval.py | 向后兼容的单/双库评测，双粒度及分组指标。 |',
        '| eval/run_windtunnel_ab.py | 三组真实实验、共享向量、断点和读前读后核对。 |',
        '| eval/results_windtunnel/version_A.json / version_B.json / version_B_fair.json | 完整逐题候选、父块文本、重排与证据、所有指标和计时。 |',
        '| eval/results_windtunnel/paired_query_comparison.json / .csv | 每题两种口径的胜负与漏失父块。 |',
        '| eval/results_windtunnel/quota_replay.json | 三条法条配额的离线回放。 |',
        '| eval/results_windtunnel/lexical_candidate_probe.json | 词法候选探针与逐题补回/丢失。 |',
        '| eval/results_windtunnel/source_sensitivity.json | 同规则案例备选和证据文本截断敏感性。 |',
        '| eval/results_windtunnel/run_manifest.json | 接口成功记录、配置签名、前后远端数量。 |',
        '| eval/analyze_windtunnel.py | 从真实结果重新生成本报告，无模型调用。 |','',
        '### 6.1 本次代码变更及完整路径','',
        '代码放在 `eval`，实验Markdown报告统一放在 `Refinement and Optimization`。以下路径均以当前项目根目录为准。','',
        '| 状态 | 源代码完整路径 | 用途 |','|---|---|---|',
        '| 新增 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\build_windtunnel_dataset.py` | 编制真实目录和来源核验后的测试集。 |',
        '| 修改 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\eval_retrieval.py` | 双目录、多目标、分真值与粒度的检索评测。 |',
        '| 新增 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\run_windtunnel_ab.py` | 三组实验、共享向量和断点记录。 |',
        '| 新增 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\test_windtunnel.py` | 指标和链路的离线校验。 |',
        '| 新增 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\analyze_windtunnel.py` | 完整结果分析、配额回放及本地词法候选探针。 |',
        '| 新增 | `G:\\MSBD5018_agentic-rag-law-main\\eval\\summarize_windtunnel_pilot.py` | 完整实验未完成时，另行汇总共同完成的小样本。 |','']
    write_json(OUT/'analysis_summary.json',{'test_core':{v:group(v) for v in VERSIONS},'loss_causes_test':causes,
                 'outcomes':{split:{key:dict(Counter(r[key] for r in compare if split=='all' or r['split']==split))
                    for key in ('all_parent_outcome','statute_parent_outcome')} for split in ('all','dev','test')}})
    report_path=ROOT_DIR/'Refinement and Optimization/report_windtunnel_ab.md'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report and offline diagnostics generated. Protected original document unchanged.')


if __name__=='__main__':main()
