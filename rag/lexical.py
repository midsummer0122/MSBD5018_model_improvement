"""Process-cached, source-local BM25 and owned/reference-aware legal identifiers."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import jieba

from .config import ROOT_DIR
from .schemas import RetrievedChunk

CONFIG_VERSION = 'legal-lexical-v1.0'
NUM = r'[零〇一二三四五六七八九十百千万两\d]+'
ARTICLE = re.compile(r'第\s*('+NUM+r')\s*条')
CASE = re.compile(r'(指导(?:性)?案例\s*第?\s*('+NUM+r')\s*号|检例\s*第?\s*('+NUM+r')\s*号|[（(]\s*\d{4}\s*[）)]\s*[\u4e00-\u9fff\d]{1,18}\s*\d+\s*号)')
TERMS = ['劳动合同法','劳动法','劳动争议调解仲裁法','劳动合同法实施条例','劳动保障监察条例',
    '社会保险法','工伤保险条例','工资支付暂行规定','竞业限制','停工留薪期','无固定期限劳动合同',
    '非全日制用工','经济补偿','赔偿金','第二倍工资','支付令','仲裁时效','举证责任','劳动报酬',
    '未','不','没有','尚未','无故','拒不','逾期','存续','一审','二审','仲裁','受理']
STOPWORDS = {'请问','您好','你好','谢谢','感谢','麻烦','劳驾'}
ALIASES = {name:'中华人民共和国'+name for name in ['劳动合同法','劳动法','社会保险法','劳动争议调解仲裁法']}
ALIASES.update({name:name for name in ['劳动合同法实施条例','劳动保障监察条例','工伤保险条例','工资支付暂行规定']})
ALIASES['劳动合同法实施条例']='中华人民共和国劳动合同法实施条例'
INTERPRETATION = '最高人民法院关于审理劳动争议案件适用法律问题的解释'
_CACHE = {}
_LOCK = threading.RLock()


def integer(value: str) -> int:
    value=value.replace('〇','零').replace('两','二')
    if value.isdigit():return int(value)
    digits=dict(zip('零一二三四五六七八九',range(10)));units={'十':10,'百':100,'千':1000,'万':10000}
    total=section=current=0
    for ch in value:
        if ch in digits:current=digits[ch]
        elif ch=='万':total+=(section+current)*10000;section=current=0
        elif ch in units:section+=(current or 1)*units[ch];current=0
    return total+section+current


def case_ids(text):
    result=set()
    for match in CASE.finditer(re.sub(r'\s+','',unicodedata.normalize('NFKC',text))):
        if match[2]:result.add('guiding_case_'+str(integer(match[2])))
        elif match[3]:result.add('procuratorial_case_'+str(integer(match[3])))
        else:result.add(re.sub(r'\s+','',match[0]).replace('（','(').replace('）',')'))
    return result


def law_names(text,extra_names=()):
    """Longest aliases win, so 劳动合同法实施条例 never becomes 劳动合同法."""
    text=unicodedata.normalize('NFKC',text)
    found=set();occupied=[]
    patterns={**ALIASES,**{v:v for v in ALIASES.values()},**{v:v for v in extra_names}}
    for alias,canonical in sorted(patterns.items(),key=lambda pair:-len(pair[0])):
        for m in re.finditer(re.escape(alias),text):
            if not any(m.start()<b and m.end()>a for a,b in occupied):
                found.add(canonical);occupied.append(m.span())
    # The bare 司法解释二 alias is intentionally limited to this labor-specific namespace.
    for m in re.finditer(r'(?:劳动争议(?:案件适用法律问题的)?司法解释|司法解释|'+INTERPRETATION+r')\s*\(?\s*([一二三四\d])\s*\)?',text):
        if '劳动争议' not in text and re.search(r'民法典|婚姻|买卖|担保|民间借贷|刑事',text):continue
        found.add(INTERPRETATION+'('+str(integer(m[1]))+')')
    if INTERPRETATION in text and not any(n.startswith(INTERPRETATION+'(') for n in found):found.add(INTERPRETATION)
    return found


def version_tag(document):
    source=str(document.get('source_path',''))
    title=str(document.get('title',''))
    head='\n'.join(p.get('text','') for p in document.get('parents',[])[:1])[:1600]
    dates=re.findall(r'(?:19|20)\d{6}',source+' '+title)
    years=re.findall(r'(?:19|20)\d{2}',source+' '+title+' '+head)
    ordinal=re.search(r'解释\s*[（(]([一二三四\d])[）)]',source+' '+head)
    return '|'.join(dict.fromkeys(([('解释'+str(integer(ordinal[1])))] if ordinal else [])+dates+(years[:1] if not dates else []))) or 'unspecified'


def structural_metadata(document,child,kind):
    title=str(document.get('title',''));path=str(child.get('title_path',''));text=str(child.get('text',''))
    # Law identity comes from the document, never from statutes cited inside a child body.
    heading=title+' '+Path(str(document.get('source_path',''))).stem
    names=set()
    if kind=='statute':
        primary=re.sub(r'\([^)]*\)','',unicodedata.normalize('NFKC',title)).strip()
        # A reply/commentary that mentions a law in its TITLE is still not that law.
        names={ALIASES.get(primary,primary)} if primary else set()
    if INTERPRETATION in heading:
        explicit=law_names(heading+' '+str(document.get('parents',[{}])[0].get('text',''))[:200])
        names={n for n in explicit if n.startswith(INTERPRETATION)} or names
        if any(n.startswith(INTERPRETATION+'(') for n in names):names.discard(INTERPRETATION)
    owned=set(integer(m[1]) for m in ARTICLE.finditer(path)) if kind=='statute' else set()
    if kind=='statute':
        owned.update(integer(m[1]) for m in re.finditer(r'^\s*第\s*('+NUM+r')\s*条(?=\s|[：:、.。]|$)',text,re.M))
    referenced={integer(m[1]) for m in ARTICLE.finditer(text)}-owned
    own_cases=set()
    if kind=='case':
        own_cases=case_ids(title+'\n'+path)
        # Only an isolated case-number header near the DOCUMENT start owns a court number.
        first_parent=document.get('parents',[{}])[0].get('text','')
        for line in str(first_parent).splitlines()[:12]:
            if re.fullmatch(r'\s*[（(]\d{4}[）)][\u4e00-\u9fff\d\s]+号\s*',line):own_cases |= case_ids(line)
        # A guiding/procuratorial number as the opening heading may be omitted in the filename.
        for line in str(first_parent).splitlines()[:4]:
            if re.match(r'^\s*(?:指导(?:性)?案例|检例)',line):own_cases |= case_ids(line)
    return {'normalized_law_names':sorted(names),'version_tag':version_tag(document),
        'owned_articles':sorted(owned),'referenced_articles':sorted(referenced),
        'owned_case_ids':sorted(own_cases),'referenced_case_ids':sorted(case_ids(text)-own_cases)}


def corpus_fingerprint(directories=None):
    directories=directories or (ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
    files=[]
    for directory in directories:
        for path in sorted(Path(directory).glob('*.json')):
            data=path.read_bytes()
            files.append({'path':Path(directory).name+'/'+path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    if not files:raise ValueError('Empty lexical corpus')
    return hashlib.sha256(json.dumps(files,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),files


class LegalTokenizer:
    def __init__(self):
        jieba.setLogLevel(logging.ERROR)
        self.tokenizer=jieba.Tokenizer()
        for word in TERMS+list(ALIASES)+list(ALIASES.values()):self.tokenizer.add_word(word,freq=100000)

    def __call__(self,text):
        text=unicodedata.normalize('NFKC',text).lower()
        tokens=[word.strip() for word in self.tokenizer.cut(text,HMM=False)
                if word.strip() not in STOPWORDS and re.search(r'[\w\u4e00-\u9fff]',word)]
        for m in re.finditer(r'第\s*('+NUM+r')\s*([条款项号])',text):
            tokens.append({'条':'art_','款':'para_','项':'item_','号':'num_'}[m[2]]+str(integer(m[1])))
        tokens.extend(sorted(case_ids(text)))
        return tokens


class BM25:
    """Okapi length normalization with positive log(1+(N-df+.5)/(df+.5)) IDF."""
    def __init__(self,documents,k1=1.5,b=.75):
        self.k1,self.b=k1,b;self.lengths=[len(d) for d in documents]
        self.avgdl=sum(self.lengths)/max(len(documents),1);self.postings=defaultdict(list);self.idf={}
        for i,doc in enumerate(documents):
            for token,count in Counter(doc).items():self.postings[token].append((i,count))
        for token,rows in self.postings.items():self.idf[token]=math.log1p((len(documents)-len(rows)+.5)/(len(rows)+.5))

    def search(self,tokens,limit=30):
        if not tokens or not self.avgdl:return []
        scores=defaultdict(float)
        for token in set(tokens):
            for i,tf in self.postings.get(token,[]):
                denominator=tf+self.k1*(1-self.b+self.b*self.lengths[i]/self.avgdl)
                scores[i]+=self.idf[token]*tf*(self.k1+1)/denominator
        return sorted(((i,s) for i,s in scores.items() if s>0),key=lambda pair:(-pair[1],pair[0]))[:limit]


class LexicalIndex:
    def __init__(self,directories=None):
        self.tokenize=LegalTokenizer();self.chunks={};self.metadata={};self.kinds={};self.ids={'statute':[],'case':[]}
        directories=directories or (ROOT_DIR/'knowledge_data/json',ROOT_DIR/'knowledge_data/json_cases')
        for directory,kind in zip(directories,('statute','case')):
            for path in sorted(Path(directory).glob('*.json')):
                d=json.loads(path.read_text(encoding='utf-8-sig'))
                for child in d['children']:
                    cid=child['chunk_id']
                    if cid in self.chunks:raise ValueError('Duplicate lexical chunk ID')
                    self.chunks[cid]={**child,'doc_id':str(d['doc_id']),'doc_type':kind}
                    self.metadata[cid]=structural_metadata(d,child,kind);self.kinds[cid]=kind;self.ids[kind].append(cid)
        self.bm25={kind:BM25([self.tokenize(f'{self.chunks[c]["title_path"]}\n{self.chunks[c]["text"]}') for c in ids])
                   for kind,ids in self.ids.items()}
        self.law_identifiers={name for m in self.metadata.values() for name in m['normalized_law_names']}

    def bm25_search(self,query,kind,limit=30):
        return [{'chunk_id':self.ids[kind][i],'score':score,'rank':rank}
                for rank,(i,score) in enumerate(self.bm25[kind].search(self.tokenize(query),limit),1)]

    def exact_search(self,query,kind,raw_query=''):
        signal=raw_query or query
        articles={integer(m[1]) for m in ARTICLE.finditer(signal)};cases=case_ids(signal);laws=law_names(signal,self.law_identifiers)
        if raw_query and not (articles or cases or laws) and not re.search(r'换个话题|另一个问题|另外一个问题|改问|不谈',raw_query):
            signal=raw_query+'\n'+query
            articles={integer(m[1]) for m in ARTICLE.finditer(signal)};cases=case_ids(signal);laws=law_names(signal,self.law_identifiers)
        # Bare current-turn article may use an active rewritten law name, but never old article/case IDs.
        if raw_query and articles and not laws and not cases:
            rewritten_articles={integer(m[1]) for m in ARTICLE.finditer(query)}
            if rewritten_articles==articles:laws=law_names(query,self.law_identifiers)
        if not articles and not cases:return []
        query_years=set(re.findall(r'((?:19|20)\d{2})(?:年)?(?:修订|修正|版本|版)',signal))
        result=[]
        for cid in self.ids[kind]:
            m=self.metadata[cid];own=set(m['owned_articles'])&articles;refs=set(m['referenced_articles'])&articles
            own_case=set(m['owned_case_ids'])&cases;ref_case=set(m['referenced_case_ids'])&cases
            law_match=bool(set(m['normalized_law_names'])&laws)
            if query_years and m['version_tag']!='unspecified' and not any(y in m['version_tag'] for y in query_years):continue
            high=(kind=='statute' and law_match and own) or (kind=='case' and own_case)
            weak=(kind=='statute' and (own or refs) and (law_match or not laws)) or (kind=='case' and ref_case)
            if high or weak:
                result.append({'chunk_id':cid,'level':'high' if high else 'weak',
                    'tightness':len(own if high else refs or own)+len(own_case if high else ref_case),
                    'matched_owned_articles':sorted(own),'matched_referenced_articles':sorted(refs),
                    'matched_owned_case_ids':sorted(own_case),'matched_referenced_case_ids':sorted(ref_case),**m})
        return sorted(result,key=lambda r:(r['level']!='high',-r['tightness'],r['chunk_id']))


def get_lexical_index(directories=None,config_version=CONFIG_VERSION):
    start=time.perf_counter();fingerprint,files=corpus_fingerprint(directories)
    key=(fingerprint,config_version)
    with _LOCK:
        hit=key in _CACHE;build_seconds=0.
        if not hit:
            build_start=time.perf_counter();_CACHE[key]=LexicalIndex(directories);build_seconds=time.perf_counter()-build_start
        index=_CACHE[key]
    return index,{'corpus_fingerprint':fingerprint,'lexical_config_version':config_version,'cache_hit':hit,
        'build_seconds':build_seconds,'access_seconds':time.perf_counter()-start,'files':files}


def fuse_candidates(dense,index,query,kind,mode,top_k=30,raw_query='',exact_cap=5,k_rrf=60,on_progress=None):
    """Never compare cosine, BM25 and reciprocal-rank scores in one numeric ordering."""
    if mode not in ('dense','dense_bm25','hybrid'):raise ValueError('Unknown retrieval mode')
    dense=copy.deepcopy(dense)
    for rank,c in enumerate(dense,1):
        c.dense_score=c.score if c.dense_score is None else c.dense_score;c.dense_rank=rank;c.retrieval_sources=['dense']
    if mode=='dense':return dense[:top_k]
    if index is None:raise ValueError('Hybrid mode requires a lexical index')
    def progress(stage,status,**values):
        if on_progress:on_progress({'stage':stage,'status':status,**values})
    progress('bm25','started')
    bm=index.bm25_search(query,kind,top_k)
    progress('bm25','completed',count=len(bm))
    if mode=='hybrid':
        progress('exact','started')
        exact=index.exact_search(query,kind,raw_query)
        progress('exact','completed',high_count=sum(r['level']=='high' for r in exact),
                 weak_count=sum(r['level']=='weak' for r in exact))
    else:exact=[]
    progress('fusion','started')
    all_items={c.chunk_id:c for c in dense};scores=defaultdict(float)
    for c in dense:scores[c.chunk_id]+=1/(k_rrf+c.dense_rank)
    def get(cid):
        if cid not in all_items:
            p=copy.deepcopy(index.chunks[cid]);all_items[cid]=RetrievedChunk(cid,str(p['parent_id']),str(p['text']),
                title_path=str(p.get('title_path') or ''),source_name=str(p.get('source_name') or '法律法规'),payload=p)
        return all_items[cid]
    for row in bm:
        c=get(row['chunk_id']);c.bm25_rank=row['rank'];c.bm25_score=row['score'];c.retrieval_sources.append('bm25')
        scores[c.chunk_id]+=1/(k_rrf+row['rank'])
    weak_rank=0;high=[]
    for row in exact:
        c=get(row['chunk_id']);c.exact_match_level=row['level'];c.payload['_exact_metadata']=row
        c.retrieval_sources.append('exact_'+row['level'])
        if row['level']=='high':high.append(c)
        else:
            weak_rank+=1;scores[c.chunk_id]+=.5/(k_rrf+weak_rank)
    for cid,c in all_items.items():c.rrf_score=scores[cid]
    high.sort(key=lambda c:(-c.payload['_exact_metadata']['tightness'],-c.rrf_score,c.chunk_id))
    reserved=high[:min(exact_cap,top_k)];reserved_ids={c.chunk_id for c in reserved}
    rest=sorted((c for c in all_items.values() if c.chunk_id not in reserved_ids and c.rrf_score>0),key=lambda c:(-c.rrf_score,c.chunk_id))
    result=reserved+rest[:max(0,top_k-len(reserved))]
    for c in result:
        c.payload['_exact_reserved'] = c.chunk_id in reserved_ids
        c.payload['_retrieval_mode']=mode
        # Legacy score is always a cosine (0 for a non-dense hit), never BM25/RRF.
        c.score=float(c.dense_score or 0.0)
    progress('fusion','completed',count=len(result),reserved_count=len(reserved))
    return result
