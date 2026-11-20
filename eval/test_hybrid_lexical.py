"""Offline behavioral tests for identifiers, BM25, RRF, cache and request signatures."""
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from eval_retrieval import ROOT_DIR
from rag.config import AppConfig
from rag.lexical import (BM25,LegalTokenizer,LexicalIndex,case_ids,corpus_fingerprint,
    fuse_candidates,get_lexical_index,law_names,structural_metadata,version_tag)
from rag.retrieval import Retriever
from rag.schemas import RetrievedChunk,QueryUnderstandingResult
from rag.skill_router import route_skills


def doc(did='law',title='中华人民共和国劳动合同法',text='第四十七条 经济补偿。',path='第四十七条'):
    return {'doc_id':did,'title':title,'source_path':title+'(20121228).txt',
        'parents':[{'parent_id':did+'_p1','title_path':path,'text':text,'child_ids':[did+'_c1']}],
        'children':[{'chunk_id':did+'_c1','parent_id':did+'_p1','title_path':path,'text':text}]}


class HybridChecks(unittest.TestCase):
    def fixture(self,documents):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);paths=(root/'json',root/'json_cases')
        for p in paths:p.mkdir()
        for kind,d in documents:(paths[0 if kind=='statute' else 1]/(d['doc_id']+'.json')).write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8-sig')
        return paths

    def test_owned_and_referenced_articles(self):
        d=doc(text='第八十七条 依照本法第四十七条支付赔偿。',path='劳动合同法 > 第八十七条')
        m=structural_metadata(d,d['children'][0],'statute')
        self.assertEqual(m['owned_articles'],[87]);self.assertEqual(m['referenced_articles'],[47])
        self.assertEqual(structural_metadata(doc(text='第四十七条规定了补偿。',path='导言'),doc(text='第四十七条规定了补偿。')['children'][0],'statute')['owned_articles'],[47])
        # The preceding check owns via title_path; a body citation alone does not own.
        d=doc(text='第四十七条规定了补偿。',path='导言')
        self.assertEqual(structural_metadata(d,d['children'][0],'statute')['owned_articles'],[])

    def test_exact_high_weak_and_silence(self):
        paths=self.fixture([('statute',doc()),('statute',doc('ref',text='第八十七条 依照第四十七条规定。',path='第八十七条'))])
        index=LexicalIndex(paths)
        rows={r['chunk_id']:r for r in index.exact_search('劳动合同法第47条','statute')}
        self.assertEqual(rows['law_c1']['level'],'high');self.assertEqual(rows['ref_c1']['level'],'weak')
        self.assertTrue(all(r['level']=='weak' for r in index.exact_search('第四十七条','statute')))
        self.assertEqual(index.exact_search('老板不给工资怎么办','statute'),[])

    def test_case_identity_not_citation(self):
        own=doc('case','指导案例182号：奖金纠纷','本案的裁判要点','奖金纠纷')
        ref=doc('reference','普通劳动纠纷','本案援引指导案例182号。','普通劳动纠纷')
        paths=self.fixture([('case',own),('case',ref)])
        rows={r['chunk_id']:r['level'] for r in LexicalIndex(paths).exact_search('指导性案例第一百八十二号','case')}
        self.assertEqual(rows,{'case_c1':'high','reference_c1':'weak'})
        self.assertEqual(case_ids('(2025)陕01 民终170 号'),{'(2025)陕01民终170号'})
        self.assertEqual(case_ids('检例第一百二十四号'),{'procuratorial_case_124'})

    def test_law_namespace_and_longest_alias(self):
        self.assertEqual(law_names('劳动合同法实施条例'),{'中华人民共和国劳动合同法实施条例'})
        self.assertIn('最高人民法院关于审理劳动争议案件适用法律问题的解释(2)',law_names('劳动争议司法解释（二）'))
        self.assertFalse(law_names('民法典司法解释二'))
        reply=doc('reply','最高人民法院对劳动合同法适用问题的答复',path='第四十七条')
        index=LexicalIndex(self.fixture([('statute',doc()),('statute',reply)]))
        hits={r['chunk_id'] for r in index.exact_search('劳动合同法第四十七条','statute') if r['level']=='high'}
        self.assertEqual(hits,{'law_c1'})

    def test_explicit_versions_and_fact_dates(self):
        d=doc();self.assertEqual(version_tag(d),'20121228')
        index=LexicalIndex(self.fixture([('statute',d)]))
        self.assertTrue(index.exact_search('2026年被辞退，劳动合同法第47条怎么规定','statute'))
        self.assertFalse(index.exact_search('2007年版劳动合同法第47条','statute'))

    def test_raw_scope_switch_and_active_rewrite(self):
        index=LexicalIndex(self.fixture([('statute',doc())]))
        self.assertFalse(index.exact_search('劳动合同法第四十七条','statute','换个话题，工伤待遇呢'))
        self.assertFalse(index.exact_search('劳动合同法第四十七条','statute','请解释劳动法第三条'))
        self.assertTrue(index.exact_search('劳动合同法第四十七条','statute','这个条文是什么意思'))
        u=QueryUnderstandingResult(rewritten_query='劳动法第三条')
        plan=route_skills('换个话题，解释劳动法第三条',u)
        self.assertEqual(plan.raw_query,'换个话题，解释劳动法第三条');self.assertEqual(plan.retrieval_mode,'dense')
        self.assertEqual(route_skills('q',u,retrieval_mode='hybrid').retrieval_mode,'hybrid')

    def test_negations_numbers_and_normalized_identifiers(self):
        tokenize=LegalTokenizer();tokens=tokenize('您好，尚未支付，没有合同，拒不提交，一审逾期，金额5000元，2026年。第四十七条')
        for word in ['尚未','没有','拒不','一审','逾期','5000','2026','art_47']:self.assertIn(word,tokens)
        self.assertNotIn('您好',tokens);self.assertIn('art_47',tokenize('第47条'))

    def test_bm25_manual_formula_and_zero_overlap(self):
        bm=BM25([['工资','工资'],['工伤']]);score=bm.search(['工资'])[0][1]
        expected=math.log(2)*2*2.5/(2+1.5*(.25+.75*2/1.5))
        self.assertAlmostEqual(score,expected);self.assertEqual(bm.search([]),[])
        self.assertEqual(bm.search(['完全不存在的词']),[])

    def test_high_cap_and_total_budget_and_zero_candidates(self):
        docs=[('statute',doc('d'+str(i))) for i in range(40)]
        index=LexicalIndex(self.fixture(docs));dense=[]
        for cid in index.ids['statute'][:30]:
            p=index.chunks[cid];dense.append(RetrievedChunk(cid,p['parent_id'],p['text'],title_path=p['title_path'],score=.8,payload=p))
        fused=fuse_candidates(dense,index,'劳动合同法第47条','statute','hybrid')
        self.assertEqual(len(fused),30);self.assertEqual(len({c.chunk_id for c in fused}),30)
        self.assertEqual(sum(c.payload['_exact_reserved'] for c in fused),5)
        self.assertEqual(index.bm25_search('zxqvUnseenWord','case'),[])

    def test_score_separation_and_fallback_order(self):
        provider=Mock();provider.rerank.side_effect=RuntimeError('test only')
        r=Retriever(AppConfig({}),None,None,provider)
        chunks=[RetrievedChunk('a','p','a',score=0,dense_score=None,rrf_score=.03),
                RetrievedChunk('b','q','b',score=.99,dense_score=.99,rrf_score=.01)]
        self.assertEqual([c.chunk_id for c in r.rerank('q',chunks,5,retrieval_mode='hybrid')],['a','b'])
        self.assertTrue(r.last_rerank_fallback);self.assertTrue(all(c.rerank_score is None for c in chunks))
        self.assertEqual([c.chunk_id for c in r.rerank('q',chunks,5,retrieval_mode='dense')],['b','a'])

    def test_content_fingerprint_cache_and_config_version(self):
        paths=self.fixture([('statute',doc())]);index,a=get_lexical_index(paths,'unit-v1');again,b=get_lexical_index(paths,'unit-v1')
        self.assertIs(index,again);self.assertTrue(b['cache_hit'])
        changed,c=get_lexical_index(paths,'unit-v2');self.assertIsNot(index,changed)
        p=next(paths[0].glob('*.json'));stat=p.stat();text=p.read_text(encoding='utf-8-sig')
        p.write_text(text.replace('经济补偿','工资支付'),encoding='utf-8-sig');os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        rebuilt,d=get_lexical_index(paths,'unit-v1');self.assertIsNot(index,rebuilt)
        self.assertNotEqual(a['corpus_fingerprint'],d['corpus_fingerprint'])

    def test_full_wire_signature_changes_for_order_endpoint_and_instruction(self):
        from rag.providers import RerankProvider
        from run_hybrid_m123 import rerank_payload,signature
        provider=RerankProvider(AppConfig({'rerank':{'model':'unit-model','api_key':'synthetic-fixture',
            'base_url':'https://unit.invalid/rerank'}}))
        first=rerank_payload(provider,'q',['a','b'],2)
        self.assertNotEqual(signature(first),signature(rerank_payload(provider,'q',['b','a'],2)))
        self.assertNotEqual(signature(first),signature({**first,'endpoint':'https://other.invalid'}))
        self.assertNotEqual(signature(first),signature({**first,'instruct':'different'}))
        self.assertNotEqual(signature(first),signature({**first,'top_n':1}))
        self.assertNotIn('api_key',first)

    def test_rrf_formula_separate_from_dense_scores(self):
        dense=[RetrievedChunk('a','p','a',score=.9),RetrievedChunk('b','q','b',score=.8)]
        index=Mock();index.bm25_search.return_value=[{'chunk_id':'b','score':99,'rank':1},{'chunk_id':'a','score':1,'rank':2}]
        rows=fuse_candidates(dense,index,'q','statute','dense_bm25',top_k=2)
        self.assertAlmostEqual(rows[0].rrf_score,1/61+1/62)
        self.assertAlmostEqual(rows[1].rrf_score,1/61+1/62)
        self.assertEqual(rows[0].score,.9);self.assertEqual(rows[0].bm25_score,1)


if __name__=='__main__':unittest.main()
