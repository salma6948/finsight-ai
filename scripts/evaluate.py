"""Offline reproducible evaluation; does not assert unmeasured LLM faithfulness.

python -m scripts.evaluate                # BM25, finance, citations
python -m scripts.evaluate --semantic     # needs Ollama + nomic-embed-text
python -m scripts.evaluate --live-llm     # needs Ollama + qwen3:4b
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

from finsight.assistant import FinSightAssistant
from finsight.documents import Chunk, DocumentIndex
from finsight.finance import FinancialStore, answer_financial_query

CASES = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'eval_cases.json').read_text())
SAMPLE = [
    Chunk('lvmh_sample.pdf', 19, 'Revenue for 2025 was EUR 80,807 million; reported revenue fell 5%.', 'lvmh_revenue'),
    Chunk('lvmh_sample.pdf', 9, 'Currency risk exposure increased due to exchange rate fluctuations.', 'lvmh_risk'),
    Chunk('lvmh_sample.pdf', 5, 'The group operates Fashion and Leather Goods, Wines and Spirits, Selective Retailing.', 'lvmh_segments'),
]


def number_support_proxy(generated: str, evidence: str) -> dict:
    """Conservative numerical grounding check, NOT semantic faithfulness verification."""
    strip_ids = lambda s: re.sub(r'\[\d+\]', '', s)
    tokens = lambda s: set(re.findall(r'(?<![\w])\d[\d,.]*(?![\w])', strip_ids(s)))
    observed = tokens(generated)
    supported = tokens(evidence)
    return {'supported_numeric_tokens': len(observed & supported),
            'generated_numeric_tokens': len(observed),
            'numeric_support_fraction': (len(observed & supported) / len(observed) if observed else None),
            'warning': 'This lexical proxy cannot validate financial semantics, units, or attribution.'}


def evaluate(semantic=False, live_llm=False):
    store = FinancialStore()
    index = DocumentIndex(SAMPLE, semantic=semantic)
    numeric = []
    for case in CASES['numeric']:
        answer = answer_financial_query(case['question'], store) or ''
        numeric.append({'question':case['question'], 'pass': case['expected'] in answer, 'expected':case['expected']})
    retrieval = []
    for case in CASES['retrieval']:
        hits = index.search(case['question'])
        predicted = [c.chunk_id for c, _ in hits[:3]]
        relevant = case['expected_chunk']
        retrieval.append({'question':case['question'], 'pass': (relevant in predicted if relevant else not predicted),
                          'relevant_id': relevant, 'retrieved_ids':predicted})
    citations = []
    for c in SAMPLE:
        citation = c.cite()
        citations.append({'chunk':c.chunk_id, 'pass': c.document in citation and f'p. {c.page}' in citation})
    result = {
        'retrieval_mode': index.mode,
        'embedding_error': index.embedding_error,
        'numeric_exact_match': {'passed':sum(x['pass'] for x in numeric), 'total':len(numeric)},
        'retrieval_recall_at_3_or_abstention': {'passed':sum(x['pass'] for x in retrieval), 'total':len(retrieval)},
        'citation_metadata_valid': {'passed':sum(x['pass'] for x in citations), 'total':len(citations)},
        'results': {'numeric':numeric,'retrieval':retrieval,'citation':citations},
        'faithfulness': 'NOT MEASURED: source citation presence alone does not prove all claims are supported',
    }
    if live_llm:
        import os
        os.environ['FINSIGHT_LLM_PROVIDER'] = 'ollama'
        assistant = FinSightAssistant(store, index)
        out = assistant.ask('What was LVMH revenue for 2025?')
        cited = set(re.findall(r'\[(\d+)\]', out['response'].split('**Auditable underlying evidence**')[0]))
        result['llm_smoke'] = {'mode':out['mode'], 'has_valid_citation':bool(cited) and all(1 <= int(n) <= 3 for n in cited),
                               'response':out['response'][:1700],
                               'number_support_proxy':number_support_proxy(out['response'].split('---')[0], SAMPLE[0].text)}
        result['faithfulness'] = 'MANUAL REVIEW REQUIRED: cross-check every number and substantive sentence against PDF.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--live-llm', action='store_true')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.semantic, args.live_llm), indent=2, ensure_ascii=False))
