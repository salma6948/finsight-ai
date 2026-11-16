from decimal import Decimal
import pytest
from finsight.documents import DocumentIndex, Chunk, split_text
from finsight.financial_tools import parse_amount, yoy_growth, profit_margin, summarize_extracted_facts
from finsight.assistant import synthesize_llm
from scripts.evaluate import evaluate


def test_amounts_and_arithmetic():
    assert parse_amount('80.8', 'bn') == Decimal('80800')
    assert parse_amount('80 807', 'millions') == Decimal('80807')
    assert round(yoy_growth(Decimal(80807), Decimal(84683)), 2) == Decimal('-4.58')
    assert profit_margin(Decimal(20), Decimal(100)) == Decimal(20)
    with pytest.raises(ValueError):
        parse_amount('42', 'USD')  # currency is not a scale
    with pytest.raises(ValueError):
        yoy_growth(Decimal(10), Decimal(0))


def test_no_wrong_slide_kpi_extraction():
    c = Chunk('LVMH.pdf', 4, '€80.8bn -1% organic REVENUE FOR 2025 €17.8bn PROFIT FROM RECURRING OPERATIONS FOR 2025', 'a')
    result = summarize_extracted_facts([c], 'What was LVMH revenue for 2025?')
    assert '80,800' in result and '17,800' not in result


def test_nonoverlap_and_page_citations():
    assert len(split_text('One paragraph.\n\nSecond paragraph.')) == 1
    index = DocumentIndex([Chunk('a.pdf', 13, 'Currency hedging and foreign exchange volatility', 'one')], semantic=False)
    hits = index.search('foreign exchange volatility')
    assert hits and hits[0][0].cite() == 'a.pdf, p. 13 [one]'
    assert index.search('pineapples telescopes') == []


def test_fallback_works_without_ollama(monkeypatch):
    def broken(_):
        raise ConnectionError('not running')
    monkeypatch.setattr('finsight.documents.embeddings', broken)
    index = DocumentIndex([Chunk('x.pdf', 2, 'Revenue growth in 2025', 'id')], semantic=True)
    assert 'lexical' in index.mode
    assert index.search('revenue growth')


def test_semantic_hybrid_mock(monkeypatch):
    def fake(texts):
        return [[1, 0] if 'currency' in t.lower() else [0, 1] for t in texts]
    monkeypatch.setattr('finsight.documents.embeddings', fake)
    index = DocumentIndex([Chunk('a.pdf', 1, 'currency risk', 'a'), Chunk('b.pdf', 2, 'sales growth', 'b')], semantic=True)
    assert index.mode.startswith('hybrid')
    assert index.search('currency')


def test_llm_rejects_fabricated_citations(monkeypatch):
    monkeypatch.setenv('FINSIGHT_LLM_PROVIDER', 'ollama')
    monkeypatch.setattr('finsight.assistant.chat', lambda *a: 'Revenue increased [9].')
    assert synthesize_llm('revenue?', '', '', '**[1] report.pdf, p. 2 [abc]**') is None
    monkeypatch.setattr('finsight.assistant.chat', lambda *a: 'Reported revenue [1].')
    assert synthesize_llm('revenue?', '', '', '**[1] report.pdf, p. 2 [abc]**') == 'Reported revenue [1].'


def test_offline_evaluation():
    result = evaluate(semantic=False)
    assert result['numeric_exact_match']['passed'] == result['numeric_exact_match']['total']
    assert result['citation_metadata_valid']['passed'] == 3
