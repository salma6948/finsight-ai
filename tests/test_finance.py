import math
import pytest
from finsight.finance import FinancialStore, answer_financial_query, detect_metric


@pytest.fixture
def store():
    return FinancialStore()


def test_apple_2026_revenue(store):
    result = store.get("AAPL", 2026, "revenue")
    assert result.value == 478834
    assert result.year_end == "2026-09-26"
    assert "sec.gov" in result.source_url


def test_microsoft_2026_operating_margin(store):
    result = store.get("MSFT", 2026, "operating_margin")
    assert math.isclose(result.value, 155237 / 331839 * 100)


def test_nvidia_revenue_growth(store):
    result = store.get("NVDA", 2026, "revenue_growth")
    assert math.isclose(result.value, (215938 / 130497 - 1) * 100)


def test_missing_fiscal_year_is_not_hallucinated(store):
    assert store.get("AAPL", 2027, "revenue") is None
    assert store.get("AAPL", 2024, "revenue_growth") is None


def test_bilingual_metric_detection():
    assert detect_metric("Quel est le chiffre d'affaires d'Apple ?") == "revenue"
    assert detect_metric("Compare la marge nette") == "net_margin"
    assert detect_metric("Revenue growth of Nvidia") == "revenue_growth"


def test_company_comparison_cites_sources(store):
    answer = answer_financial_query("Compare Apple and Microsoft revenues for 2026", store)
    assert "478,834" in answer and "331,839" in answer
    assert "https://" in answer
    assert "2026-09-26" in answer and "2026-06-30" in answer


def test_no_company_is_not_assumed(store):
    assert answer_financial_query("What was the revenue?", store) is None


def test_invalid_metric(store):
    with pytest.raises(ValueError):
        store.compare(["AAPL"], [2026], "price_target")
