from finsight.assistant import FinSightAssistant
from finsight.documents import DocumentIndex, Chunk
from finsight.finance import FinancialStore


def test_financial_assistant_without_llm(monkeypatch):
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "none")
    assistant = FinSightAssistant(FinancialStore())
    result = assistant.ask("What is Microsoft revenue in 2026?")
    assert result["mode"] == "grounded_no_llm"
    assert "331,839" in result["response"]
    assert "microsoft.com" in result["response"]


def test_document_assistant_without_llm(monkeypatch):
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "none")
    index = DocumentIndex([Chunk("report.pdf", 7, "Revenue exposure to currency risk in Europe.", "c1")])
    result = FinSightAssistant(FinancialStore(), index).ask("currency risk in Europe")
    assert "report.pdf, p. 7" in result["response"]


def test_unanswerable_question(monkeypatch):
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "none")
    result = FinSightAssistant(FinancialStore()).ask("What is the weather tomorrow?")
    assert result["mode"] == "no_evidence"


def test_overlong_question_rejected():
    try:
        FinSightAssistant(FinancialStore()).ask("x" * 2001)
    except ValueError:
        pass
    else:
        raise AssertionError("Overlong question was accepted")
