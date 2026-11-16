import fitz
import pytest
from finsight.documents import Chunk, DocumentIndex, extract_pdf, split_text


def pdf_bytes(text="Risk factor: currency exposure may affect our operating results."):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pdf_preserves_page_citation():
    chunks = extract_pdf(pdf_bytes(), "sample.pdf")
    assert len(chunks) == 1
    assert chunks[0].document == "sample.pdf"
    assert chunks[0].page == 1
    assert "Risk factor" in chunks[0].text


def test_retrieval_is_grounded():
    index = DocumentIndex([
        Chunk("risk.pdf", 2, "Currency risk from foreign exchange volatility.", "abc"),
        Chunk("market.pdf", 4, "Revenue grew in the cloud segment.", "def"),
    ])
    hits = index.search("foreign exchange currency risk")
    assert hits and hits[0][0].page == 2
    assert hits[0][0].document == "risk.pdf"


def test_no_match_returns_empty():
    index = DocumentIndex([Chunk("x.pdf", 1, "Operating revenue earnings.", "a")])
    assert index.search("pineapples telescope planetary") == []


def test_reject_masquerading_file():
    with pytest.raises(ValueError):
        extract_pdf(b"not a PDF", "fake.pdf")


def test_invalid_filename():
    with pytest.raises(ValueError):
        extract_pdf(pdf_bytes(), "../../escape.pdf")


def test_invalid_chunk_settings():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=100, overlap=100)
