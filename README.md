# FinSight AI 2.0

**Open-source local financial research copilot.** An auditable research workflow combining financial statement analytics, PDF ingestion, hybrid BM25 + semantic retrieval, deterministic financial tools, and locally generated cited answers.

> Research prototype, **not financial advice**. The dataset is a dated snapshot, not live prices. Uploaded PDFs are not automatically added to the structured financial dashboard.

## Features

- **Financial analytics** for Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA), FY2024–FY2026, from the project's source-linked CSV.
- **Document research**: upload text-based annual report PDFs (maximum 20 MB / 250 pages each), split content by paragraph with overlap, preserve page-level citations and stable chunk IDs.
- **Hybrid retrieval**: built-in, dependency-light BM25 + optional dense vectors from **Ollama `nomic-embed-text`** and Reciprocal Rank Fusion. It automatically falls back to BM25 if Ollama or its embedding model is unavailable. Retrieval scores are **not probabilities**.
- **Local LLM**: **Qwen3 4B** on Ollama, constrained to quoted source evidence. Unsupported citation IDs cause the generated prose to be rejected; raw evidence remains available. This is a **guardrail**, not proof that the LLM's interpretation is correct.
- **Financial operations**: ratios on known company data, Decimal-based growth and margin operations, currency/unit-sensitive extraction of strictly labeled PDF revenue figures. Ambiguous tables are deliberately not auto-certified.
- **Evaluation**: unit tests; numeric exact-match smoke cases; retrieval recall@3 / abstention; citation metadata verification; optional live LLM smoke with a numeric-token support *proxy* (not full factual faithfulness).
- **Windows-friendly display**: no `st.dataframe` tables (the two original tables were replaced with native cards/links to avoid PyArrow DLL load on certain App Control policies). No `scikit-learn` dependency.

## Quick start — Windows PowerShell

**1.** Install [Python 3.11](https://www.python.org/downloads/) and [Ollama](https://ollama.com/download).

**2.** In PowerShell, start from the project root (`app.py` directory):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

**3.** Download the **open-weight local models** (once, while online). Recommended models:

```powershell
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

Make sure Ollama is running locally (the desktop application normally runs its local server). In another terminal you can test:

```powershell
ollama list
```

**4.** Start FinSight (use `python -m` on Windows to avoid a blocked `streamlit.exe` entry point):

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501. Upload a PDF in the sidebar, click **Index selected reports**, then ask e.g. **"What was LVMH revenue in 2025?"**. The first indexing pass may take longer depending on model speed and report length.

By default, FinSight tries local Ollama for LLM and embeddings. No API key or account required. To run entirely without the local models, set the following in PowerShell **before launching Streamlit**:

```powershell
$env:FINSIGHT_LLM_PROVIDER="none"
$env:FINSIGHT_SEMANTIC="0"
python -m streamlit run app.py
```

The default `OLLAMA_BASE_URL` is `http://127.0.0.1:11434` (Ollama native API, **not** `/v1`). Custom model IDs can be set using `OLLAMA_CHAT_MODEL` and `OLLAMA_EMBED_MODEL`. See `.env.example`; these variables must be set in the environment, because the example file is not auto-loaded.

### Troubleshooting: Windows code integrity policy

If a `.dll` or `.pyd` is blocked by enterprise policy, do not disable the security policy. This version removed the direct `scikit-learn` dependency and the explicit `st.dataframe` calls. Other third-party packages can still load native binaries. Consult the policy administrator if further blocks occur. Model execution may run in Ollama's own process; it is still subject to local device policy.

## Evaluation

```powershell
python -m pytest -v
python -m scripts.evaluate
python -m scripts.evaluate --semantic
python -m scripts.evaluate --semantic --live-llm
```

`--semantic` requires running Ollama and `nomic-embed-text`; `--live-llm` additionally requires `qwen3:4b` and a working local Ollama server. Offline tests use small **synthetic, labeled** document passages. They are not a representative benchmark of real regulatory filings or PDF table extraction.

**Quality interpretation:**

| Measure | What it validates | What it does not validate |
| --- | --- | --- |
| Numeric exact match | Preset calculations vs expected substrings | All calculations/cross-company comparability |
| Retrieval recall@3 | Labeled passage appears in top 3 | All relevant passages / production retrieval accuracy |
| Citation metadata | File/page reference is retained | Citation really supports every claim |
| Citation ID validation | Generated `[n]` exists in context | Correct claim-to-source attribution |
| Numeric token support proxy | Literal numbers appear in cited context | Semantic faithfulness, currency/units, derived calculations |

A rigorous faithfulness score requires a larger human-labeled set, sentence-to-evidence verification and independent checks of financial figures. Not claimed here.

## Architecture

```text
Streamlit UI
    |---- Source-linked financial CSV -> FinancialStore -> deterministic ratio tools
    |---- Text PDF -> PyMuPDF -> paragraph-aware page chunks
                                   |-- BM25 (pure Python)
                                   |-- Ollama embeddings -> cosine similarity
                                   +-- Reciprocal Rank Fusion -> top cited passages
    +---- LangGraph 3-node workflow (sequential with optional library fallback)
           financial_tools -> retrieve -> compose (local Qwen3)
                                              -> citation ID checks
                                              -> answer + auditable source passages
```

This is a **constrained research workflow**, not a self-directed trading agent. No investment recommendations, order execution, live prices, forecasting or guarantee of generated accuracy.

## Repository structure

```text
app.py                       # Streamlit UI
finsight/finance.py          # Source-linked financial dataset and ratios
finsight/financial_tools.py  # Decimal arithmetic and conservative PDF figure extraction
finsight/documents.py        # PDF ingestion and hybrid retrieval
finsight/ollama.py           # Local API client (chat and embeddings)
finsight/assistant.py        # LangGraph workflow and evidence-first synthesis
data/financials.csv          # Financial snapshot, sources and fiscal year-ends
data/eval_cases.json         # Small labeled evaluation cases
scripts/evaluate.py          # Reproducible smoke evaluation
tests/                      # Unit and integration-level mocked tests
```

## Reproducible data & scope

The financial figures and official URLs are included in `data/financials.csv`. Source links are accessible from the app. Verify them against the original filings before presenting numerical findings to stakeholders, and do not assume data is live or audited by this software. Importing new-company PDF reports adds evidence for research, **not new companies to the CSV dashboard**. PDF extraction covers selectable text; scanned documents, complex tables, footnotes and graphics may require additional parsing.

## Privacy and security

- The app communicates only with the local Ollama HTTP interface at a loopback address. No paid LLM API is used.
- Raw uploaded documents are kept in Streamlit memory for this session; clearing documents clears the indexed passages.
- Never deploy the app publicly without authentication, session isolation, request limits, document retention policy and a deployment security review.
- Document text is untrusted input. The prompt instructs the LLM to ignore embedded instructions, but this cannot fully prevent prompt injection.
- LLM-generated narratives can be wrong even with valid citations. Confirm figures, reporting periods and currency manually.

## Example research workflow

1. Download LVMH's official full-year 2025 results PDF from LVMH investor relations.
2. Upload and index it. The sidebar identifies **hybrid** vs **BM25 fallback**.
3. Ask: `What was LVMH revenue in 2025?` or `Compare its 2025 and 2024 revenue and explain the reported change.`
4. Check each `[n]` claim against the corresponding PDF page shown beneath the answer.
5. Test a question absent from the PDF; expect abstention or evidence-only output rather than fabricated exact figures.

License: MIT (application code). The model weights follow their respective upstream licenses.
