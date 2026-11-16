"""FinSight AI — launch with `python -m streamlit run app.py`."""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from finsight.assistant import FinSightAssistant
from finsight.documents import DocumentIndex, extract_pdf
from finsight.finance import DATA_PATH, METRIC_LABELS, FinancialStore, display_value
from finsight.ollama import available

st.set_page_config(page_title="FinSight AI | Financial Research", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1250px; padding-top: 1.5rem;}
[data-testid="stMetricValue"] {font-size: 1.75rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    return FinancialStore(DATA_PATH)


store = load_data()
if "doc_index" not in st.session_state:
    st.session_state.doc_index = DocumentIndex()
if "doc_fingerprints" not in st.session_state:
    st.session_state.doc_fingerprints = set()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("◈ FinSight AI")
    st.caption("Financial research, with verifiable sources")
    st.divider()
    st.subheader("Document workspace")
    uploads = st.file_uploader(
        "Upload financial reports (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="20 MB / 250 pages per file. Text PDFs only.",
    )
    if uploads and st.button("Index selected reports", use_container_width=True):
        import hashlib

        chunks = list(st.session_state.doc_index.chunks)
        count = 0
        for f in uploads:
            blob = f.getvalue()
            fingerprint = hashlib.sha256(blob).hexdigest()
            if fingerprint in st.session_state.doc_fingerprints:
                continue
            try:
                parsed = extract_pdf(blob, f.name)
                if not parsed:
                    st.warning(f"{f.name}: no extractable text (scanned PDF/OCR not supported)")
                    continue
                chunks.extend(parsed)
                st.session_state.doc_fingerprints.add(fingerprint)
                count += 1
            except ValueError as exc:
                st.error(f"{f.name}: {exc}")
        st.session_state.doc_index.build(chunks)
        st.success(f"Indexed {count} new document(s), {len(chunks)} searchable passages")
        st.caption(f"Retrieval mode: {st.session_state.doc_index.mode}")
        if st.session_state.doc_index.embedding_error:
            st.warning(f"Semantic embedding unavailable — BM25 fallback: {st.session_state.doc_index.embedding_error}")
    st.caption(f"{len(st.session_state.doc_fingerprints)} PDFs indexed in this session")
    if st.button("Clear uploaded documents", use_container_width=True):
        st.session_state.doc_index = DocumentIndex()
        st.session_state.doc_fingerprints = set()
        st.rerun()
    st.divider()
    provider = os.getenv("FINSIGHT_LLM_PROVIDER", "ollama").lower()
    st.caption(f"Synthesis provider: **{provider}**")
    st.caption(f"Embedding model: {os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')}")
    if st.button("Check local LLM", use_container_width=True):
        if available():
            st.success("Local LLM available")
        else:
            st.warning("Ollama or chat model unavailable. Download the models from README.")
    st.caption("PDFs and prompts stay local when Ollama runs on localhost. No API key required.")

st.title("FinSight AI")
st.caption("Source-grounded equity research copilot · Public filings · USD millions")
analytics, research, sources = st.tabs(
    ["📈 Financial analytics", "💬 Research assistant", "📚 Provenance & methodology"]
)

with analytics:
    a, b, c = st.columns([2, 1, 1])
    with a:
        tickers = st.multiselect("Companies", store.tickers, default=store.tickers)
    with b:
        metric = st.selectbox(
            "Metric", list(METRIC_LABELS), format_func=lambda m: METRIC_LABELS[m]
        )
    with c:
        year = st.selectbox("Fiscal year", [2026, 2025, 2024])

    observations = store.compare(tickers, [year], metric)
    if observations:
        cols = st.columns(len(observations))
        for col, obs in zip(cols, observations):
            with col:
                st.metric(obs.company, display_value(obs))

        st.subheader("Selected financial figures")
        # Native Streamlit text/links only: avoids st.dataframe -> PyArrow DLL.
        for obs in observations:
            with st.container(border=True):
                left, middle, right = st.columns([2, 2, 1])
                with left:
                    st.markdown(f"**{obs.company} ({obs.ticker})**")
                    st.caption(f"Fiscal year: {obs.year} · FY end: {obs.year_end}")
                with middle:
                    st.markdown(f"**{METRIC_LABELS[metric]}:** {display_value(obs)}")
                    st.caption(f"Unit: {obs.unit}")
                with right:
                    st.link_button("Official source", obs.source_url, use_container_width=True)
    else:
        st.info("Select at least one company to display financial results.")

    historic = store.compare(tickers, [2024, 2025, 2026], metric)
    if historic:
        frame = pd.DataFrame(
            [{"Fiscal year": o.year, "Company": o.company, "Value": o.value} for o in historic]
        )
        fig = px.line(
            frame,
            x="Fiscal year",
            y="Value",
            color="Company",
            markers=True,
            title=METRIC_LABELS[metric],
            labels={
                "Value": "%" if metric.endswith("margin") or metric == "revenue_growth"
                else "USD million"
            },
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Fiscal years have different end dates across companies. "
        "Growth is year-over-year by each company's fiscal year, not calendar year. "
        "No prices, forecasts, or investment recommendations are included."
    )

with research:
    st.markdown(
        "Ask for **2026 revenue of Apple and Microsoft**, **NVIDIA revenue growth 2026**, "
        "**Microsoft operating margin 2025**, or search a PDF after indexing it."
    )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    user_question = st.chat_input("Ask a financial research question…")
    if user_question:
        st.session_state.messages.append({"role": "user", "text": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            try:
                response = FinSightAssistant(store, st.session_state.doc_index).ask(user_question)
                st.markdown(response["response"])
                st.caption(f"Answer mode: {response['mode']} · Retrieval: {response.get('retrieval_mode', 'n/a')}")
                st.session_state.messages.append(
                    {"role": "assistant", "text": response["response"]}
                )
            except ValueError as exc:
                st.error(str(exc))
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

with sources:
    st.markdown("### Verified source registry")
    # Avoid st.dataframe: it loads PyArrow's native DLL, blocked on some Windows PCs.
    official = store.frame[["company", "source_label", "source_url"]].drop_duplicates()
    for entry in official.itertuples(index=False):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**{entry.company}**")
                st.caption(entry.source_label)
            with right:
                st.link_button("Open filing", entry.source_url, use_container_width=True)

    st.markdown("""### Methodology & limitations
- **Dataset:** figures copied from FY2026 annual filings, referring to FY2024–FY2026. The CSV is a dated snapshot, **not live financial data**.
- **Units:** all accounting figures stored in USD millions; margin and growth ratios are calculated on demand.
- **Retrieval:** paragraph-aware per-page chunks, local BM25 + optional Ollama embeddings with reciprocal-rank fusion. Falls back to BM25 if local embeddings are unavailable.
- **Sources:** financial values link to the official filing; uploaded PDFs are cited with local document name and page.
- **LLM:** local Qwen3 via Ollama, optional; generated citations are checked against passage IDs and source evidence remains displayed. Citation presence does not prove claim accuracy.
- **Scope:** the assistant does not perform valuations, investment recommendations, forecasting or suitability reviews.
- **Privacy:** Ollama endpoint is restricted to localhost. PDFs remain in Streamlit process memory and are not saved by this app. Avoid confidential uploads to publicly hosted deployments.
""")
    st.download_button(
        "Download source financial dataset",
        store.frame.to_csv(index=False).encode(),
        file_name="finsight_financials.csv",
        mime="text/csv",
    )
