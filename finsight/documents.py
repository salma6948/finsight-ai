"""Page-cited PDF retrieval: paragraph chunking, BM25, optional Ollama semantic reranking."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import math
import os
import re

import fitz

from finsight.ollama import embeddings

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 250
TOKEN_RE = re.compile(r"[\w]+(?:[.,][\w]+)?", re.UNICODE)
STOPWORDS = set("the a an and or is are was were what which who where when why how for of to in on by at with from its it they their this that as has have had do does did our about main group company tell me please".split())


@dataclass(frozen=True)
class Chunk:
    document: str
    page: int
    text: str
    chunk_id: str

    def cite(self) -> str:
        return f"{self.document}, p. {self.page} [{self.chunk_id}]"


def split_text(text: str, chunk_size: int = 1100, overlap: int = 160) -> list[str]:
    """Split at paragraph/sentence boundaries when available; never cross PDF pages."""
    if chunk_size <= overlap or overlap < 0:
        raise ValueError('chunk_size must exceed overlap >= 0')
    paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', text)]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return []
    output: list[str] = []
    buffer = ''
    for p in paragraphs:
        while len(p) > chunk_size:
            if buffer:
                output.append(buffer)
                buffer = ''
            cut = p.rfind('. ', chunk_size // 2, chunk_size)
            if cut < chunk_size // 2:
                cut = p.rfind(' ', chunk_size // 2, chunk_size)
            cut = cut + 1 if cut >= chunk_size // 2 else chunk_size
            output.append(p[:cut].strip())
            p = p[max(0, cut - overlap):].strip()
        if not p:
            continue
        if not buffer:
            buffer = p
        elif len(buffer) + len(p) + 1 <= chunk_size:
            buffer += ' ' + p
        else:
            output.append(buffer)
            buffer = (buffer[-overlap:] + ' ' + p).strip() if overlap else p
    if buffer:
        output.append(buffer)
    return output


def extract_pdf(content: bytes, name: str) -> list[Chunk]:
    if not content or len(content) > MAX_PDF_BYTES:
        raise ValueError('PDF must be non-empty and at most 20 MB')
    if not content.startswith(b'%PDF-'):
        raise ValueError('Only real PDF files are accepted')
    if not re.fullmatch(r'[\w.() -]{1,120}\.pdf', name, flags=re.I) or name.startswith('.'):
        raise ValueError('Invalid PDF filename')
    try:
        with fitz.open(stream=io.BytesIO(content), filetype='pdf') as document:
            if document.needs_pass:
                raise ValueError('Password-protected PDFs are not supported')
            if document.page_count > MAX_PAGES:
                raise ValueError('PDF has too many pages (limit: 250)')
            chunks = []
            for page_num, page in enumerate(document, 1):
                for idx, text in enumerate(split_text(page.get_text('text'))):
                    digest = hashlib.sha256(f'{name}:{page_num}:{idx}:{text}'.encode()).hexdigest()[:12]
                    chunks.append(Chunk(name, page_num, text, digest))
            return chunks
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise ValueError('PDF is invalid or corrupted') from exc


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS and len(token) > 1]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class DocumentIndex:
    def __init__(self, chunks: list[Chunk] | None = None, semantic: bool | None = None):
        self.chunks: list[Chunk] = []
        self.semantic = semantic if semantic is not None else os.getenv('FINSIGHT_SEMANTIC', '1') == '1'
        self.mode = 'lexical (BM25)'
        self.embedding_error = ''
        self._vectors: list[list[float]] = []
        self._terms: list[Counter] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0
        if chunks:
            self.build(chunks)

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks[:]
        self._terms = [Counter(tokenize(c.text)) for c in chunks]
        self._df = Counter(t for counts in self._terms for t in counts)
        self._avgdl = sum(sum(c.values()) for c in self._terms) / len(chunks) if chunks else 0
        self._vectors = []
        self.mode = 'lexical (BM25)'
        self.embedding_error = ''
        if chunks and self.semantic:
            try:
                self._vectors = embeddings([c.text for c in chunks])
                self.mode = 'hybrid (BM25 + semantic embeddings)'
            except Exception as exc:
                self.embedding_error = f'{type(exc).__name__}: {str(exc)[:150]}'

    def _bm25(self, query: str) -> list[float]:
        query_terms = set(tokenize(query))
        n = len(self.chunks)
        scores = []
        for counts in self._terms:
            dl = sum(counts.values())
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / max(1, self._avgdl)))
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 4, threshold: float = 0.035) -> list[tuple[Chunk, float]]:
        if not self.chunks or not query.strip():
            return []
        lexical = self._bm25(query)
        lexical_rank = sorted(range(len(lexical)), key=lambda i: lexical[i], reverse=True)
        relevant_lexical = [i for i in lexical_rank if lexical[i] > 0]
        if self._vectors:
            try:
                qvec = embeddings([query])[0]
                similarities = [cosine(qvec, v) for v in self._vectors]
                semantic_rank = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
                # Reciprocal Rank Fusion. Require at least a small semantic signal when purely semantic.
                candidate = set(relevant_lexical) | {i for i in semantic_rank[:max(top_k * 3, 12)] if similarities[i] >= 0.25}
                lr = {idx: rank for rank, idx in enumerate(lexical_rank, 1)}
                sr = {idx: rank for rank, idx in enumerate(semantic_rank, 1)}
                fused = {i: (1 / (60 + lr[i]) if lexical[i] > 0 else 0) +
                         (1 / (60 + sr[i]) if similarities[i] >= 0.25 else 0)
                         for i in candidate}
                ranked = sorted(fused, key=fused.get, reverse=True)
                return [(self.chunks[i], round(fused[i], 5)) for i in ranked[:top_k] if fused[i] > 0]
            except Exception as exc:
                self.embedding_error = f'Query embedding unavailable: {type(exc).__name__}: {str(exc)[:120]}'
                self.mode = 'lexical (BM25 fallback)'
        # Only return lexical matches; avoid feeding unrelated text to the LLM.
        ranked = relevant_lexical[:top_k]
        return [(self.chunks[i], round(lexical[i], 5)) for i in ranked if lexical[i] >= threshold]


def evidence_text(hits: list[tuple[Chunk, float]]) -> str:
    if not hits:
        return 'No relevant supporting passages found in the uploaded documents.'
    return '\n\n'.join(
        f'**[{i}] {chunk.cite()}** (retrieval score {score:.4f})\n> {chunk.text[:1100]}'
        for i, (chunk, score) in enumerate(hits, 1)
    )
