"""
PaperLens — RAG pipeline.

Builds a chat-with-PDF system using:
- Hybrid retrieval (semantic + BM25)
- Cross-encoder reranking
- Grounded citations with page numbers
"""

from __future__ import annotations
import os

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import CrossEncoder
import re
from dataclasses import dataclass, field
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import ANSWER_PROMPT, SUMMARY_PROMPT, FOLLOWUP_PROMPT


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL = "models/gemini-embedding-001"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHAT_MODEL = "gemini-2.5-flash"
TOP_K_RERANK = 5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K_RETRIEVE = 20
MAX_FILES = 5
MAX_PAGES = 200
MAX_EXTRACTED_CHARS = 1_000_000

def load_pdf(file, source_name: str) -> list[Document]:
    """Read a PDF and return one Document per page.

    Args:
        file: A file path (string) or a file-like object (e.g. from Streamlit).
        source_name: Display name to attach as metadata (usually the filename).

    Returns:
        List of Documents — one per non-empty page, with metadata
        {"source": source_name, "page": page_number}.
    """
    reader = PdfReader(file)
    docs: list[Document] = []

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # skip blank pages (scans, dividers, etc.)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": source_name, "page": i},
            )
        )

    return docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split per-page Documents into ~900-character chunks with overlap.

    Metadata (source + page) is preserved on every chunk, so a chunk that came
    from page 4 of paper.pdf stays tagged with that page number — critical for
    citation accuracy downstream.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)

# ---------------------------------------------------------------------------
# Hybrid retrieval: semantic + BM25, fused via Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _doc_key(doc: Document) -> str:
    """Stable identity key for a chunk — used to detect duplicates across lists."""
    return f"{doc.metadata.get('source')}::{doc.metadata.get('page')}::{doc.page_content[:80]}"


def _rrf_merge(
    list_a: list[Document], list_b: list[Document], k: int, c: int = 60
) -> list[Document]:
    """Reciprocal Rank Fusion.

    Combines two ranked lists by summing 1/(c + rank). Score-free, robust,
    parameter-light. c=60 is the standard value from the original paper.
    """
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}

    for ranked_list in (list_a, list_b):
        for rank, doc in enumerate(ranked_list):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank + 1)
            by_key[key] = doc

    fused_keys = sorted(scores, key=scores.get, reverse=True)
    return [by_key[key] for key in fused_keys[:k]]


class HybridRetriever:
    """Combines dense vector search (FAISS) with BM25 keyword search.

    Why hybrid: dense retrieval is great for semantic similarity but misses
    exact technical terms (acronyms, version numbers, equation labels). BM25
    catches those. RRF merges the two ranked lists without score normalization.
    """

    def __init__(self, chunks: list[Document], api_key: str):
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=api_key,
        )
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = TOP_K_RETRIEVE
        self.k = TOP_K_RETRIEVE

    def retrieve(self, query: str) -> list[Document]:
        semantic = self.vector_store.similarity_search(query, k=self.k)
        keyword = self.bm25.invoke(query)
        return _rrf_merge(semantic, keyword, k=self.k)
    
    # ---------------------------------------------------------------------------
# Reranking with a cross-encoder
# ---------------------------------------------------------------------------


class Reranker:
    """Re-scores (query, chunk) pairs with a cross-encoder.

    Cross-encoders are slower than bi-encoders because they encode the query
    and chunk jointly. We only run them on the top-K from retrieval, never on
    the whole corpus. Net result: much higher precision at the top.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int = TOP_K_RERANK,
    ) -> list[tuple[Document, float]]:
        """Return the top_k docs with cross-encoder relevance scores."""
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs).tolist()
        ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
    
# ---------------------------------------------------------------------------
# Data classes for structured answers
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    marker: int
    source: str
    page: int
    excerpt: str
    score: float


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    followups: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------


def _truncate(text: str, n: int) -> str:
    """Tidy excerpt for citation display: single-line, max n chars."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rsplit(" ", 1)[0] + "…"


def _build_context(
    ranked: list[tuple[Document, float]],
) -> tuple[list[str], list[Citation]]:
    """Turn ranked docs into numbered context blocks + aligned citation list."""
    blocks: list[str] = []
    citations: list[Citation] = []
    for i, (doc, score) in enumerate(ranked, start=1):
        source = doc.metadata.get("source", "unknown")
        page = int(doc.metadata.get("page", 0))
        blocks.append(f"[{i}] (Source: {source}, page {page})\n{doc.page_content}")
        citations.append(
            Citation(
                marker=i,
                source=source,
                page=page,
                excerpt=_truncate(doc.page_content, 280),
                score=float(score),
            )
        )
    return blocks, citations


def _select_citations(
    answer_text: str,
    citations: list[Citation],
) -> list[Citation]:
    """Return citations referenced by valid inline markers, in answer order."""
    by_marker = {citation.marker: citation for citation in citations}
    selected: list[Citation] = []
    seen: set[int] = set()

    for raw_marker in re.findall(r"\[(\d+)\]", answer_text):
        marker = int(raw_marker)
        if marker in by_marker and marker not in seen:
            selected.append(by_marker[marker])
            seen.add(marker)

    return selected


def _parse_followups(raw: str) -> list[str]:
    """Clean up the model's followups: strip numbering/bullets, drop blanks."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if line and len(line) < 200:
            cleaned.append(line)
    return cleaned[:3]


class Generator:
    """Wraps Gemini for the three generation tasks: answer, summary, follow-ups."""

    def __init__(self, api_key: str, model: str = CHAT_MODEL):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.2,
        )

    def answer(
        self,
        question: str,
        ranked: list[tuple[Document, float]],
        history: str = "",
    ) -> Answer:
        context_blocks, citations = _build_context(ranked)
        prompt = ANSWER_PROMPT.format(
            question=question,
            context="\n\n".join(context_blocks),
            history=history or "(none)",
        )

        response = self.llm.invoke(prompt)
        usage = getattr(response, "usage_metadata", {}) or {}

        answer_text = str(response.content)
        return Answer(
            text=answer_text,
            citations=_select_citations(answer_text, citations),
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
        )

    def summarize(self, docs: list[Document], source_name: str) -> str:
        # Use up to the first 6 pages, capped at 6000 chars, to keep cost low
        sample = "\n\n".join(d.page_content for d in docs[:6])[:6000]
        prompt = SUMMARY_PROMPT.format(text=sample, source=source_name)
        return self.llm.invoke(prompt).content.strip()

    def suggest_followups(self, question: str, answer_text: str) -> list[str]:
        prompt = FOLLOWUP_PROMPT.format(question=question, answer=answer_text)
        raw = self.llm.invoke(prompt).content
        return _parse_followups(raw)
    
# ---------------------------------------------------------------------------
# Orchestrator — bundles ingest + ask into one object for the UI
# ---------------------------------------------------------------------------


class PaperLens:
    """End-to-end pipeline. Ingest PDFs once, then ask questions about them."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        self.retriever: HybridRetriever | None = None
        self.reranker = Reranker()
        self.generator = Generator(self.api_key)
        self.summaries: dict[str, str] = {}
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def ingest(self, files) -> None:
        """Accept iterable of (filename, file-like-object) pairs."""
        files = list(files)
        if len(files) > MAX_FILES:
            raise ValueError(f"Upload at most {MAX_FILES} PDFs at a time.")

        all_docs: list[Document] = []
        for name, file_obj in files:
            page_docs = load_pdf(file_obj, source_name=name)
            if not page_docs:
                continue
            all_docs.extend(page_docs)
            if len(all_docs) > MAX_PAGES:
                raise ValueError(
                    f"Uploaded PDFs exceed the {MAX_PAGES}-page limit."
                )
            extracted_chars = sum(len(doc.page_content) for doc in all_docs)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(
                    "Uploaded PDFs contain too much extracted text. "
                    "Use a smaller document set."
                )
            self.summaries[name] = self.generator.summarize(page_docs, name)

        if not all_docs:
            raise RuntimeError("No readable text found in uploaded PDFs.")

        chunks = chunk_documents(all_docs)
        self.retriever = HybridRetriever(chunks, self.api_key)

    def ask(self, question: str, history: str = "") -> Answer:
        if self.retriever is None:
            raise RuntimeError("No documents ingested yet.")
        candidates = self.retriever.retrieve(question)
        ranked = self.reranker.rerank(question, candidates)
        answer = self.generator.answer(question, ranked, history=history)
        answer.followups = self.generator.suggest_followups(question, answer.text)
        self.total_tokens_in += answer.tokens_in
        self.total_tokens_out += answer.tokens_out
        return answer
