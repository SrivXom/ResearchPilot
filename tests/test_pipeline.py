import pytest
from langchain_core.documents import Document

from pipeline import (
    Citation,
    MAX_FILES,
    PaperLens,
    _build_context,
    _parse_followups,
    _rrf_merge,
    _select_citations,
    _truncate,
    chunk_documents,
)


def doc(text: str, page: int = 1) -> Document:
    return Document(
        page_content=text,
        metadata={"source": "paper.pdf", "page": page},
    )


def test_chunking_preserves_source_and_page():
    chunks = chunk_documents(
        [doc("Sentence one. " * 150, page=4)]
    )

    assert len(chunks) > 1
    assert all(chunk.metadata == {"source": "paper.pdf", "page": 4} for chunk in chunks)


def test_rrf_merge_deduplicates_and_combines_rankings():
    first = doc("alpha", page=1)
    second = doc("beta", page=2)
    third = doc("gamma", page=3)

    merged = _rrf_merge(
        [first, second],
        [second, third],
        k=3,
    )

    assert merged[0].page_content == "beta"
    assert {item.page_content for item in merged} == {"alpha", "beta", "gamma"}


def test_context_and_citations_keep_matching_markers():
    blocks, citations = _build_context(
        [(doc("Evidence A", 2), 0.8), (doc("Evidence B", 5), 0.6)]
    )

    assert blocks[0].startswith("[1] (Source: paper.pdf, page 2)")
    assert citations[1].marker == 2
    assert citations[1].page == 5


def test_only_cited_sources_are_returned_in_answer_order():
    citations = [
        Citation(1, "a.pdf", 1, "A", 0.9),
        Citation(2, "b.pdf", 2, "B", 0.8),
        Citation(3, "c.pdf", 3, "C", 0.7),
    ]

    selected = _select_citations(
        "The result is supported by [3] and [1]. [99] is invalid. [3]",
        citations,
    )

    assert [citation.marker for citation in selected] == [3, 1]


def test_followup_parser_removes_bullets_and_limits_output():
    parsed = _parse_followups(
        "1. First question?\n- Second question?\n* Third question?\n4. Fourth?"
    )

    assert parsed == [
        "First question?",
        "Second question?",
        "Third question?",
    ]


def test_truncate_normalizes_whitespace():
    assert _truncate(" alpha\n beta   gamma ", 16) == "alpha beta gamma"


def test_ingest_rejects_too_many_files_before_external_calls():
    lens = PaperLens.__new__(PaperLens)
    files = [(f"{index}.pdf", object()) for index in range(MAX_FILES + 1)]

    with pytest.raises(ValueError, match="Upload at most"):
        lens.ingest(files)
