"""Manual Gemini-backed end-to-end check for PaperLens."""

from dotenv import load_dotenv
import os

from pipeline import Generator, HybridRetriever, Reranker, chunk_documents, load_pdf


def main() -> None:
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is required.")

    docs = load_pdf("test.pdf", "test.pdf")
    chunks = chunk_documents(docs)
    retriever = HybridRetriever(chunks, key)
    reranker = Reranker()
    generator = Generator(key)
    query = "What is this document about?"
    ranked = reranker.rerank(query, retriever.retrieve(query))
    answer = generator.answer(query, ranked)
    print(answer.text)


if __name__ == "__main__":
    main()
