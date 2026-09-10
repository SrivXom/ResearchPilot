# Security and Data Handling

## PDF data flow

PaperLens extracts text from uploaded PDFs. Extracted text is sent to Google
Gemini for document summaries, embeddings, question answering, and follow-up
generation. Retrieved chunks are also processed by a local Hugging Face
cross-encoder for reranking.

Do not upload confidential, regulated, personal, copyrighted, or otherwise
sensitive documents unless the relevant policy and model-provider terms allow
that processing.

## Application boundary

- The public demo is a portfolio application, not a document-management
  system.
- Uploaded documents and indexes live in Streamlit session memory and are not
  intentionally persisted by the application.
- The app limits one ingestion batch to five PDFs, 200 extracted pages, one
  million extracted characters, and 20 MB per uploaded file.
- Image-only or scanned PDFs require OCR, which is not implemented.
- Inline citation markers are checked against retrieved chunks, but PaperLens
  does not perform independent factual verification or entailment checking.

## Secrets

Never commit `.env`, Streamlit secrets, API keys, or private PDFs. Rotate a
credential immediately if it appears in source control, logs, screenshots,
issues, or shared conversations.
