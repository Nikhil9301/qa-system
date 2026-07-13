# Retrieval-Based QA System

A retrieval-augmented QA app I built to answer natural-language questions
from PDF documents. Upload one or more PDFs, ask a question in plain
English, and get an answer grounded in the actual document content — not
a generic LLM guess.

I originally built this as a notebook experiment to understand how
retrieval-augmented generation (RAG) works end to end, then wrapped it in
a Streamlit interface so it's usable as an actual demo rather than
something that only runs cell-by-cell in Colab.

**Live demo:** _add your deployed link here once it's live_

## How it works

The core idea behind RAG is simple: instead of asking an LLM to answer
from memory (where it can hallucinate or simply not know your document),
you first retrieve the most relevant chunks of your document and hand
those to the LLM as context. The pipeline has five stages:

1. **Document loading** — uploaded PDFs are parsed page by page with
   `PyPDFLoader`, preserving source and page metadata for each chunk so
   answers can be traced back to where they came from.

2. **Chunking** — documents are split into ~1000-character chunks with a
   200-character overlap (`CharacterTextSplitter`). The overlap matters:
   without it, an answer that spans a chunk boundary can get cut in half
   and lose context.

3. **Embedding** — each chunk is converted into a vector representation
   using OpenAI's embedding model. Semantically similar text ends up
   close together in vector space, which is what makes retrieval possible
   in the first place.

4. **Indexing & retrieval** — the chunk vectors are stored in a FAISS
   index (in-memory, rebuilt per session). When a question comes in, it's
   embedded the same way, and FAISS does a similarity search to pull back
   the top-k most relevant chunks (`k=4` here).

5. **Answer generation** — the retrieved chunks are "stuffed" into the
   prompt alongside the question and passed to an OpenAI chat model,
   which generates an answer constrained to that context. The app also
   surfaces the source chunks used, so you can verify the answer isn't
   made up.

## Tech stack

- **LangChain** — orchestrates the loading → splitting → embedding →
  retrieval → generation pipeline
- **FAISS** — vector similarity search
- **OpenAI API** — embeddings (`text-embedding` model) + chat completion
  for answer generation
- **Streamlit** — the web interface
- **PyPDF** — PDF parsing

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Paste an OpenAI API key into the sidebar (never stored — it only lives in
that session), upload a PDF, and ask a question.

## Design notes

- The app takes the user's own OpenAI key at runtime instead of a
  hardcoded one, so it's safe to deploy publicly — no secrets live in the
  code or repo.
- The vector index is rebuilt per session rather than persisted, which
  keeps the app stateless and simple, at the cost of re-embedding on every
  new upload. Fine for a demo; a production version would persist the
  index (e.g. in a hosted vector DB) so repeat queries don't re-index.

## Possible next steps

- Swap `CharacterTextSplitter` for a recursive/semantic splitter for
  better chunk boundaries on messier documents
- Add re-ranking after retrieval to improve answer relevance on larger
  document sets
- Support non-PDF formats (docx, plain text, web pages)
- Persist the vector index instead of rebuilding it per session
