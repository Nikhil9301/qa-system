# Retrieval-Based QA System — Streamlit Demo

A web app version of a document QA pipeline: upload PDFs, and ask questions
answered using retrieval-augmented generation (RAG).

**Pipeline:** PDF upload → text chunking (LangChain `CharacterTextSplitter`) →
embeddings (`OpenAIEmbeddings`) → vector index (`FAISS`) → retrieval + answer
generation (`RetrievalQA` with `ChatOpenAI`).

## Files

- `app.py` — the Streamlit app
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`), paste
your OpenAI API key into the sidebar, upload a PDF, and ask a question.

## Security note

The app asks each visitor to paste their own OpenAI API key in the sidebar
at runtime — nothing is hardcoded or stored. **Never commit an API key to
GitHub or paste one into a screenshot/notebook that gets shared.** If a key
is ever exposed, revoke it immediately in the OpenAI dashboard
(Settings → API keys) and generate a new one.

## Deploy publicly (Streamlit Community Cloud — free)

1. **Push this project to GitHub.**
   ```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "Retrieval-based QA Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   Make sure `app.py` and `requirements.txt` are the only files needed —
   don't commit PDFs with sensitive data or any `.env`/secrets file.

2. **Sign in at [share.streamlit.io](https://share.streamlit.io)** with your
   GitHub account.

3. Click **"New app"**, select your repo, branch (`main`), and set the main
   file path to `app.py`.

4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and starts
   the app automatically. You'll get a public URL like:
   `https://<your-app-name>.streamlit.app`

5. Test it: open the URL, paste an OpenAI API key, upload a sample PDF
   (e.g. a policy doc), and ask a question to confirm it returns a grounded
   answer with sources.

## Using this for your LinkedIn project

- **Project URL:** the live Streamlit link from step 4.
- **Media:** a screenshot of the deployed app answering a real question,
  plus the sources panel expanded.
- **Description:** lead with what it does and the outcome, e.g. "Built and
  deployed a retrieval-augmented QA system that answers natural-language
  questions from uploaded PDF documents in under 5 seconds, using LangChain,
  FAISS vector search, and OpenAI embeddings/LLM."
- **Skills to tag:** RAG, LangChain, Vector Databases (FAISS), OpenAI API,
  Python, Streamlit, NLP.
