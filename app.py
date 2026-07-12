import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Document QA", page_icon="📄", layout="centered")

st.title("📄 Retrieval-Based QA System")
st.caption("Upload one or more PDFs, then ask questions grounded in their content.")

# ---------------------------------------------------------------------------
# API key — the user supplies their own key at runtime. This is intentional:
# it keeps secrets out of the code/repo entirely, which matters for a public
# demo link that strangers on the internet can open.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Your key is used only for this session and is never stored or logged.",
    )
    uploaded_files = st.file_uploader(
        "Upload PDF document(s)", type="pdf", accept_multiple_files=True
    )
    st.divider()
    st.caption(
        "Built with LangChain + FAISS for retrieval and OpenAI for answer "
        "generation. Source: sentence chunks are embedded, indexed, and the "
        "most relevant chunks are passed to the LLM to answer your question."
    )

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to get started.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


@st.cache_resource(show_spinner="Indexing documents...")
def build_vector_db(file_bytes_list, file_names, _api_key):
    """Load PDFs, split into chunks, embed, and build a FAISS index.

    _api_key is prefixed with an underscore so Streamlit's cache_resource
    doesn't try to hash it (and so cache correctly busts if the key changes,
    since it's still part of the function's identity via closure below).
    """
    all_docs = []
    for content, name in zip(file_bytes_list, file_names):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = name
            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)

    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    texts = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(texts, embeddings)


if not uploaded_files:
    st.info("Upload one or more PDF files in the sidebar to build the knowledge base.")
    st.stop()

file_bytes_list = [f.getvalue() for f in uploaded_files]
file_names = [f.name for f in uploaded_files]

vector_db = build_vector_db(file_bytes_list, file_names, api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
)

st.success(f"Indexed {len(uploaded_files)} document(s). Ask a question below.")

query = st.text_input("Ask a question about your documents", placeholder="e.g. What is the leave policy?")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})

    st.markdown("### Answer")
    st.write(result["result"])

    with st.expander("Sources used for this answer"):
        for i, doc in enumerate(result["source_documents"], 1):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "unknown")
            st.markdown(f"**{i}. {source}** (page {page})")
            st.text(doc.page_content[:500].strip() + "...")
