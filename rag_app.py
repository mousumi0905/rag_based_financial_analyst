# rag_app.py
# All 3 PDFs pre-loaded and merged
# Users just ask — no selection, no upload needed
# Deploy to Hugging Face Spaces
# Add HF_TOKEN in Space Settings → Secrets

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import os

# ── Token ─────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    st.error(
        "HF_TOKEN not found. "
        "Add in Space Settings → Secrets."
    )
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Financial Document Analyst",
    page_icon="📄",
    layout="wide"
)

# ── Documents — all 3 pre-loaded ─────────────────────
# File names must match exactly
DOCUMENTS = [
    {
        "name": "RBI Annual Report 2024-25",
        "file": "rbi_report.pdf",
        "source": "RBI"
    },
    {
        "name": "SEBI Annual Report 2024-25",
        "file": "sebi_report.pdf",
        "source": "SEBI"
    },
    {
        "name": "HDFC Annual Report 2024-25",
        "file": "hdfc_report.pdf",   # ← update filename if different
        "source": "HDFC"
    }
]

# ── Cross-document example questions ─────────────────
EXAMPLES = [
    "What are the key economic risks mentioned across all reports?",
    "Summarise the overall state of Indian financial markets",
    "What do RBI and SEBI say about market stability?",
    "What were the major regulatory changes in 2024-25?",
    "What is the outlook for Indian economy based on all reports?",
    "How does HDFC's business performance align with RBI's monetary policy?"
]

# ── Single-document example questions ────────────────
RBI_QUESTIONS = [
    "What is RBI's stance on inflation?",
    "What was the repo rate decision?",
    "What are key risks to GDP growth?"
]
SEBI_QUESTIONS = [
    "What new regulations did SEBI introduce?",
    "What enforcement actions were taken?",
    "How did market volumes change?"
]
HDFC_QUESTIONS = [
    "What was HDFC's net interest income?",
    "How did HDFC's loan book grow in 2024-25?",
    "What are HDFC's key risk factors mentioned?"
]

# ── Header ────────────────────────────────────────────
st.title("📄 Financial Document Analyst")
st.caption(
    "Ask questions across RBI, SEBI, and HDFC "
    "reports simultaneously. No upload needed. "
    "Answers grounded in all 3 documents."
)

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.header("Documents loaded")
    for doc in DOCUMENTS:
        st.write(f"✅ {doc['name']}")
    st.divider()
    st.markdown("**How it works**")
    st.write(
        "All 3 PDFs are pre-loaded and indexed "
        "into one combined vector database. "
        "When you ask a question, the system "
        "searches across all documents "
        "simultaneously and returns the most "
        "relevant sections — from any report."
    )
    st.divider()
    st.markdown("**Try cross-document questions**")
    st.write(
        "e.g. How does RBI's inflation view "
        "align with HDFC's lending growth outlook?"
    )
    st.divider()
    st.markdown("**Built by Mousumi Kundu**")
    st.markdown(
        "DATA SCIENCE | Gen AI | LangChain | RAG"
    )
    st.markdown(
        "[GitHub](https://github.com/mousumi0905) | "
        "[LinkedIn](https://www.linkedin.com/in/mousumi-kundu-00109819b/)"
    )
    st.divider()
    st.markdown("**Stack**")
    st.write(
        "LangChain · FAISS · Mistral 7B · "
        "Sentence Transformers · "
        "Streamlit · HF Spaces"
    )

# ── Load embedding model once ─────────────────────────
@st.cache_resource(
    show_spinner="Loading embedding model..."
)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# ── Load LLM once ─────────────────────────────────────
@st.cache_resource(
    show_spinner="Loading language model..."
)
def load_llm():
    return HuggingFaceHub(
        repo_id=(
            "mistralai/Mistral-7B-Instruct-v0.2"
        ),
        model_kwargs={
            "temperature": 0.1,
            "max_new_tokens": 512,
            "repetition_penalty": 1.1
        }
    )

# ── Load ALL documents and merge into one FAISS ───────
@st.cache_resource(
    show_spinner="Loading and indexing all documents..."
)
def load_all_documents():
    embeddings = load_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    loaded = []
    total_pages = 0

    for doc in DOCUMENTS:
        file_path = doc["file"]

        # Check file exists
        if not os.path.exists(file_path):
            st.warning(
                f"PDF not found: {file_path} — "
                f"skipping"
            )
            continue

        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Add source name to each chunk's metadata
        # so we know which document it came from
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source_doc"] = (
                doc["name"]
            )
            chunk.metadata["source"] = doc["source"]

        all_chunks.extend(chunks)
        total_pages += len(pages)
        loaded.append(doc["name"])

    if not all_chunks:
        return None

    # Build ONE combined FAISS index
    # from ALL document chunks
    vectorstore = FAISS.from_documents(
        all_chunks, embeddings
    )

    return {
        "vectorstore": vectorstore,
        "total_chunks": len(all_chunks),
        "total_pages": total_pages,
        "loaded_docs": loaded
    }

# ── Load everything at startup ────────────────────────
data = load_all_documents()

if data is None:
    st.error(
        "No PDFs found. Make sure these files "
        "are in the same folder as rag_app.py:\n"
        "- rbi_report.pdf\n"
        "- sebi_report.pdf\n"
        "- hdfc_report.pdf"
    )
    st.stop()

# ── Show loaded documents and metrics ────────────────
st.subheader("Documents loaded and ready")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Documents", len(data["loaded_docs"]))
d2.metric("Total pages", data["total_pages"])
d3.metric("Total chunks", data["total_chunks"])
d4.metric("Retrieved per query", 6)

st.divider()

# ── Build RAG chain ───────────────────────────────────
retriever = (
    data["vectorstore"]
    .as_retriever(search_kwargs={"k": 6})
)
llm = load_llm()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# ── Question section ──────────────────────────────────
st.subheader("Ask your question")

# Tabs for question types
tab1, tab2 = st.tabs([
    "Cross-document questions",
    "Single document questions"
])

selected_q = None

with tab1:
    st.write(
        "These questions search across "
        "all 3 documents simultaneously:"
    )
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"cross_{i}"):
            selected_q = q

with tab2:
    st.write("**RBI specific:**")
    rbi_cols = st.columns(3)
    for i, q in enumerate(RBI_QUESTIONS):
        if rbi_cols[i].button(q, key=f"rbi_{i}"):
            selected_q = q

    st.write("**SEBI specific:**")
    sebi_cols = st.columns(3)
    for i, q in enumerate(SEBI_QUESTIONS):
        if sebi_cols[i].button(q, key=f"sebi_{i}"):
            selected_q = q

    st.write("**HDFC specific:**")
    hdfc_cols = st.columns(3)
    for i, q in enumerate(HDFC_QUESTIONS):
        if hdfc_cols[i].button(q, key=f"hdfc_{i}"):
            selected_q = q

# Free text input
question = st.text_input(
    "Or type your own question",
    value=selected_q if selected_q else "",
    placeholder=(
        "e.g. How does RBI inflation outlook "
        "compare with HDFC's lending growth strategy?"
    )
)

# ── Generate answer ───────────────────────────────────
if question:
    with st.spinner(
        "Searching across all 3 documents..."
    ):
        try:
            result = qa_chain({"query": question})
            answer = result['result']
            sources = result['source_documents']

            # Clean answer
            for marker in [
                "Helpful Answer:",
                "Answer:",
                "Response:"
            ]:
                if marker in answer:
                    answer = answer.split(
                        marker)[-1].strip()
            answer = answer.strip()

            # Display answer
            st.subheader("Answer")
            st.markdown(answer)

            # Show which documents were used
            source_docs_used = list(set([
                doc.metadata.get(
                    "source_doc", "Unknown"
                )
                for doc in sources
            ]))

            st.info(
                f"Sources used: "
                f"{', '.join(source_docs_used)}"
            )

            # Show source chunks
            with st.expander(
                f"Source sections — "
                f"{len(sources)} chunks from "
                f"{len(source_docs_used)} documents"
            ):
                for i, doc in enumerate(sources):
                    page_num = (
                        doc.metadata.get('page', 0)
                        + 1
                    )
                    source_name = doc.metadata.get(
                        "source_doc", "Unknown"
                    )
                    st.markdown(
                        f"**Section {i+1} — "
                        f"{source_name} — "
                        f"Page {page_num}**"
                    )
                    st.write(
                        doc.page_content[:400]
                    )
                    if i < len(sources) - 1:
                        st.divider()

            st.divider()
            st.caption(
                "Answer grounded in documents. "
                "Source sections shown above "
                "with document name and page number."
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info(
                "Try rephrasing your question."
            )
