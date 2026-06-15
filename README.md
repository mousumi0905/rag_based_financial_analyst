# 📄 Financial Document Analyst — RAG App

A Retrieval-Augmented Generation (RAG) application built with **Streamlit** and **LangChain** that lets users ask natural-language questions across three pre-loaded financial annual reports — **RBI**, **SEBI**, and **HDFC** (2024–25) — without needing to upload or select any documents.

All three PDFs are chunked, embedded, and merged into a **single FAISS vector index** at startup. When a user asks a question, the app retrieves the most relevant chunks from across all three reports and passes them to an LLM to generate a grounded, source-cited answer.

---

## ✨ Key Features

- **Zero-upload experience** — all 3 reports are pre-loaded and indexed automatically when the app starts.
- **Cross-document retrieval** — a single combined FAISS index means questions can pull context from RBI, SEBI, and HDFC simultaneously (e.g. *"How does RBI's inflation view align with HDFC's lending growth outlook?"*).
- **Example question library** — built-in buttons for both cross-document and single-document (RBI / SEBI / HDFC specific) questions.
- **Free-text Q&A** — users can also type their own question.
- **Source transparency** — every answer shows which documents were used, plus an expandable view of the exact source chunks with page numbers.
- **Live metrics** — sidebar/header shows number of documents, total pages, and total chunks indexed.

---

## 🧱 How It Works (RAG Pipeline)

1. **Load** — Each PDF is loaded page-by-page using `PyPDFLoader`.
2. **Split** — Pages are split into ~800-character chunks (100-character overlap) using `RecursiveCharacterTextSplitter`.
3. **Tag** — Each chunk is tagged with metadata (`source_doc`, `source`) identifying which report it came from.
4. **Embed** — Chunks are embedded using the `sentence-transformers/all-MiniLM-L6-v2` model (CPU, normalized embeddings).
5. **Index** — All chunks from all three documents are combined into **one FAISS vector store**.
6. **Retrieve** — On each query, the top **6** most relevant chunks (across all documents) are retrieved.
7. **Generate** — The retrieved context is passed to an LLM along with the question to produce a concise, professional, grounded answer.
8. **Cite** — The app displays which document(s) and page(s) the answer was drawn from.

All expensive steps (embedding model, LLM client, document indexing) are cached with `@st.cache_resource` so they only run once per app session.

---

## 🔀 Two App Versions

This repo contains two interchangeable implementations of the same app:

### `app.py` — Groq-powered (recommended)
| | |
|---|---|
| **LLM provider** | [Groq](https://groq.com/) |
| **Model** | `llama-3.3-70b-versatile` |
| **Required secret** | `GROQ_API_KEY` |
| **Integration style** | Direct Groq SDK call (custom prompt + `chat.completions.create`) |
| **Why use it** | Fast, free tier available, stable — avoids HuggingFace Inference endpoint reliability issues |

### `rag_app.py` — HuggingFace Hub–powered
| | |
|---|---|
| **LLM provider** | HuggingFace Inference API |
| **Model** | `mistralai/Mistral-7B-Instruct-v0.2` |
| **Required secret** | `HF_TOKEN` |
| **Integration style** | LangChain `RetrievalQA` chain (`chain_type="stuff"`) |
| **Why use it** | Fully open-source stack via LangChain abstractions |

Both versions share the same UI, retrieval pipeline, embedding model, and example questions — only the LLM backend and prompting mechanism differ.

---

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Orchestration**: LangChain
- **Vector store**: FAISS
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Llama 3.3 70B (via Groq) *or* Mistral 7B Instruct (via HuggingFace Hub)
- **PDF parsing**: `PyPDFLoader`
- **Hosting**: Hugging Face Spaces

---

## 📂 Project Structure

```
.
├── app.py              # Main app (Groq + Llama 3.3 70B)
├── rag_app.py          # Alternative app (HuggingFace Hub + Mistral 7B)
├── rbi_report.pdf       # RBI Annual Report 2024-25
├── sebi_report.pdf      # SEBI Annual Report 2024-25
├── hdfc_report.pdf      # HDFC Annual Report 2024-25
└── requirements.txt     # Python dependencies
```

> ⚠️ The three PDF files must sit in the **same directory** as the app file and be named exactly `rbi_report.pdf`, `sebi_report.pdf`, and `hdfc_report.pdf` (or update the `DOCUMENTS` list in the script to match your filenames).

---

## ⚙️ Setup & Deployment (Hugging Face Spaces)

1. **Create a new Space**
   - Space type: **Streamlit**
   - Add `app.py` (or `rag_app.py`) as your main file, plus the three PDFs.

2. **Add `requirements.txt`** with at least:
   ```
   streamlit
   langchain
   langchain-community
   langchain-text-splitters
   sentence-transformers
   faiss-cpu
   pypdf
   groq                # only needed for app.py
   ```

3. **Add your secret API key**
   - Go to **Space Settings → Secrets**
   - For `app.py`: add `GROQ_API_KEY` (get one free at [console.groq.com](https://console.groq.com))
   - For `rag_app.py`: add `HF_TOKEN` (get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

4. **Deploy** — the Space will build automatically. On first load it will index all three PDFs (this is cached, so subsequent loads are fast).

---

## 💻 Running Locally

```bash
# Clone the repo
git clone <your-repo-url>
cd <your-repo>

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GROQ_API_KEY="your-key-here"      # for app.py
# or
export HF_TOKEN="your-token-here"        # for rag_app.py

# Run the app
streamlit run app.py
# or
streamlit run rag_app.py
```

---

## 🗣️ Example Questions

**Cross-document:**
- What are the key economic risks mentioned across all reports?
- How does HDFC's business performance align with RBI's monetary policy?
- What were the major regulatory changes in 2024-25?

**RBI-specific:**
- What is RBI's stance on inflation?
- What was the repo rate decision?

**SEBI-specific:**
- What new regulations did SEBI introduce?
- How did market volumes change?

**HDFC-specific:**
- What was HDFC's net interest income?
- How did HDFC's loan book grow in 2024-25?

---

## 🔧 Customization

- **Swap documents**: update the `DOCUMENTS` list (`name`, `file`, `source`) to point to different PDFs.
- **Change chunking**: adjust `chunk_size` / `chunk_overlap` in `RecursiveCharacterTextSplitter`.
- **Change retrieval depth**: adjust `search_kwargs={"k": 6}` to retrieve more/fewer chunks per query.
- **Swap the LLM**: edit `load_llm()` (Groq model name in `app.py`, or `repo_id` in `rag_app.py`).

---

## 👩‍💻 Author

**Mousumi Kundu**
Data Science | Gen AI | LangChain | RAG

- [GitHub](https://github.com/mousumi0905)
- [LinkedIn](https://www.linkedin.com/in/mousumi-kundu-00109819b/)

---

## 📜 Disclaimer

This tool is intended for informational and educational purposes. Answers are generated by an LLM based on retrieved document excerpts and may not be fully accurate — always verify against the original source reports for financial or regulatory decisions.
