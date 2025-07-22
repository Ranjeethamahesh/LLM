import os
import hashlib
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.llms import Ollama
from langchain_community.document_loaders import UnstructuredFileLoader

# ------------------ CONFIG ------------------ #
MAX_MEMORY_LENGTH = 5
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={"device": "cpu"})

# ------------------ PAGE SETUP ------------------ #
st.set_page_config(page_title="📄 Document Q&A App", layout="centered")
st.title("📄 Ask Questions from Your Documents")

# ------------------ SESSION STATE ------------------ #
if "qa_memory_all" not in st.session_state:
    st.session_state.qa_memory_all = {}

if "all_docs" not in st.session_state:
    st.session_state.all_docs = []

if "indexed_file_hashes" not in st.session_state:
    st.session_state.indexed_file_hashes = set()

if "combined_vectorstore" not in st.session_state:
    st.session_state.combined_vectorstore = None
    
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ------------------ UTILS ------------------ #
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def parse_and_chunk_file(file, file_bytes, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = UnstructuredFileLoader(temp_path, strategy="ocr_only")
        documents = loader.load()
    except Exception as e:
        st.error(f"❌ Failed to load {filename}: {e}")
        return []

    for doc in documents:
        doc.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = chunk.metadata.get("source", filename)

    return chunks

# ------------------ FILE UPLOAD ------------------ #
uploaded_files = st.file_uploader("📂 Upload PDFs (scanned/digital) or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_bytes = file.read()
        file_hash = get_file_hash(file_bytes)

        if file_hash not in st.session_state.indexed_file_hashes:
            chunks = parse_and_chunk_file(file, file_bytes, file.name)

            if chunks:
                st.session_state.all_docs.extend(chunks)
                st.session_state.indexed_file_hashes.add(file_hash)

                if st.session_state.combined_vectorstore:
                    st.session_state.combined_vectorstore.add_documents(chunks)
                else:
                    st.session_state.combined_vectorstore = FAISS.from_documents(chunks, embedding_model)

                # ✅ Q&A chain is (re)built only when new chunks are added
                llm = Ollama(model="llama3.2:3b", base_url="http://host.docker.internal:11434")
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state.combined_vectorstore.as_retriever(),
                    return_source_documents=True
                )

            st.success(f"✅ Indexed: {file.name}")
        else:
            st.info(f"♻️ Already processed: {file.name}")

# ------------------ QA CHAIN USAGE ------------------ #
query = st.text_input("❓ Ask a question")

if query:
    # ------------------ CHECK MEMORY FIRST ------------------ #
    if query in st.session_state.qa_memory_all:
        st.markdown("### 📌 Answer (from memory)")
        answer_text = st.session_state.qa_memory_all[query]["answer"]
        source = st.session_state.qa_memory_all[query]["source"]
        st.write(answer_text)
        st.markdown(f"📎 **Source**\n{source}")

    # ------------------ IF NOT IN MEMORY → RUN QA ------------------ #
    else:
        if st.session_state.get("qa_chain") is None:
            st.warning("⚠️ Please upload and index documents first.")
        else:
            result = st.session_state.qa_chain(query)
            answer_text = result["result"]
            source_docs = result.get("source_documents", [])

            # Show Answer
            st.markdown("### 📌 Answer")
            st.write(answer_text)

            # Show Source if applicable
            generic_phrases = [
                "I don't know",
                "I couldn't find",
                "no relevant",
                "not enough information"
            ]

            if (
                not any(phrase in answer_text.lower() for phrase in generic_phrases)
                and source_docs
            ):
                first_source = source_docs[0].metadata.get("source", "Unknown")
                st.markdown(f"📎 **Source**\n{first_source}")

                # Save to memory
                if len(st.session_state.qa_memory_all) >= MAX_MEMORY_LENGTH:
                    # Remove oldest
                    oldest = list(st.session_state.qa_memory_all.keys())[0]
                    del st.session_state.qa_memory_all[oldest]
                st.session_state.qa_memory_all[query] = {
                    "answer": answer_text,
                    "source": first_source
                }

# ------------------ PREVIOUS QA MEMORY DISPLAY ------------------ #
if st.session_state.qa_memory_all:
    st.markdown("### 📝 Previous Questions:")
    for q, info in reversed(st.session_state.qa_memory_all.items()):
        with st.expander(q):
            st.write(info['answer'])
            st.caption(f"📎 Source: {info['source']}")