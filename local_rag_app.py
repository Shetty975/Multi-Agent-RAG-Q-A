# import os
# import faiss
# import streamlit as st
# import logging

# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# # Set Streamlit upload size limit to 1000 MB (if needed)
# st.set_option('server.maxUploadSize', 1000)

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# # Load local embedding model
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Load local QA model
# qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# # Helper to split document into chunks
# def chunk_text(text, max_length=300):
#     sentences = text.split('. ')
#     chunks, chunk = [], ""
#     for sentence in sentences:
#         if len(chunk) + len(sentence) < max_length:
#             chunk += sentence + '. '
#         else:
#             chunks.append(chunk.strip())
#             chunk = sentence + '. '
#     if chunk:
#         chunks.append(chunk.strip())
#     return chunks

# # Upload and index documents
# def process_documents(files):
#     texts, metadatas = [], []
#     for file in files:
#         if file.size > 1 * 1024 * 1024:  # Limit to 1MB per file
#             st.warning(f"{file.name} is too large. Please upload files smaller than 1MB.")
#             continue

#         content = file.read().decode("utf-8")
#         chunks = chunk_text(content)
#         texts.extend(chunks)
#         metadatas.extend([{"source": file.name}] * len(chunks))
#     return texts, metadatas

# # Build FAISS vector store
# def build_vector_store(texts):
#     embeddings = embed_model.encode(texts)
#     index = faiss.IndexFlatL2(embeddings[0].shape[0])
#     index.add(embeddings)
#     return index, embeddings

# # Search top K documents
# def retrieve_top_k(query, texts, index, embeddings, k=3):
#     query_vec = embed_model.encode([query])
#     _, indices = index.search(query_vec, k)
#     return [texts[i] for i in indices[0]]

# # Streamlit UI
# st.title("🧠 Local RAG Q&A Assistant")

# uploaded_files = st.file_uploader("Upload documents (TXT/MD under 1MB)", type=["txt", "md"], accept_multiple_files=True)

# if uploaded_files:
#     texts, metadatas = process_documents(uploaded_files)
    
#     if texts:
#         index, embeddings = build_vector_store(texts)
#         st.success("✅ Documents processed and indexed.")

#         query = st.text_input("Ask a question:")
#         if query:
#             top_chunks = retrieve_top_k(query, texts, index, embeddings)
#             context = " ".join(top_chunks)

#             result = qa_model(question=query, context=context)

#             st.markdown("**🔎 Retrieved Chunks:**")
#             for i, chunk in enumerate(top_chunks, 1):
#                 st.markdown(f"**Chunk {i}:** {chunk}")

#             st.markdown("### 🤖 Answer:")
#             st.success(result['answer'])
#     else:
#         st.warning("No valid documents to process.")
# rag_app_enhanced.py

import os
import faiss
import streamlit as st
import logging
import re

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.schema import Document
from PyPDF2 import PdfReader
import docx
import pandas as pd
from io import StringIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load local embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load local QA model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Helper to split document into chunks
def chunk_text(text, max_length=300):
    sentences = re.split(r'[.!?]\s+', text)
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += sentence + '. '
        else:
            chunks.append(chunk.strip())
            chunk = sentence + '. '
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Read different file types
def read_file(file):
    ext = os.path.splitext(file.name)[-1].lower()
    if ext == '.pdf':
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == '.docx':
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == '.csv':
        df = pd.read_csv(file)
        return df.to_string(index=False)
    elif ext in ['.txt', '.md']:
        return file.read().decode("utf-8")
    else:
        return ""

# Upload and index documents
def process_documents(files):
    texts, metadatas = [], []
    for file in files:
        content = read_file(file)
        if content:
            chunks = chunk_text(content)
            texts.extend(chunks)
            metadatas.extend([{"source": file.name}] * len(chunks))
    return texts, metadatas

# Build FAISS vector store
def build_vector_store(texts):
    embeddings = embed_model.encode(texts)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)
    return index, embeddings

# Search top K documents
def retrieve_top_k(query, texts, index, k=3):
    query_vec = embed_model.encode([query])
    _, indices = index.search(query_vec, k)
    return [texts[i] for i in indices[0]]

# Streamlit app
st.set_page_config(page_title="🧠 Enhanced RAG Q&A Assistant")
st.title("🧠 Enhanced RAG Q&A Assistant")

uploaded_files = st.file_uploader("Upload documents", type=["txt", "md", "pdf", "docx", "csv"], accept_multiple_files=True)

if uploaded_files:
    texts, metadatas = process_documents(uploaded_files)
    index, embeddings = build_vector_store(texts)
    st.success("✅ Documents processed and indexed.")

    query = st.text_input("Ask a question:")
    if query:
        top_chunks = retrieve_top_k(query, texts, index)
        context = " ".join(top_chunks)

        result = qa_model(question=query, context=context)
        st.markdown("**🔎 Retrieved Chunks:**")
        for i, chunk in enumerate(top_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")

        st.markdown("### 🤖 Answer:")
        st.success(result['answer'])

        if st.button("📥 Download Answer"):
            st.download_button(
                label="Download as Text File",
                data=result['answer'],
                file_name="answer.txt",
                mime="text/plain"
            )
