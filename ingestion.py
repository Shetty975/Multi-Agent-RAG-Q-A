# ingestion.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter


def build_vector_store(doc_dir="docs", index_path="faiss_index"):
    documents = []
    for file in os.listdir(doc_dir):
        if file.endswith('.txt'):
            loader = TextLoader(os.path.join(doc_dir, file))
            documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

    print(f"Vector index built and saved to `{index_path}/`")
