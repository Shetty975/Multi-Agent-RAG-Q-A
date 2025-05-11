# retriever.py
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def retrieve_top_k(query, k=3, index_path="faiss_index"):
    vectorstore = FAISS.load_local(index_path, OpenAIEmbeddings())
    return vectorstore.similarity_search(query, k=k)
