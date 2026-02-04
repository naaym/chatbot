"""RAG utilities for loading data, storing in ChromaDB, and answering queries."""
from __future__ import annotations

import os
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

DATASET_PATH = "data/Ecommerce_FAQ_Chatbot_dataset.csv"
COLLECTION_NAME = "ecommerce_faq"
PERSIST_DIR = "chroma_db"


def load_dataset(path: str = DATASET_PATH) -> List[str]:
    """Load the CSV dataset and return formatted documents."""
    dataframe = pd.read_csv(path)
    documents: List[str] = []
    for _, row in dataframe.iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        documents.append(f"Question: {question}\nAnswer: {answer}")
    return documents


def get_collection() -> chromadb.Collection:
    """Initialize ChromaDB in persistent mode and return a collection."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return collection


def populate_vector_store(collection: chromadb.Collection, documents: List[str]) -> None:
    """Insert documents into the ChromaDB collection if empty."""
    if collection.count() > 0:
        return

    ids = [f"faq_{index}" for index in range(len(documents))]
    collection.add(documents=documents, ids=ids)


def search_documents(collection: chromadb.Collection, query: str, top_k: int = 5) -> List[str]:
    """Search the collection and return top matching documents."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return results.get("documents", [[]])[0]


def generate_answer(query: str, documents: List[str]) -> str:
    """Generate an answer using Gemini based on retrieved documents."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")

    context = "\n\n".join(documents)
    prompt = (
        "You are an academic assistant. Answer the user question using ONLY the "
        "information in the provided documents. Do NOT add external knowledge or "
        "make up answers. If the answer is not in the documents, say you do not know.\n\n"
        f"Documents:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    response = model.generate_content(prompt)
    return response.text.strip()
