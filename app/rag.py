from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from fastapi import HTTPException

from app.config import Settings


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict


def load_faq_documents(dataset_path: str) -> list[Document]:
    candidate_paths = [
        dataset_path,
        "data/ecommerce_faq.csv",
        "/app/data/ecommerce_faq.csv",
    ]
    data_frame = None
    for candidate in candidate_paths:
        try:
            data_frame = pd.read_csv(candidate)
            break
        except FileNotFoundError:
            continue
    if data_frame is None:
        tried_paths = ", ".join(candidate_paths)
        raise FileNotFoundError(
            "Dataset file not found. Please download the Kaggle dataset and place "
            f"the CSV at one of the expected paths: {tried_paths}."
        )
    normalized_columns = {column.lower().strip(): column for column in data_frame.columns}
    question_column = normalized_columns.get("question") or normalized_columns.get("questions")
    answer_column = normalized_columns.get("answer") or normalized_columns.get("answers")

    if not question_column or not answer_column:
        raise ValueError(
            "Dataset must contain 'question' and 'answer' columns. "
            f"Found columns: {list(data_frame.columns)}"
        )

    documents: list[Document] = []
    for index, row in data_frame.iterrows():
        question = str(row[question_column]).strip()
        answer = str(row[answer_column]).strip()
        if not question or not answer:
            continue
        text = f"Question: {question}\nReponse: {answer}"
        documents.append(
            Document(
                doc_id=str(index),
                text=text,
                metadata={"question": question, "answer": answer},
            )
        )
    return documents


def build_collection(settings: Settings, documents: Iterable[Document]):
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    client = chromadb.Client(
        ChromaSettings(
            is_persistent=True,
            persist_directory=settings.chroma_persist_dir,
        )
    )
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    documents = list(documents)
    if not documents:
        raise ValueError("No documents available to populate the vector store.")

    if collection.count() < len(documents):
        collection.add(
            ids=[doc.doc_id for doc in documents],
            documents=[doc.text for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )
    return collection


def retrieve_documents(collection, query: str, top_k: int = 5) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    payloads: list[dict] = []
    for doc_text, metadata in zip(documents, metadatas):
        payloads.append(
            {
                "text": doc_text,
                "metadata": metadata,
            }
        )
    return payloads


def generate_answer(settings: Settings, query: str, top_documents: list[dict]) -> str:
    if not top_documents:
        return "Je suis desole, aucune information pertinente n'a ete trouvee."

    context = "\n\n".join(
        [f"Document {index + 1}: {doc['text']}" for index, doc in enumerate(top_documents)]
    )

    system_prompt = (
        "Tu es un assistant e-commerce. Reponds uniquement avec les informations "
        "presentes dans les documents. Si la reponse est inconnue, dis que "
        "l'information n'est pas disponible."
    )
    user_prompt = (
        f"Question utilisateur: {query}\n\nDocuments:\n{context}\n\nReponse:"
    )

    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return _call_ollama(settings, system_prompt, user_prompt)
    if provider == "openrouter":
        return _call_openrouter(settings, system_prompt, user_prompt)
    if provider == "gemini":
        return _call_gemini(settings, system_prompt, user_prompt)

    raise HTTPException(status_code=400, detail="Unsupported LLM provider")


def _call_ollama(settings: Settings, system_prompt: str, user_prompt: str) -> str:
    response = requests.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=response.text)
    payload = response.json()
    return payload.get("message", {}).get("content", "").strip()


def _call_openrouter(settings: Settings, system_prompt: str, user_prompt: str) -> str:
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is missing")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=response.text)
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def _call_gemini(settings: Settings, system_prompt: str, user_prompt: str) -> str:
    if not settings.gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing")
    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.llm_model}:generateContent",
        params={"key": settings.gemini_api_key},
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}"},
                    ],
                }
            ]
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=response.text)
    payload = response.json()
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""
    return candidates[0]["content"]["parts"][0]["text"].strip()
