"""FastAPI application for the academic RAG chatbot."""
from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import (
    get_collection,
    load_dataset,
    populate_vector_store,
    search_documents,
    generate_answer,
)

app = FastAPI(title="Academic RAG Chatbot", version="1.0.0")

collection = get_collection()
documents = load_dataset()
populate_vector_store(collection, documents)


class SearchRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    top_documents: list[str]
    response: str


@app.get("/")
def root():
    return {"version": "1.0.0", "swagger": "/docs"}


@app.post("/search")
def search(request: SearchRequest):
    top_docs = search_documents(collection, request.query, top_k=5)
    return top_docs


@app.post("/chat", response_model=ChatResponse)
def chat(request: SearchRequest):
    top_docs = search_documents(collection, request.query, top_k=5)
    answer = generate_answer(request.query, top_docs)
    return ChatResponse(query=request.query, top_documents=top_docs, response=answer)
