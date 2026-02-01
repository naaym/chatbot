from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.rag import build_collection, generate_answer, load_faq_documents, retrieve_documents

app = FastAPI(title="E-commerce FAQ RAG", version="1.0.0")
settings = get_settings()
collection = None


class QueryRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    query: str
    top_documents: list[dict]


class ChatbotResponse(BaseModel):
    query: str
    top_documents: list[dict]
    response: str


@app.on_event("startup")
def startup_event() -> None:
    global collection
    documents = load_faq_documents(settings.dataset_path)
    collection = build_collection(settings, documents)


@app.get("/")
def root() -> dict:
    return {"version": settings.app_version, "swagger": "/docs"}


@app.post("/search", response_model=SearchResponse)
def search_docs(payload: QueryRequest) -> SearchResponse:
    if not collection:
        raise HTTPException(status_code=500, detail="Vector store is not ready")
    top_documents = retrieve_documents(collection, payload.query, top_k=5)
    return SearchResponse(query=payload.query, top_documents=top_documents)


@app.post("/chat", response_model=ChatbotResponse)
def chat(payload: QueryRequest) -> ChatbotResponse:
    if not collection:
        raise HTTPException(status_code=500, detail="Vector store is not ready")
    top_documents = retrieve_documents(collection, payload.query, top_k=5)
    response = generate_answer(settings, payload.query, top_documents)
    return ChatbotResponse(
        query=payload.query,
        top_documents=top_documents,
        response=response,
    )
