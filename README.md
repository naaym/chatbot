# Academic RAG Chatbot (FastAPI + ChromaDB + Gemini)

## Architecture
- **FastAPI** exposes REST endpoints for search and chat.
- **ChromaDB** stores FAQ documents with sentence-transformer embeddings in persistent mode.
- **Gemini (gemini-pro)** generates answers strictly from retrieved documents.
- **Dataset** is read locally from `data/Ecommerce_FAQ_Chatbot_dataset.csv` (no downloads).

## Project Structure
```
app/
  main.py        # FastAPI app and endpoints
  rag.py         # Dataset loader, ChromaDB setup, search, Gemini response
data/
  Ecommerce_FAQ_Chatbot_dataset.csv
Dockerfile
docker-compose.yml
requirements.txt
README.md
```

## Installation (Local)
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY="YOUR_KEY"
   ```
4. Run the API:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Docker (Recommended)
1. Export the API key:
   ```bash
   export GEMINI_API_KEY="YOUR_KEY"
   ```
2. Build and run:
   ```bash
   docker compose up --build
   ```

## API Endpoints
- `GET /` → returns API version and Swagger location.
- `POST /search` → returns top 5 relevant documents from ChromaDB.
- `POST /chat` → returns retrieved documents and Gemini response.

### Example `POST /search`
```json
{
  "query": "What payment methods do you accept?"
}
```

### Example `POST /chat`
```json
{
  "query": "What payment methods do you accept?"
}
```

Response:
```json
{
  "query": "What payment methods do you accept?",
  "top_documents": ["Question: ... Answer: ..."],
  "response": "We accept credit cards and PayPal."
}
```
