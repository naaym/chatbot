# Chatbot RAG E-commerce (FastAPI + ChromaDB + LLM)

## Objectif
Ce projet implémente un chatbot RAG (Retrieval Augmented Generation) pour un examen académique. Il interroge un dataset FAQ e-commerce (Kaggle), stocke les documents dans ChromaDB et génère des réponses via un LLM (Gemini / OpenRouter / Ollama).

## Architecture
- **FastAPI** : API REST (endpoints /, /search, /chat) + Swagger (/docs).
- **ChromaDB** : base vectorielle persistante en mode embedded.
- **Sentence-Transformers** : embeddings sémantiques.
- **LLM** : génération contrôlée (réponses basées uniquement sur les documents).

```
[Client] -> [FastAPI] -> [ChromaDB] -> [Top 5 documents] -> [LLM] -> [Réponse]
```

## Dataset (obligatoire)
Téléchargez **exclusivement** le dataset Kaggle :
- https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset

Placez le fichier CSV dans `data/` et nommez-le :
```
/workspace/chatbot/data/ecommerce_faq.csv
```
Le CSV doit contenir deux colonnes : `question` et `answer`.

## Installation locale (optionnel)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer le projet (Docker recommandé)
```bash
docker-compose up --build
```
L'API est disponible sur :
- http://localhost:8000
- Swagger : http://localhost:8000/docs

## Endpoints
### `GET /`
Réponse :
```json
{
  "version": "1.0.0",
  "swagger": "/docs"
}
```

### `POST /search`
Entrée :
```json
{ "query": "What payment method do you accept please?" }
```
Sortie : top 5 documents (FAQ) les plus proches.

### `POST /chat`
Entrée :
```json
{ "query": "What payment method do you accept please?" }
```
Sortie :
```json
{
  "query": "...",
  "top_documents": [ ... ],
  "response": "..."
}
```

## Configuration (LLM)
Par défaut, le projet utilise **Ollama** via Docker.
Variables d'environnement principales :
- `LLM_PROVIDER`: `ollama` | `openrouter` | `gemini`
- `LLM_MODEL`: nom du modèle (ex: `llama3`, `gpt-4o-mini`, `gemini-1.5-pro`)
- `OLLAMA_BASE_URL`: `http://ollama:11434`
- `OPENROUTER_API_KEY`
- `GEMINI_API_KEY`

## Choix techniques (explication académique)
- **RAG** : permet d'ancrer les réponses dans des documents vérifiés, réduisant les hallucinations.
- **ChromaDB** : base vectorielle légère, adaptée à des prototypes académiques et dockerisable.
- **Sentence-Transformers** : embeddings efficaces pour la recherche sémantique.
- **LLM contrôlé** : prompt strict, réponses uniquement basées sur les documents récupérés.

## Structure du projet
```
app/
  config.py
  main.py
  rag.py
  __init__.py
Dockerfile
docker-compose.yml
requirements.txt
README.md
```
