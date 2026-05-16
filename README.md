# RAG API Wrapper

> Express API wrapper for RAG vector database — serves paper search, browse, and LLM answers.

## Quick Start

```bash
npm install
cp .env.example .env
npm start
```

## Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/rag/health` | GET | No | Health check |
| `/api/rag/stats` | GET | API Key | Corpus stats |
| `/api/rag/browse` | GET | API Key | Browse papers |
| `/api/rag/search` | POST | API Key | Semantic search |
| `/api/rag/answer` | POST | API Key | Query + LLM answer |

## Configuration

```env
PORT=3004
RAG_API_HOST=127.0.0.1
RAG_API_KEY=your-api-key
OPENROUTER_API_KEY=sk-or-...
```

## License

MIT
