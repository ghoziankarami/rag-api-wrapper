# RAG API Wrapper

An Express API for paper search, browsing, corpus statistics, and generated
answers. A Python provider reads a ChromaDB corpus and optional supporting
metadata. This repository includes the service code, not the indexed papers.

## Requirements

- Node.js 22+ and npm.
- Python 3.10+ with the packages in `requirements.txt`.
- A populated ChromaDB directory with `papers` and `papers_summary` collections.
- A Z.ai API key if generated answers are required.

The service cannot reproduce the hosted corpus from a source checkout alone.

## Install and configure

```bash
git clone https://github.com/ghoziankarami/rag-api-wrapper.git
cd rag-api-wrapper
npm ci
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your paths and keys, then run `npm start` from the activated
Python environment. The server binds to `127.0.0.1:3004`. Keep the virtual
environment active so the provider uses the installed Python dependencies.

| Variable | Purpose |
| --- | --- |
| `PORT` | Local API port; default 3004. |
| `RAG_API_KEY` | Key used by remote callers. Set a unique value. |
| `RAG_WORKSPACE` | Workspace root for supporting metadata. |
| `RAG_PAPERS_DB` | ChromaDB corpus directory. |
| `RAG_OBSIDIAN_PAPERS_DIR` | Optional Markdown paper summaries. |
| `RAG_ACTIVE_PDF_ROOT` | Optional PDF directory used by coverage checks. |
| `RAG_TRACKER_PATH` | Optional paper-tracker JSON file. |
| `RAG_PARITY_STATE` | Optional corpus parity-state JSON file. |
| `ZAI_API_KEY`, `ZAI_URL`, `ZAI_MODEL` | Optional answer-provider configuration. |
| `RAG_STATS_TTL_MS` | Statistics cache lifetime; default 60000. |

Legacy path defaults remain for the existing hosted installation. Set the path
variables explicitly for a new installation. Keep `.env` and corpus files out
of Git.

## API

All routes are under `/api/rag`.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Service and corpus readiness. |
| GET | `/stats` | Corpus statistics. |
| GET | `/browse?page=1&limit=20` | Paginated paper list. |
| POST | `/search` | Search with `query` and optional `top_k`. |
| POST | `/answer` | Answer with `query`, optional `top_k`, and optional conversation `history`. |

```bash
curl http://127.0.0.1:3004/api/rag/health
curl -X POST http://127.0.0.1:3004/api/rag/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"kriging","top_k":5}'
```

Loopback requests bypass API-key checks. Other callers must send
`X-API-Key`; the health route also passes through that middleware.
A health response can report `degraded` with HTTP 200 when the corpus is
unavailable, so monitor its JSON status as well as the HTTP response.

## Deployment and limitations

Use a reverse proxy for TLS and restrict access to the loopback service.
The current middleware trusts one proxy hop and exempts loopback callers;
review that behaviour before forwarding public traffic. This service has no
user accounts or per-user data isolation.

Provider calls run synchronously in the Express process. Large corpora or slow
answer requests can block other requests; measure workload before production use.
Generated answers require source review. Paper rights are separate from the
software licence.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
The code is licensed under [MIT](LICENSE).
