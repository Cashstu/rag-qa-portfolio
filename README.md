# RAG Q&A API

A retrieval-augmented generation Q&A service: ask a question, it retrieves the most
relevant passages from a small corpus of foundational NLP/LLM papers via semantic
search, and asks Gemini to answer grounded strictly in that retrieved context —
citing which paper and page it came from.

Built as an end-to-end portfolio project: LangChain + Google Gemini for the RAG
pipeline, ChromaDB for the vector store, Flask for the API, Docker + Cloud Run for
deployment, and GitHub Actions (with keyless Workload Identity Federation) for CI/CD.

## How it works

```
question ──► Chroma similarity search (top-k chunks) ──┬──► answer (Gemini, grounded in context)
                                                          └──► sources (paper + page)
```

The corpus is five open-access arXiv papers on retrieval and language models
(Attention Is All You Need, the original RAG paper, BERT, Dense Passage Retrieval,
GPT-3). On first use, the app loads and chunks the PDFs, embeds each chunk with
Gemini's `text-embedding-004`, and persists the result in a local Chroma store. Every
later question retrieves the top-k most similar chunks and asks
`gemini-2.5-flash` to answer using only that context — the prompt explicitly
instructs it not to answer from outside knowledge.

See [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) for the full build history: every
decision made, why, alternatives considered, and one real bug that local Docker
testing caught before it reached production.

## API

**`GET /health`** — liveness check, does not touch Gemini or the vector store.
```
curl https://<cloud-run-url>/health
# {"status": "ok"}
```

**`POST /ask`** — ask a question, get a grounded answer with sources.
```
curl -X POST https://<cloud-run-url>/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What problem does retrieval-augmented generation solve?"}'
```
```json
{
  "answer": "According to the retrieved context, RAG combines a pretrained parametric...",
  "sources": ["retrieval_augmented_generation.pdf", "dense_passage_retrieval.pdf"]
}
```

## Architecture decisions

- **Lazy ingestion, not a baked-in or committed vector store.** Cloud Run instances
  have ephemeral local disk, so a persisted store can't simply ship in the image and
  survive across deploys. Instead, the vector store is built on first use (a few
  seconds, ~5 small papers) and cached for the life of the container instance. A
  production system at real scale would back this with an external/managed vector
  store instead of re-embedding per cold start.
- **`GET /health` never touches Gemini or Chroma.** It's a pure liveness check, kept
  fully independent from the RAG chain (`app/main.py`'s `get_rag_chain()` builds the
  chain lazily on the *first* `/ask`, not at import or startup time). This was
  validated by an actual bug: an earlier version of `entrypoint.sh` ran ingestion
  eagerly before starting the server, which meant a bad API key took down health
  checks along with everything else. See the build log for the full story.
- **Two GCP service accounts, not one.** The Cloud Run runtime identity can only read
  the `GOOGLE_API_KEY` secret — nothing else. The GitHub Actions deploy identity can
  push images and deploy revisions — but can't read that secret. A vulnerability in
  the running app therefore can't be used to redeploy the service, and a compromised
  CI pipeline can't exfiltrate the Gemini key.
- **Keyless CI/CD.** GitHub Actions authenticates to GCP via Workload Identity
  Federation — OIDC token exchange scoped to this exact repo — instead of a
  long-lived service-account JSON key stored as a GitHub secret.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env
# edit .env and set GOOGLE_API_KEY (https://aistudio.google.com/app/apikey)

python scripts/download_corpus.py   # fetches the 5 papers into data/raw/

pytest                              # fully mocked, no network/API key needed
ruff check .

python -m flask --app app.main run --port 8080
# first POST /ask will take a few seconds while it builds the vector store
curl -X POST localhost:8080/ask -H "Content-Type: application/json" \
  -d '{"question": "What is attention in a transformer?"}'
```

## Running with Docker

```bash
docker build -t rag-qa-api .
docker run -p 8080:8080 --env-file .env rag-qa-api
curl localhost:8080/health
```

## Deployment

Infrastructure (GCP project, Artifact Registry repo, Secret Manager secret, two
service accounts, Workload Identity Federation) is provisioned once via `gcloud` —
see `docs/BUILD_LOG.md` for the exact commands and rationale. After that, every push
to `main` runs tests and, if they pass, builds the image, pushes it to Artifact
Registry, and deploys it to Cloud Run via `.github/workflows/deploy.yml`.

One-time manual steps before the pipeline can deploy:
1. Add the real Gemini key to Secret Manager (run this in your own terminal, not
   through an AI assistant, so the key never appears in any chat/tool transcript):
   ```bash
   printf '%s' 'YOUR_REAL_KEY' | gcloud secrets versions add GOOGLE_API_KEY --data-file=- --project=rag-qa-portfolio-02173
   ```
2. Set these as GitHub Actions **repository variables** (Settings → Secrets and
   variables → Actions → Variables) — not secrets, since Workload Identity
   Federation needs no secret material on the GitHub side:
   | Variable | Value |
   |---|---|
   | `GCP_PROJECT_ID` | `rag-qa-portfolio-02173` |
   | `ARTIFACT_REPO` | `rag-qa-repo` |
   | `WIF_PROVIDER` | `projects/896614942432/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
   | `DEPLOY_SERVICE_ACCOUNT` | `rag-qa-deployer@rag-qa-portfolio-02173.iam.gserviceaccount.com` |

## Tech stack

Python 3.12 · Flask · gunicorn · LangChain · `langchain-google-genai` (Gemini
`gemini-2.5-flash` + `text-embedding-004`) · ChromaDB · Docker · Google Cloud Run ·
Google Secret Manager · GitHub Actions (Workload Identity Federation) · pytest · ruff
