# Build Log

A running, chronological record of what was built, why, and what alternatives were
considered along the way. Kept as supplementary documentation alongside the README —
useful for revisiting design decisions later (e.g. for interview prep) and as a record
of methodology for this portfolio project.

---

## 2026-09-14 — Environment check & scoping

**What:** Checked the working directory (empty except `.venv`, `.idea`) and local
tooling. Found: Python 3.12.3 in a venv, Homebrew available, but no `gcloud`, `docker`,
or `gh` CLI installed.

**Decisions made with the user** (via clarifying questions before writing any code):
- **Corpus**: a small set of open-access arXiv papers on RAG/LLM topics (Attention Is
  All You Need, the original RAG paper, BERT, Dense Passage Retrieval, GPT-3), rather
  than a scraped corpus or user-supplied documents. *Why:* thematically ties the
  project to its own subject matter (good interview story — "a RAG system that
  answers questions about RAG"), license-clean (arXiv is open access), and avoids the
  extra scope of a general-purpose scraper.
- **Gemini access**: Google AI Studio API key via `langchain-google-genai`, not Vertex
  AI. *Why:* Vertex AI requires IAM/service-account plumbing that adds setup cost
  without adding to the RAG/LangChain story this project is meant to demonstrate.
- **Scope**: minimal — one `POST /ask` + `GET /health`, no auth, no ingestion
  endpoint, no frontend. *Why:* keeps the project demonstrable and finishable
  end-to-end rather than half-featured.
- **GCP**: no project exists yet; will be created from scratch with the user running
  the interactive `gcloud auth login` themselves (an agent should not hold or use a
  user's cloud credentials directly).
- **CI/CD auth**: Workload Identity Federation instead of a static service-account
  JSON key. *Why:* current best practice (no long-lived credential to leak), and a
  concrete resume talking point ("keyless CI/CD auth to GCP").

**Plan approved:** full build plan (architecture, file layout, build order) written to
a plan file and approved before any code was written — see the plan-mode transcript
for the complete architecture rationale (persistence strategy, chain design, etc.),
summarized and expanded on below as each piece is actually built.

---

## Dependency choices (`requirements.txt`)

Used range pins (`>=x,<x+1`) rather than exact versions. *Why:* exact pins written
from training knowledge can easily reference a version that no longer resolves or has
been yanked; range pins let `pip` resolve a mutually-compatible set at install time
while still capping at the next major/minor to avoid breaking changes. Notable
choices:
- `langchain-chroma` (not `Chroma` from `langchain-community`) — Chroma integration
  was split into its own package; using the current location avoids deprecation
  warnings.
- `chromadb` pulls in `onnxruntime` as a transitive dependency even though this
  project never uses Chroma's built-in embedding function (Gemini embeddings are used
  explicitly instead) — this makes the Docker image larger than strictly necessary.
  Flagged here as a known optimization opportunity, not fixed, since it doesn't affect
  correctness and swapping vector stores is out of scope for a minimal build.

`pip install -r requirements-dev.txt` resolved cleanly on the first try (langchain
0.3.30, langchain-google-genai 2.1.12, langchain-chroma 0.2.3, chromadb 0.5.23) —
range pins paid off, no manual version-hunting needed.

---

## App code (`app/config.py`, `app/ingest.py`, `app/rag.py`, `app/main.py`)

**Ingestion (`app/ingest.py`)**: `load_or_build_vector_store()` checks whether
`CHROMA_PERSIST_DIR` exists and is non-empty; if so it opens the existing Chroma store,
otherwise it loads every PDF in `RAW_DOCS_DIR` with `PyPDFLoader`, splits with
`RecursiveCharacterTextSplitter` (1000 chars / 150 overlap — small enough to keep
retrieved chunks focused, generous enough overlap that an answer straddling a chunk
boundary doesn't lose context), embeds with `GoogleGenerativeAIEmbeddings
(models/gemini-embedding-001)`, and persists. This function is the one that turns into
"lazy ingestion at container startup" once wired into the entrypoint script.

**RAG chain (`app/rag.py`)**: built as one LCEL pipeline rather than
`retriever.invoke()` + `chain.invoke()` as two separate calls, specifically to avoid
retrieving the same documents twice (once for the answer's context, once for the
reported `sources`). Shape:
```
RunnableParallel(docs=retriever, question=passthrough)
  -> RunnableParallel(answer=(format_docs | prompt | llm | parser), sources=...)
```
Both retrieval and generation happen in a single `.invoke(question)` call, and the
same retrieved `docs` feed both branches. `_format_docs` embeds the source filename
and page number into the context block that's actually shown to Gemini (not just
returned to the caller), so the model can naturally reference "according to the
Attention paper..." if asked to.

**Flask app (`app/main.py`)**: the `RagChain` (which builds/loads the vector store and
constructs the Gemini client) is instantiated lazily on first `/ask` call via a
module-level `_rag_chain` singleton, not at import time or app-factory time. *Why:*
importing `app.main` — which `pytest` and gunicorn's module loader both do — must not
require a valid `GOOGLE_API_KEY` or a working vector store; `/health` in particular
needs to succeed even before the first real question is asked.

## Testing strategy

Both test files avoid any network call or real API key:
- `tests/test_rag.py` swaps the retriever and the LLM for
  `langchain_core.runnables.RunnableLambda` fakes (patched in via
  `app.rag.load_or_build_vector_store` and `app.rag.ChatGoogleGenerativeAI`), so the
  *real* LCEL plumbing — `RunnableParallel`, the prompt template, `StrOutputParser` —
  is still exercised, only the network-touching edges are faked. This catches wiring
  bugs (wrong dict keys between the two `RunnableParallel` stages, for instance) that
  a fully-mocked `RagChain.ask` would miss.
- `tests/test_api.py` patches `app.main.get_rag_chain` directly with a stub object,
  testing Flask routing/validation/error-handling in isolation from the RAG internals.

Result: `pytest` (7 tests) and `ruff check .` both pass locally with zero network
access, so CI can run the test job without any GCP secret.

---

## Docker: local build/run, and a real bug caught by it

**Tooling:** no `docker`, `gcloud`, or `gh` CLI were present locally. Installed via
Homebrew: `colima` + `docker` (CLI-only Docker runtime — chosen over Docker Desktop so
the container could actually be built and run headlessly, without needing a GUI app
opened), plus `google-cloud-sdk` and `gh`.

**Bug found and fixed:** the first version of `entrypoint.sh` ran `python -m
app.ingest` *before* starting gunicorn, as a belt-and-suspenders "make sure the store
exists" step. Testing the built image locally with `docker run -e
GOOGLE_API_KEY=dummy-placeholder` surfaced the problem immediately: ingestion failed
(invalid key against the real Gemini API), the container exited before gunicorn ever
bound to the port, and `GET /health` was unreachable — `curl` got "empty reply from
server." This defeated the entire point of making `/health` not depend on the LLM
(see the Flask app section above): a bad key, a transient network blip, or a Gemini
outage would take down health checks too, which on Cloud Run means the whole revision
fails to become ready and the deploy fails outright, not just the `/ask` path.

**Fix:** removed the eager ingestion step entirely. `app/ingest.py`'s
`load_or_build_vector_store()` is already called lazily from `app/rag.py`'s
`RagChain.__init__`, itself called lazily from `app/main.py`'s `get_rag_chain()` on
the first `/ask` request — so the pre-step in `entrypoint.sh` was both redundant and
actively harmful. `entrypoint.sh` now just execs gunicorn directly.

**Re-verified after the fix**, container run with a deliberately invalid API key:
- `GET /health` → `200 {"status":"ok"}` (works regardless of key validity)
- `POST /ask` with a body → `500 {"error":"failed to generate an answer"}` (the
  Gemini/embedding error is caught and logged, not a crash)
- `POST /ask` with no `question` → `400 {"error":"'question' is required"}`

This is exactly the isolation the design was supposed to have; a live key still needs
to be tested end-to-end for an actual grounded answer (pending — see below).

---

## GCP provisioning

Ran interactively after the user authenticated via `gcloud auth login` and created a
billing account through the console (payment setup can't be scripted). Everything
after that — project creation, API enablement, IAM — was run directly, since it was
already covered by the approved plan.

- **Project**: `rag-qa-portfolio-02173` (project number `896614942432`), billing
  linked to the user's billing account.
- **APIs enabled**: Cloud Run, Artifact Registry, Secret Manager, IAM Credentials,
  Cloud Resource Manager.
- **Artifact Registry**: Docker repo `rag-qa-repo` in `us-central1`. First creation
  attempt hit `PERMISSION_DENIED` on `artifactregistry.repositories.create` despite
  the account holding `roles/owner` — this was IAM/API-enablement propagation lag
  right after `gcloud services enable`, not a real permissions problem. Solved with a
  retry loop (15s backoff) rather than a fixed sleep, since propagation time isn't
  predictable; it succeeded on the second attempt.
- **Two separate service accounts**, deliberately not one:
  - `rag-qa-runtime@...` — what the Cloud Run service itself runs as. Only granted
    `roles/secretmanager.secretAccessor`, and only on the one `GOOGLE_API_KEY`
    secret. This is the identity that would be compromised if the deployed app had a
    vulnerability, so it gets the least possible privilege.
  - `rag-qa-deployer@...` — what GitHub Actions impersonates via WIF to build/push
    images and deploy. Granted `roles/run.admin` and `roles/artifactregistry.writer`
    at the project level, plus `roles/iam.serviceAccountUser` scoped to the runtime
    SA specifically (needed so it's allowed to deploy a Cloud Run revision *running
    as* that SA — `deploy-cloudrun` fails without this).
  *Why split them:* the deploy identity needs broad-ish project permissions but never
  needs to read the Gemini key; the runtime identity needs the key but should never
  be able to redeploy the service. Collapsing them into one SA would give a simple
  code vulnerability in the running app a path to redeploy itself.
- **Secret Manager**: `GOOGLE_API_KEY` secret created (empty — no version added yet).
  Deliberately did not create a version through this session: doing so would mean
  typing the real key into a command that either the assistant runs (putting the key
  in tool-call history) or the user types with a `!` prefix (putting it in the chat
  transcript). Left for the user to add themselves, directly in their own terminal —
  see the deployment section of README.md for the exact command.
- **Workload Identity Federation**: pool `github-pool` + OIDC provider
  `github-provider` trusting `token.actions.githubusercontent.com`, with an
  **attribute condition restricting it to `repository == 'Cashstu/rag-qa-portfolio'`**
  — not just "any token this Google Cloud org trusts," but specifically tokens minted
  for workflow runs in this one repo. The deployer SA's `workloadIdentityUser`
  binding is further scoped with a `principalSet` naming that same repo, so both the
  provider and the binding independently enforce the repo restriction (defense in
  depth — either one alone would already block a token from a different repo).

Resulting identifiers (become GitHub Actions repo *variables*, not secrets, since WIF
needs no secret material on the GitHub side):
```
GCP_PROJECT_ID   = rag-qa-portfolio-02173
ARTIFACT_REPO    = rag-qa-repo
WIF_PROVIDER     = projects/896614942432/locations/global/workloadIdentityPools/github-pool/providers/github-provider
DEPLOY_SERVICE_ACCOUNT = rag-qa-deployer@rag-qa-portfolio-02173.iam.gserviceaccount.com
```

---

## GitHub: repo, push, CI/CD wiring

User authenticated `gh` themselves (`gh auth login`, HTTPS protocol, browser flow) —
same reasoning as `gcloud auth login`: an agent should not hold or use a user's
account credentials directly.

- `git init`, staged everything, double-checked `git status` before committing to
  confirm `.env` (the real API key) was excluded and only `.env.example` was staged —
  it was.
- `gh repo create Cashstu/rag-qa-portfolio --public --source=. --remote=origin` then
  `git push -u origin main`.
- Set the four non-secret identifiers from GCP provisioning as **GitHub Actions repo
  variables** (`gh variable set`) — not secrets, deliberately, since Workload Identity
  Federation means none of them are sensitive (a project ID and service account email
  reveal nothing exploitable without also controlling the OIDC trust relationship).
- The initial push triggered `.github/workflows/deploy.yml` automatically. Watching
  it via `gh run watch` — expect the `test` job to pass (fully mocked, no secrets
  needed) and the `deploy` job to reach Cloud Run but fail at the final step, since
  `GOOGLE_API_KEY` in Secret Manager still has no version (the user hasn't added
  their real key yet — see README's Deployment section for that one remaining manual
  step). Actual outcome recorded below once the run finishes.

**Three CI runs, two real bugs found and fixed by actually running the pipeline
rather than just reading the YAML:**

1. **Run 1** — `test` job passed. `deploy` job got through WIF auth, Docker
   build/push to Artifact Registry, then failed at `gcloud run deploy`:
   ```
   Permission 'iam.serviceaccounts.actAs' denied on service account
   896614942432-compute@developer.gserviceaccount.com
   ```
   The workflow never told `deploy-cloudrun` which service account to run the
   *service* as, so it defaulted to the project's default compute SA — which the
   deployer identity was never granted `iam.serviceAccountUser` on (only the
   purpose-built `rag-qa-runtime` SA got that grant, deliberately, per the
   least-privilege design above). Fix attempt 1: added `service_account:
   rag-qa-runtime@...` to the action's `with:` block.

2. **Run 2** — same exact error, unchanged. Inspecting the actual `gcloud run
   deploy` command the action logged (`gh run view --job <id> --log`) showed it had
   **no `--service-account` flag at all** — the action's `service_account` input
   simply wasn't being translated into the gcloud invocation in this code path.
   Lesson: a GitHub Action's documented input isn't guaranteed to do what its name
   implies for every deployment mode; the log of the actual underlying command it
   runs is the ground truth, and reading the YAML alone would not have caught this.
   Fix: moved it into `flags: --service-account=...` instead — `flags` was already
   confirmed (from run 1's log) to pass through to the real command verbatim.

3. **Run 3** — the service-account fix worked; the logged command now correctly
   includes `--service-account rag-qa-runtime@...`. Deploy got much further (past
   IAM policy setup, into "Creating Revision") and failed with:
   ```
   Secret projects/896614942432/secrets/GOOGLE_API_KEY/versions/latest was not found
   ```
   This is the expected, final, deliberately-not-yet-solved step: the secret exists
   but has zero versions because adding the real Gemini key is left to the user (see
   the Secret Manager note above — a value never gets typed through this session on
   purpose). Once the user runs the one `gcloud secrets versions add` command from
   README.md and re-runs the workflow, this should be the last blocker.

---

## Getting a real Gemini key: three unexpected blockers

None of these were anticipated in the plan — all found by actually trying to get a
working key, not by reading docs in advance.

1. **User's machine was reset** mid-project. Checked survivability: `gcloud`/`gh`
   auth tokens, Homebrew-installed CLIs, and the project's `.venv`/`.env` all
   persisted (they live outside whatever got reset), so nothing needed redoing.
2. **API keys disallowed on an auto-provisioned project.** The user first tried
   creating a key via `console.cloud.google.com/agent-platform/studio/settings/api-keys`,
   which was scoped to an auto-generated project Google created behind the scenes
   (`project-939b296c-...`, not `rag-qa-portfolio-02173`), and that project's default
   policy blocked raw API key creation, recommending ADC instead. Diagnosed by
   comparing the project ID in the URL against the one we actually provisioned — they
   didn't match. Fix: go to the classic `aistudio.google.com/app/apikey` URL instead
   and explicitly pick `rag-qa-portfolio-02173` as the backing project when creating
   the key, which worked. (Briefly considered rearchitecting the whole app onto
   Vertex AI + ADC instead, which would also have solved it and arguably fits the
   existing service-account setup even better — but the simpler fix worked, so that
   rework wasn't needed.)
3. **`models/text-embedding-004` no longer exists.** First live call to the real
   Gemini API failed with `404 ... is not found for API version v1beta`. Rather than
   guess a replacement name, queried the live model list directly:
   `curl "https://generativelanguage.googleapis.com/v1beta/models?key=$KEY"`. Current
   embedding models are `gemini-embedding-001` (stable) and `gemini-embedding-2`
   (newer) — `text-embedding-004` has been fully retired since this project's design
   was written. Switched `EMBEDDING_MODEL` (config default, `.env`, `.env.example`,
   README) to `models/gemini-embedding-001`. `gemini-2.5-flash` (the chat model) was
   still live in the same model list, so it didn't need changing.
4. **Gemini API prepay credits depleted** on the very first real embedding call —
   `429 ResourceExhausted`. The user, reasonably, didn't want to add money without
   understanding the actual cost exposure. Set up a **Cloud Billing budget** (separate
   from the Gemini prepay balance) as a safety net for the postpaid GCP/Cloud Run side:
   ```bash
   gcloud services enable billingbudgets.googleapis.com --project=rag-qa-portfolio-02173
   gcloud billing budgets create --billing-account=011464-2696C2-5D2EA4 \
     --display-name="RAG portfolio project - cost guard" --budget-amount=10USD \
     --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 \
     --filter-projects=projects/rag-qa-portfolio-02173
   ```
   Also explained the actual cost model to the user: the Gemini side is *prepaid*, so
   there's no overspend risk there by construction (you only ever lose what you
   deposit); Cloud Run's free tier should keep the GCP side near $0 at this traffic
   level regardless. Left the decision of how much to prepay to the user.
