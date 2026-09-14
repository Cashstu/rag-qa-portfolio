#!/bin/sh
set -e

# Ingestion is intentionally NOT run here. app.main lazily builds the vector store
# (app/ingest.py's load_or_build_vector_store) on the first POST /ask, not at
# container startup — see docs/BUILD_LOG.md. Running it here instead would make
# GET /health, and therefore Cloud Run's readiness check, depend on a valid
# GOOGLE_API_KEY and a successful Gemini call: any ingestion failure would then take
# the whole container down instead of just failing individual /ask requests.
exec gunicorn --bind "0.0.0.0:${PORT:-8080}" --workers 1 --threads 4 --timeout 120 app.main:app
