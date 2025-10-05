# Murmuration — Exoplanet Classifier (Kepler/TESS)

AI-powered discovery of exoplanet candidates.
- **Frontend:** Next.js (App Router), Tailwind v4, Radix primitives, Recharts
- **Backend:** FastAPI (JSON endpoints), UVicorn

## Quickstart (local)

### 0) Requirements
- Node 20+ (use `.nvmrc`)
- Python 3.11+
- macOS/Linux/WSL2

### 1) Backend (API)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r api/requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8081 --reload
