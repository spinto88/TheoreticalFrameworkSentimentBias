# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI service that estimates latent sentiment bias in media coverage. Given mention counts broken down by (outlet, subject, sentiment polarity), it fits a latent-trait model relating outlet bias to subject sentiment via maximum penalised likelihood, and serves a static-HTML/Chart.js frontend for uploading data and visualising the fitted parameters.

## Commands

```bash
# Install
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Run the server (foreground)
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Run the server (background, via PID/log file — see run.sh)
bash run.sh
kill $(cat server.pid)   # stop it

# Run all tests
pytest tests/ -v

# Run a single test file / test
pytest tests/test_service.py -v
pytest tests/test_service.py::TestRunAnalysis::test_output_outlet_count -v

# Docker
docker-compose up --build
```

There is no lint/format/typecheck tooling configured in this repo (no ruff/black/mypy config present).

The frontend (`src/static/`) is plain HTML/JS with no build step — edits take effect on page reload, served directly by FastAPI's `StaticFiles` mount at `/static` and `index.html` at `/`.

## Architecture

### The model (`src/service.py`)

The log-odds of a positive vs. negative mention for outlet *i* covering subject *j*, with *D* latent dimensions:

```
q_ij = dot(z[i], a[j]) + b[j]
```

- `z[i]` — outlet *i*'s latent bias vector, length D
- `a[j]` — subject *j*'s discrimination vector, length D (couples with `z` dimension-wise)
- `b[j]` — subject *j*'s scalar baseline sentiment, shared across all dimensions

Setting `D=1` recovers the original scalar bias model; `D=2` is also supported (`n_dimensions` is validated to `[1, 2]` in `AnalysisInput`). Mention counts per (outlet, subject) follow a 3-category multinomial (negative/neutral/positive) parameterised by `q_ij`.

Parameters are estimated by maximising a Gaussian-(L2)-penalised log-likelihood with **SciPy's `minimize` using L-BFGS-B and an analytical gradient** (`grad_negative_log_likelihood`) — not `differential_evolution` (README describes an earlier version of the model). All parameters are box-constrained to `[-5, 5]`.

Key functions, all in `src/service.py`:
- `build_tensor` — flat `Mention` list → dense `(m, k, 3)` count tensor + sorted outlet/subject name lists (sorting determines index order used everywhere downstream)
- `log_likelihood` / `negative_log_likelihood` / `grad_negative_log_likelihood` — objective and gradient over a single flat parameter vector `x`, laid out as `[z (m*D), a (k*D), b (k)]`
- `aproximate_bayesian_information_criteria` — BIC computed from the fitted negative log-likelihood, used as a model-selection metric alongside raw loss (lower is better for both); it's an approximation since it treats the L2 penalty's effect on the log-posterior as negligible
- `run_analysis` — orchestrates build_tensor → minimize → BIC → `build_output`; this is what both API endpoints call into
- `generate_data` / `generate_mentions` — the inverse direction: given known `z`/`a`/`b`, sample synthetic mention counts from the generative model (used for validation/testing and by the "generate synthetic data" notebooks)

### API (`src/app.py`)

Two POST endpoints, thin wrappers around `service.py`:
- `POST /analyze` — `AnalysisInput` (mentions + `n_dimensions`) → `AnalysisOutput` (`z` per outlet, `a`+`b` per subject, `loss`, `bic`)
- `POST /generate` — `GenerateInput` (known `z`/`a`/`b` + `amount_of_mentions`) → `AnalysisInput` of synthetic mentions, sampled from the generative model — useful for round-trip testing (generate synthetic data with known ground-truth parameters, then run `/analyze` and check recovery)

`GET /` serves `src/static/index.html`; `/static/*` serves the rest of the frontend.

### Schemas (`src/schemas.py`)

Pydantic models shared by both endpoints. Note `z` and `a` are always `List[float]` (length D) even when D=1 — the frontend and any client code must index into these vectors rather than treat them as scalars.

### Frontend (`src/static/`)

No build step; self-contained SPA:
- `main.js` — upload/drag-drop, client-side CSV→JSON conversion (backend only ever receives JSON), API calls
- `charts.js` — Chart.js bar charts, one per parameter family (z, a, b)
- `utils.js` — CSV parsing, PNG/SVG chart export

### Notebooks (`notebooks/`)

Exploratory/example usage (data fitting via gradient descent, synthetic data estimation, calling the API directly, model-consistency checks). Not part of the runtime path; useful as worked examples of calling into `src/service.py` or the HTTP API directly.

## Testing conventions

- `conftest.py` puts the repo root on `sys.path` so `src` is importable regardless of invocation directory — always run pytest from the repo root or rely on this.
- Tests that exercise `run_analysis` mock `src.service.minimize` (`unittest.mock.patch`) rather than actually optimising, so the suite stays fast and deterministic (see `tests/test_service.py::TestRunAnalysis`).
- Test files mirror `src/` 1:1: `test_service.py` (model/math), `test_schemas.py` (Pydantic validation), `test_app.py` (HTTP endpoints via `TestClient`).
