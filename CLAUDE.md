# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python package (`sentiment_bias`) that estimates latent sentiment bias in media coverage. Given mention counts broken down by (outlet, subject, sentiment polarity), it fits a latent-trait model relating outlet bias to subject sentiment by maximum penalised likelihood.

It is **not** published to PyPI — collaborators clone the repo and `pip install -e .` into a virtualenv. The package is the primary interface; an optional FastAPI layer (`sentiment_bias.api`, behind the `api` extra) wraps it in HTTP and serves a static-HTML/Chart.js frontend.

## Commands

```bash
# Install (editable; -e means src/ edits take effect with no reinstall)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,dev]"       # core + FastAPI + pytest; plain `pip install -e .` for core only

# Run the server
sentiment-bias-api                             # console script; SENTIMENT_BIAS_HOST / _PORT
bash run.sh                                    # backgrounded, writes server.pid / server.log
kill $(cat server.pid)                         # stop it
uvicorn sentiment_bias.api.app:app --reload    # development

# Tests
pytest
pytest -m "not slow"                           # skip the end-to-end recovery test
pytest tests/test_model.py::TestLogLikelihood -v
```

There is no lint/format/typecheck tooling configured (no ruff/black/mypy config present).

The frontend (`src/sentiment_bias/api/static/`) is plain HTML/JS with no build step — edits take effect on page reload.

## Architecture

The split is deliberate: the core is pure numpy/pandas so it can be used from a notebook without installing FastAPI, and `api/` is the only place Pydantic and FastAPI appear.

### The model (`src/sentiment_bias/model.py`)

The log-odds of a positive vs. negative mention for outlet *i* covering subject *j*, with *D* latent dimensions:

```
q_ij = dot(z[i], a[j]) + b[j]
```

- `z[i]` — outlet *i*'s latent bias vector, length D
- `a[j]` — subject *j*'s discrimination vector, length D (couples with `z` dimension-wise)
- `b[j]` — subject *j*'s scalar baseline sentiment, shared across all dimensions

Mention counts per (outlet, subject) follow a 3-category multinomial (negative/neutral/positive) parameterised by `q_ij`. `D=1` recovers the original scalar model; `n_dimensions` is validated to `[1, 2]` in `AnalysisInput` (the HTTP boundary only — `fit` itself accepts any D ≥ 1).

This module is pure NumPy, with no pandas, Pydantic or I/O: it is the only file that must be read to verify the mathematics. It holds `log_likelihood` / `negative_log_likelihood` / `grad_negative_log_likelihood` (objective and analytical gradient over the flat parameter vector `x`, laid out as `[z (m*D), a (k*D), b (k)]`), `pack`/`unpack`/`log_odds` (parameter algebra), `bic`, and `canonicalize_1d`.

`bic` is an approximation: it treats the L2 penalty's effect on the log-posterior as negligible. Lower is better, as for raw loss.

### The rest of the core

- `data.py` — `MentionTensor` (a frozen dataclass wrapping an `(m, k, 3)` int array plus sorted outlet/subject name lists; **the sorting determines index order everywhere downstream**), the `from_wide` / `from_long` / `from_records` / `from_array` constructors, the `to_tensor` dispatcher, and `read_csv` / `read_json`. This is the only module that knows about DataFrame column names. Duplicate `(outlet, subject, mention_type)` rows are summed; missing pairs become zeros.
- `results.py` — `BiasFit`, the frozen dataclass returned by `fit`. `z` and `a` are DataFrames indexed by name with one column per dimension (`dim_1`, …); `b` is a Series, not a frame, since there is genuinely one scalar per subject. Also carries `losses` (one per restart, which reveals a multimodal objective), `converged`, `data` (the fitted tensor) and `x`.
- `estimate.py` — `fit`, which orchestrates `to_tensor` → `minimize` (L-BFGS-B with analytical gradient, box constraints `[-5, 5]`) over `n_restarts` random initialisations → `canonicalize_1d` when `n_dimensions == 1` → `bic` → `BiasFit`. Takes `seed` so restarts are reproducible.

The `D=1` canonicalization convention is **unit norm**: `sum(z) == 0` and `sum(z**2) == 1`, which means `var(z) == 1/m` (not 1 — with zero mean the two normalisations are mutually exclusive, differing by `sqrt(m)`). It runs *after* the optimisation, so it is purely presentational: the box constraints bound the raw parameter vector, and the reported `a` — `sqrt(m)` times larger than under unit variance — is not bounded by them. `D > 1` output is not canonicalized at all: no centring, scaling, sign or rotation convention.
- `simulate.py` — the generative direction, as first-class as `fit`: `simulate(z, a, b, ...)` samples counts from known parameters, and `thin(tensor, fraction)` subsamples a tensor to a fraction of its mentions via a multivariate hypergeometric. `n_mentions` accepts a fixed `int`, `"poisson"` (with `mean_mentions`), or an explicit `(m, k)` array.

### API (`src/sentiment_bias/api/`)

- `app.py` — two POST routes, each three lines: `adapters` → `fit`/`simulate` → `adapters`. `POST /analyze` takes `AnalysisInput` → `AnalysisOutput`; `POST /generate` takes `GenerateInput` → `AnalysisInput` of synthetic mentions (useful for round-trip testing). `GET /` serves the frontend. Static paths resolve against `STATIC_DIR = Path(__file__).resolve().parent / "static"`, so the server works from any cwd.
- `schemas.py` — the Pydantic models. Note `z` and `a` are always `List[float]` (length D) even when D=1, so clients must index into these vectors rather than treat them as scalars.
- `adapters.py` — every conversion between the wire format and the core's numpy/pandas types. Keeping these here is what lets the core stay Pydantic-free.

### Notebooks (`notebooks/`)

`Usar_service_directo.ipynb` (in-process use) and `Ejemplo_llamar_API_modelo.ipynb` (HTTP client) are worked examples. The other five (`DataFitting*`, `SyntheticDataEstimation*`, `Model_consistency`) predate the package and re-implement the model inline; they are kept as the record of the analyses behind the paper. Careful if migrating them: `SyntheticDataEstimation` applies the L2 penalty *inside* the double loop, so its penalty is multiplied by the number of rows/columns — it is not the same model as the package.

## Testing conventions

- Test files mirror the source layout: `test_model.py`, `test_data.py`, `test_results.py`, `test_estimate.py`, `test_simulate.py`, and `tests/api/` for the HTTP layer.
- Tests that exercise `fit` mock `sentiment_bias.estimate.minimize` rather than actually optimising, so the suite stays fast and deterministic. For this to keep working, `estimate.py` must keep importing `minimize` at module level, and `api/app.py` must keep importing `fit`/`simulate` at module level and calling them by bare name.
- The one end-to-end test that really runs the optimiser is marked `slow`.
- pytest runs with `--import-mode=importlib` (set in `pyproject.toml`), so there is no `tests/__init__.py` and no root `conftest.py` — the package resolves through the editable install, from any cwd. If imports only work from the repo root, the install is broken.
