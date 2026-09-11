# Sentiment Bias

A Python package for estimating latent sentiment bias in media coverage.  Given a dataset of mention counts broken down by outlet, subject, and sentiment polarity, the model infers a set of parameters that explain how each outlet's editorial slant interacts with each subject's intrinsic reception.

It can be used in two ways:

- **As a library** — `import sentiment_bias as sb` from any notebook or script.  This is all most people need.
- **As an HTTP service** — an optional FastAPI server with a browser frontend, for running the model without writing code.

---

## The Model

The log-odds of a positive vs. negative mention for outlet *i* covering subject *j* is

```
q_ij = z[i] · a[j] + b[j]
```

and the mention counts for that pair follow a three-category multinomial:

```
P(negative) = exp(-q_ij) / Z      P(neutral) = 1 / Z      P(positive) = exp(q_ij) / Z
Z = exp(-q_ij) + 1 + exp(q_ij)
```

### Parameters

| Parameter | Shape | Interpretation |
|-----------|-------|----------------|
| `z[i]` | vector of length *D*, one per outlet | Latent bias for outlet *i*. Has no standalone direction — its sign only becomes meaningful in combination with `a[j]`. |
| `a[j]` | vector of length *D*, one per subject | Discrimination parameter for subject *j*. Controls how strongly outlet bias influences sentiment toward this subject. If `z[i] · a[j] > 0` the outlet reinforces positive sentiment; if `< 0` it reinforces negative sentiment. Larger `\|a[j]\|` amplifies the effect. |
| `b[j]` | scalar, one per subject | Baseline sentiment for subject *j*, independent of which outlet covers it, and shared across all dimensions. |

*D* is the number of latent dimensions (`n_dimensions`); each dimension couples `z[i,d]` only with `a[j,d]`.  `D=1` recovers the original scalar model.

### Estimation

Parameters are estimated by maximising a **penalised log-likelihood** with Gaussian (L2) regularisation on all parameters, using SciPy's **L-BFGS-B** with an analytical gradient and box constraints `[-5, 5]`.

The objective is non-convex, so the solver can settle into different local minima depending on its initialisation.  Use `n_restarts` to run several independent restarts and keep the best; `fit.losses` reports every restart's final loss, which tells you whether they agreed.

With `D=1` the model is only identified up to an affine reparametrisation of `z`, so the fitted parameters are canonicalized: `z` is standardised to zero mean and unit variance, with the rescaling absorbed into `a` and the shift into `b`.  The remaining sign ambiguity is pinned by `sign_reference_outlet`.

---

## Installation

The package is **not** on PyPI.  Clone the repository and install it into a virtual environment:

```bash
git clone <repository-url>
cd TheoreticalFrameworkSentimentBias

python3 -m venv .venv
source .venv/bin/activate

pip install -e .              # the model only (numpy, scipy, pandas)
```

`-e` installs in *editable* mode: edits under `src/` take effect immediately, with no reinstall.  Once installed, `import sentiment_bias` works from any directory.

Optional extras:

```bash
pip install -e ".[api]"       # + FastAPI server and frontend
pip install -e ".[dev]"       # + pytest, httpx
pip install -e ".[nb]"        # + jupyter, matplotlib, requests
pip install -e ".[api,dev,nb]"   # everything
```

Requires Python 3.10 or newer.

---

## Use as a library

### Fitting your own data

```python
import sentiment_bias as sb

tensor = sb.read_csv("data/input.csv")      # or read_json(...), or a DataFrame
fit = sb.fit(tensor, n_dimensions=1, n_restarts=10, seed=0)

fit.z            # outlet bias, DataFrame indexed by outlet, one column per dimension
fit.subjects     # a and b, DataFrame indexed by subject
fit.loss, fit.bic
print(fit.summary())
```

`fit` accepts anything `to_tensor` understands: a `MentionTensor`, a wide or long DataFrame, a raw `(m, k, 3)` array, or a list of mention records.

### Generating synthetic data

The generative direction is a first-class part of the package — simulate with known parameters, fit, and check how well they are recovered:

```python
import numpy as np, sentiment_bias as sb
from scipy.stats import spearmanr

z = np.linspace(-2, 2, 50)        # 50 outlets
a = np.linspace(-1, 1, 6)         # 6 subjects
b = np.random.default_rng(0).normal(0, 0.05, 6)

tensor = sb.simulate(z, a, b, n_mentions="poisson", mean_mentions=100, seed=123457)
fit = sb.fit(tensor, n_dimensions=1, n_restarts=5, seed=1)

abs(spearmanr(z, fit.z["dim_1"]).statistic)    # ≈ 0.99
```

`n_mentions` takes three forms:

| Value | Meaning |
|---|---|
| `int` (default `100`) | the same number of mentions for every pair |
| `"poisson"` with `mean_mentions=μ` | `N_ij ~ Poisson(μ)`, drawn independently per pair |
| array of shape `(m, k)` | an explicit count for each pair |

`thin(tensor, fraction)` subsamples a tensor down to a fraction of its total mentions, drawing without replacement — useful for measuring how the fit degrades as data thins out.

### Public API

| Name | Purpose |
|---|---|
| `fit` | Estimate parameters from mention counts |
| `simulate`, `thin` | Draw synthetic counts; subsample counts |
| `BiasFit` | Fitted parameters and diagnostics |
| `MentionTensor` | The `(m, k, 3)` count container |
| `to_tensor`, `from_wide`, `from_long`, `from_records`, `from_array` | Build a tensor from any input shape |
| `read_csv`, `read_json` | Load counts from disk |
| `model` | The raw likelihood, gradient, BIC and canonicalization functions |

---

## Data formats

A mention record says how many times a given outlet mentioned a given subject with a given sentiment polarity.

| Field | Type | Allowed values |
|-------|------|----------------|
| `outlet` | string | any non-empty string |
| `subject` | string | any non-empty string |
| `mention_type` | string | `positive`, `neutral`, `negative` |
| `amount_of_mentions` | integer | ≥ 0 |

Outlets and subjects are ordered alphabetically, and this ordering determines the index order everywhere downstream.  Rows repeating the same `(outlet, subject, mention_type)` triple are summed; absent combinations become zeros.

### Long layout

One row per sentiment.  This is what `data/input.csv` holds and what the HTTP API receives.

```
outlet,subject,mention_type,amount_of_mentions
Reuters,Economy,positive,12
Reuters,Economy,negative,3
FoxNews,Economy,negative,9
FoxNews,Politics,positive,7
```

The JSON equivalent wraps the same records in a `data` key:

```json
{
  "data": [
    { "outlet": "Reuters", "subject": "Economy", "mention_type": "positive", "amount_of_mentions": 12 },
    { "outlet": "Reuters", "subject": "Economy", "mention_type": "negative", "amount_of_mentions": 3 }
  ]
}
```

### Wide layout

One row per (outlet, subject) pair — the tensor with its last axis unrolled.  Accepted by `fit` and produced by `tensor.to_frame("wide")`.

```
outlet,subject,negative,neutral,positive
Reuters,Economy,3,0,12
FoxNews,Economy,9,0,0
```

---

## Run the API

Requires the `api` extra.

```bash
sentiment-bias-api                                   # honours SENTIMENT_BIAS_HOST / _PORT
bash run.sh                                          # same, backgrounded with a PID/log file
uvicorn sentiment_bias.api.app:app --reload          # development, with auto-reload
```

The application is available at `http://localhost:8000`.  `bash run.sh` writes `server.pid` and `server.log`; stop it with `kill $(cat server.pid)`.

### `POST /analyze`

Runs the model on mention data.  Request body (`AnalysisInput`):

```json
{
  "data": [ ... mention records ... ],
  "n_dimensions": 1,
  "n_restarts": 1,
  "sign_reference_outlet": null,
  "sign_reference_positive": true
}
```

Response (`AnalysisOutput`):

```json
{
  "outlets": [ { "outlet": "Reuters", "z": [1.52] } ],
  "subjects": [ { "subject": "Economy", "a": [0.94], "b": 0.31 } ],
  "loss": 26.27,
  "bic": 78.70,
  "n_restarts": 1
}
```

`sign_reference_outlet` is only valid when `n_dimensions` is 1, and must name an outlet present in the data.

### `POST /generate`

Samples synthetic mention data from known parameters, and returns it as an `AnalysisInput` ready to feed back into `/analyze`.  Request body (`GenerateInput`):

```json
{
  "outlets": [ { "outlet": "A", "z": [1.0] } ],
  "subjects": [ { "subject": "X", "a": [0.8], "b": 0.1 } ],
  "amount_of_mentions": 100,
  "poisson": false
}
```

With `"poisson": true`, `amount_of_mentions` is read as the Poisson mean rather than an exact per-pair count.

### `GET /`

Serves the frontend.

Interactive API documentation is at `http://localhost:8000/docs` while the server is running.

---

## Frontend

A self-contained single-page application with no build step, bundled with the package under `sentiment_bias/api/static/`.

- **Input**: drag & drop or click-to-upload accepting `.json` and `.csv`.  CSV is parsed client-side and converted to JSON before being sent, so the backend always receives JSON.
- **Charts**: three bar charts rendered with [Chart.js](https://www.chartjs.org/), one per parameter family (*z*, *a*, *b*).
- **Export**: each chart downloads as a high-resolution PNG or as a true vector SVG.

---

## Tests

```bash
pip install -e ".[api,dev]"
pytest
```

Tests that exercise `fit` mock the solver rather than optimising, so the suite stays fast and deterministic.  The end-to-end recovery test that does run the optimiser is marked `slow`; skip it with `pytest -m "not slow"`.

---

## Project structure

```
.
├── pyproject.toml
├── run.sh                        # start the API in the background
├── data/
│   ├── input.json                # sample input, JSON
│   └── input.csv                 # sample input, CSV (same data)
├── src/sentiment_bias/
│   ├── model.py                  # likelihood, gradient, BIC, canonicalization
│   ├── data.py                   # MentionTensor, conversions, loaders
│   ├── results.py                # BiasFit
│   ├── estimate.py               # fit()
│   ├── simulate.py               # simulate(), thin()
│   └── api/                      # optional HTTP layer
│       ├── app.py                # FastAPI routes
│       ├── schemas.py            # Pydantic request/response models
│       ├── adapters.py           # Pydantic ↔ core translation
│       ├── __main__.py           # sentiment-bias-api entry point
│       └── static/               # frontend (index.html, main.js, charts.js, utils.js)
├── tests/
└── notebooks/
```

---

## Notebooks

- `Usar_service_directo.ipynb` — using the package directly, in-process.
- `Ejemplo_llamar_API_modelo.ipynb` — calling the HTTP API from Python.
- `DataFitting*.ipynb`, `SyntheticDataEstimation*.ipynb`, `Model_consistency.ipynb` — self-contained research notebooks that predate the package and re-implement the model inline.  They are kept as the record of the analyses behind the paper; new work should use the package instead.
