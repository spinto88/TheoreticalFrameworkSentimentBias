# Sentiment Bias

A Python package for estimating latent sentiment bias in media coverage.  Given a dataset of mention counts broken down by outlet, subject, and sentiment polarity, the model infers a set of parameters that explain how each outlet's editorial slant interacts with each subject's intrinsic reception.

It can be used in two independent ways, and you only need to set up the one you want:

- **As a library** — `import sentiment_bias as sb` from any notebook or script. Nothing runs in the background, no localhost, no web dependencies. This is all most people need; see [As a library](#as-a-library-no-localhost-no-server).
- **As an HTTP service** — an optional FastAPI server with a browser frontend, for running the model without writing code, and for sharing it with people who don't use Python. See [Run the API](#run-the-api) and [Deploy on a server](#deploy-on-a-server).

The HTTP layer is built on top of the library, so the model behaves identically either way.

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

With `D=1` the model is only identified up to an affine reparametrisation of `z`, so the fitted parameters are canonicalized to

```
sum(z) = 0        sum(z²) = 1
```

with the rescaling absorbed into `a` and the shift into `b`, leaving every `q_ij` unchanged.  The remaining sign ambiguity is pinned by `sign_reference_outlet`.

Note this is unit **norm**, not unit variance.  With zero mean the two are mutually exclusive — `sum(z²) = m · var(z)` — so under this convention `var(z) = 1/m`, and `a` is `√m` times larger than it would be under unit variance.  Canonicalization runs after the optimisation, so it is purely a change of representation: the `[-5, 5]` box constrains the raw parameter vector, not the reported `a`.

With `D > 1` the output is **not** canonicalized: it comes straight from the optimiser, with no centring, scaling, sign or rotation convention applied.

---

## Installation

The package is **not** on PyPI.  You install it from a clone of this repository, into a virtual environment.  Requires Python 3.10 or newer.

Pick the path that matches what you want to do:

| You want to… | Install | Then |
|---|---|---|
| Run the model from your own notebooks or scripts, no server involved | `pip install -e .` | [Use as a library](#use-as-a-library) |
| Serve the web interface to other people | `pip install -e ".[api]"` | [Deploy on a server](#deploy-on-a-server) |
| Work on this codebase | `pip install -e ".[api,dev,nb]"` | [Tests](#tests) |

### As a library (no localhost, no server)

This is all you need if you just want to fit the model in Python.  It installs **only** numpy, scipy and pandas — no FastAPI, no web stack:

```bash
git clone <repository-url>
cd TheoreticalFrameworkSentimentBias

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

That's it.  Verify it worked from a directory *outside* the repository — if this prints a version, the install is good and nothing depends on your working directory:

```bash
cd ~ && python -c "import sentiment_bias as sb; print(sb.__version__)"
```

Now use it from any notebook or script, with no `sys.path` juggling and nothing running in the background:

```python
import sentiment_bias as sb

fit = sb.fit(sb.read_csv("mentions.csv"))
```

If your notebook runs on a different kernel than this virtual environment, register it once:

```bash
pip install ipykernel
python -m ipykernel install --user --name sentiment-bias
```

### Extras

```bash
pip install -e ".[api]"          # + FastAPI server and frontend
pip install -e ".[dev]"          # + pytest, httpx
pip install -e ".[nb]"           # + jupyter, matplotlib, requests
pip install -e ".[api,dev,nb]"   # everything
```

### What `-e` means

`-e` installs in *editable* mode: the virtual environment points at `src/` in your clone rather than copying it, so your edits take effect on the next Python start with no reinstall.  Frontend edits need nothing but a page reload.

Reinstall (`pip install -e .` again) only when you change `pyproject.toml` — for example after adding a dependency, or to pick up a new console script.

### Upgrading an existing clone

```bash
git pull
source .venv/bin/activate
pip install -e ".[api]"    # only needed if pyproject.toml changed
```

If you have an older clone from before the package restructure, its virtual environment (often `env/`) predates the new layout and will not work.  Create a fresh one as above; you can delete the old directory.

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

You only need this if you want the web interface.  Requires the `api` extra, and the virtual environment must be **activated** — `run.sh` calls the `sentiment-bias-api` console script, which lives inside it.

```bash
source .venv/bin/activate

sentiment-bias-api                             # foreground
bash run.sh                                    # background, writes server.pid and server.log
uvicorn sentiment_bias.api.app:app --reload    # development, reloads on source edits
```

The application is then at `http://localhost:8000`, with interactive API docs at `/docs`.  Stop the backgrounded server with `kill $(cat server.pid)`.

Host and port come from the environment, defaulting to `0.0.0.0:8000`:

```bash
SENTIMENT_BIAS_HOST=127.0.0.1 SENTIMENT_BIAS_PORT=9000 sentiment-bias-api
```

Note the default binds to **all** interfaces, so the server is reachable from other machines as soon as the port is open.  Use `127.0.0.1` to keep it local.

---

## Deploy on a server

`run.sh` is fine for a quick session, but it dies with your shell and won't come back after a reboot.  For something you leave running, use a systemd unit.

Install the package as a dedicated user, then create `/etc/systemd/system/sentiment-bias.service`:

```ini
[Unit]
Description=Sentiment Bias Analyzer
After=network.target

[Service]
Type=simple
User=sentiment-bias
WorkingDirectory=/srv/sentiment-bias
Environment=SENTIMENT_BIAS_HOST=127.0.0.1
Environment=SENTIMENT_BIAS_PORT=8000
ExecStart=/srv/sentiment-bias/.venv/bin/sentiment-bias-api
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sentiment-bias
systemctl status sentiment-bias
journalctl -u sentiment-bias -f        # logs
```

Pointing `ExecStart` straight at the console script inside the virtual environment means there is nothing to activate — no shell involved.

### Behind a reverse proxy

Bind the app to `127.0.0.1` (as above) and let nginx terminate TLS and face the network:

```nginx
server {
    listen 443 ssl;
    server_name sentiment-bias.example.edu;

    # ssl_certificate / ssl_certificate_key ...

    client_max_body_size 32m;          # mention files can be large

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;       # a fit with many restarts is slow
    }
}
```

Two things worth knowing before you expose this:

- **Requests are CPU-bound.** The routes are synchronous, so FastAPI runs them in a thread pool rather than blocking the event loop — but the optimiser is pure Python, so concurrent fits contend for the GIL and each one slows the others down. A fit with a large dataset and many restarts can take minutes, hence the generous proxy timeout. If several people will use it at once, run separate worker *processes* instead of the console script: `ExecStart=/srv/sentiment-bias/.venv/bin/uvicorn sentiment_bias.api.app:app --host 127.0.0.1 --workers 4`.
- **There is no authentication, rate limiting or request-size cap in the app.** Anyone who can reach the port can post an arbitrarily large job. On a public address, put access control in the proxy (HTTP basic auth, an IP allowlist, or a VPN) or keep it on an internal network.

### Updating a deployment

```bash
cd /srv/sentiment-bias
sudo -u sentiment-bias git pull
sudo -u sentiment-bias .venv/bin/pip install -e ".[api]"   # only if pyproject.toml changed
sudo systemctl restart sentiment-bias
```

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

- **`Usar_paquete.ipynb`** — start here. Uses the package directly, in-process: loading data, fitting, generating synthetic data, verifying parameter recovery, and the `thin()` consistency curve.
- `Ejemplo_llamar_API_modelo.ipynb` — calling the HTTP API from Python.
- `DataFitting*.ipynb`, `SyntheticDataEstimation*.ipynb`, `Model_consistency.ipynb` — self-contained research notebooks that predate the package and re-implement the model inline.  They are kept as the record of the analyses behind the paper; new work should use the package instead.
