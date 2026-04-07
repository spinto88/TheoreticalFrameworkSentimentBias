# Sentiment Bias Analyzer

A tool for estimating latent sentiment bias in media coverage.  Given a dataset of mention counts broken down by outlet, subject, and sentiment polarity, the model infers a set of parameters that explain how each outlet's editorial slant interacts with each subject's intrinsic reception.

---

## The Model

The core idea is that the log-odds of a positive vs. negative mention for outlet *i* covering subject *j* follows a linear relationship:

```
log-odds(i, j) = z[i] · a[j] + b[j]
```

### Parameters

| Parameter | Shape | Interpretation |
|-----------|-------|----------------|
| `z[i]` | one per outlet | Latent bias score for outlet *i*. Has no standalone direction — its sign only becomes meaningful in combination with `a[j]`. |
| `a[j]` | one per subject | Discrimination parameter for subject *j*. Controls how strongly outlet bias influences sentiment toward this subject. If `z[i] · a[j] > 0` the outlet reinforces positive sentiment; if `z[i] · a[j] < 0` it reinforces negative sentiment. Larger `|a[j]|` amplifies this effect. |
| `b[j]` | one per subject | Baseline sentiment for subject *j*, independent of which outlet covers it. A positive `b[j]` indicates a general tendency across all outlets to mention this subject positively; a negative value indicates the opposite. |

### Estimation

Parameters are estimated by maximising a **penalised log-likelihood** with Gaussian (L2) regularisation on all parameters.  The optimisation uses SciPy's `differential_evolution` solver with bounds `[-5, 5]` on every parameter.

---

## Project Structure

```
.
├── data/
│   ├── input.json          # Sample input in JSON format
│   └── input.csv           # Sample input in CSV format (same data)
├── src/
│   ├── app.py              # FastAPI application and route definitions
│   ├── schemas.py          # Pydantic request / response models
│   ├── service.py          # Model implementation (tensor, likelihood, solver)
│   └── static/
│       ├── index.html      # Single-page frontend
│       ├── main.js         # File upload, drag & drop, API call
│       ├── charts.js       # Chart rendering (Chart.js)
│       └── utils.js        # CSV parser, PNG/SVG export, chart factory
├── tests/
│   ├── test_service.py     # Unit tests for service layer (27 tests)
│   ├── test_schemas.py     # Pydantic validation tests (23 tests)
│   └── test_app.py         # HTTP endpoint tests via TestClient (15 tests)
├── conftest.py             # pytest path configuration
├── requirements.txt
└── run.sh                  # Convenience script to start the server
```

---

## Input Format

The model accepts data as a flat list of mention records.  Each record describes how many times a given outlet mentioned a given subject with a specific sentiment polarity.

### JSON

```json
{
  "data": [
    { "outlet": "Reuters",  "subject": "Economy", "mention_type": "positive", "amount_of_mentions": 12 },
    { "outlet": "Reuters",  "subject": "Economy", "mention_type": "negative", "amount_of_mentions": 3  },
    { "outlet": "FoxNews",  "subject": "Economy", "mention_type": "negative", "amount_of_mentions": 9  },
    { "outlet": "FoxNews",  "subject": "Politics","mention_type": "positive", "amount_of_mentions": 7  }
  ]
}
```

### CSV

The CSV format uses the same field names as column headers:

```
outlet,subject,mention_type,amount_of_mentions
Reuters,Economy,positive,12
Reuters,Economy,negative,3
FoxNews,Economy,negative,9
FoxNews,Politics,positive,7
```

**Field constraints**

| Field | Type | Allowed values |
|-------|------|----------------|
| `outlet` | string | any non-empty string |
| `subject` | string | any non-empty string |
| `mention_type` | string | `positive`, `neutral`, `negative` |
| `amount_of_mentions` | integer | ≥ 0 |

Multiple rows with the same `(outlet, subject, mention_type)` triple are summed when building the internal mention tensor.

---

## API

The server exposes two endpoints.

### `POST /analyze`

Runs the bias estimation model on the provided mention data.

**Request body** — `application/json`, schema `AnalysisInput`:

```json
{
  "data": [ <list of Mention objects> ]
}
```

**Response** — `200 OK`, schema `AnalysisOutput`:

```json
{
  "outlets": [
    { "outlet": "Reuters", "z": 1.52 },
    { "outlet": "FoxNews", "z": -0.87 }
  ],
  "subjects": [
    { "subject": "Economy", "a": 0.94, "b": 0.31 },
    { "subject": "Politics","a": -0.62,"b": -0.15 }
  ]
}
```

Interactive API documentation is available at `http://localhost:8000/docs` once the server is running.

### `GET /`

Serves the frontend application.

---

## Getting Started

### 1. Install dependencies

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Run the server

```bash
bash run.sh
# or directly:
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`.

### 3. Use the frontend

1. Open `http://localhost:8000` in a browser.
2. Upload `data/input.json` or `data/input.csv` (or drag and drop the file).
3. Click **Run Analysis**.
4. Explore the three parameter charts.  Each chart can be exported individually as PNG or SVG; use **All PNG** / **All SVG** to download all three at once.

### 4. Run tests

```bash
pytest tests/ -v
```

All 65 tests should pass in under a second.  Tests that involve the optimisation solver mock `differential_evolution` to remain fast and deterministic.

---

## Frontend

The browser interface is a self-contained single-page application with no build step.

- **Input**: drag & drop or click-to-upload zone accepting `.json` and `.csv` files.  CSV is parsed client-side and converted to the equivalent JSON payload before being sent to the API — the backend always receives JSON regardless of the original file format.
- **Charts**: three bar charts rendered with [Chart.js](https://www.chartjs.org/), one per parameter family (*z*, *a*, *b*), with a white background, labelled axes, and per-bar colour coding (positive vs. negative values).
- **Export**: each chart can be downloaded as a high-resolution PNG (2× canvas pixel size, white background) or as a true vector SVG reconstructed directly from the chart data model.
