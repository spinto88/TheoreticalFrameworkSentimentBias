"""
Integration tests for the FastAPI application.

Uses FastAPI's TestClient to exercise the HTTP layer without spinning up a
real server.  `fit` and `simulate` are patched where noted so tests stay fast
and deterministic — note that the mocks must return real BiasFit /
MentionTensor objects, because the adapters run on the way back out.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sentiment_bias.api.app import STATIC_DIR, app
from sentiment_bias.data import from_array
from sentiment_bias.results import BiasFit, dimension_columns

client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def make_fit(z, a, b, outlets, subjects, *, loss=12.34, bic=20.5, n_restarts=1) -> BiasFit:
    """Build a BiasFit the way fit() would, for use as a mock return value."""
    n_dimensions = z.shape[1]
    columns = dimension_columns(n_dimensions)
    counts = np.zeros((len(outlets), len(subjects), 3), dtype=np.int64)
    return BiasFit(
        z=pd.DataFrame(z, index=pd.Index(outlets, name="outlet"), columns=columns),
        a=pd.DataFrame(a, index=pd.Index(subjects, name="subject"), columns=columns),
        b=pd.Series(b, index=pd.Index(subjects, name="subject"), name="b"),
        loss=loss,
        bic=bic,
        n_dimensions=n_dimensions,
        n_restarts=n_restarts,
        converged=True,
        losses=[loss],
        data=from_array(counts, outlets=outlets, subjects=subjects),
        x=np.concatenate([z.ravel(), a.ravel(), b]),
    )


VALID_PAYLOAD = {
    "data": [
        {"outlet": "A", "subject": "X", "mention_type": "positive", "amount_of_mentions": 10},
        {"outlet": "A", "subject": "X", "mention_type": "negative", "amount_of_mentions": 3},
        {"outlet": "B", "subject": "X", "mention_type": "neutral", "amount_of_mentions": 6},
        {"outlet": "B", "subject": "Y", "mention_type": "positive", "amount_of_mentions": 5},
    ]
}

MOCK_FIT = make_fit(
    z=np.array([[1.0], [-0.5]]),
    a=np.array([[0.8], [-0.3]]),
    b=np.array([0.2, 0.1]),
    outlets=["A", "B"],
    subjects=["X", "Y"],
)

MOCK_TENSOR = from_array(
    np.array([[[8, 12, 30]]], dtype=np.int64), outlets=["A"], subjects=["X"]
)


# ---------------------------------------------------------------------------
# GET / and static assets
# ---------------------------------------------------------------------------

class TestIndexRoute:
    def test_returns_200(self):
        assert client.get("/").status_code == 200

    def test_returns_html(self):
        assert "text/html" in client.get("/").headers["content-type"]

    def test_static_dir_is_packaged(self):
        """The frontend must ship inside the package, not sit next to the cwd."""
        assert (STATIC_DIR / "index.html").is_file()

    def test_static_assets_are_served(self):
        for asset in ["main.js", "charts.js", "utils.js"]:
            assert client.get(f"/static/{asset}").status_code == 200, asset

    def test_static_dir_lives_under_the_package(self):
        assert STATIC_DIR.parent.name == "api"
        assert STATIC_DIR.parent.parent.name == "sentiment_bias"


# ---------------------------------------------------------------------------
# POST /analyze — happy path
# ---------------------------------------------------------------------------

class TestAnalyzeRoute:
    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_returns_200(self, _mock):
        assert client.post("/analyze", json=VALID_PAYLOAD).status_code == 200

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_response_has_outlets_and_subjects(self, _mock):
        body = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert "outlets" in body
        assert "subjects" in body

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_outlet_schema(self, _mock):
        outlets = client.post("/analyze", json=VALID_PAYLOAD).json()["outlets"]
        assert all("outlet" in o and "z" in o for o in outlets)

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_subject_schema(self, _mock):
        subjects = client.post("/analyze", json=VALID_PAYLOAD).json()["subjects"]
        assert all("subject" in s and "a" in s and "b" in s for s in subjects)

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_outlet_count_matches_mock(self, _mock):
        assert len(client.post("/analyze", json=VALID_PAYLOAD).json()["outlets"]) == 2

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_subject_count_matches_mock(self, _mock):
        assert len(client.post("/analyze", json=VALID_PAYLOAD).json()["subjects"]) == 2

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_z_values_are_lists_of_numbers(self, _mock):
        outlets = client.post("/analyze", json=VALID_PAYLOAD).json()["outlets"]
        assert all(isinstance(o["z"], list) for o in outlets)
        assert all(isinstance(v, (int, float)) for o in outlets for v in o["z"])

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_a_b_values_correct_types(self, _mock):
        subjects = client.post("/analyze", json=VALID_PAYLOAD).json()["subjects"]
        assert all(isinstance(s["a"], list) for s in subjects)
        assert all(isinstance(v, (int, float)) for s in subjects for v in s["a"])
        assert all(isinstance(s["b"], (int, float)) for s in subjects)

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_loss_and_bic_are_returned(self, _mock):
        body = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert body["loss"] == pytest.approx(12.34)
        assert body["bic"] == pytest.approx(20.5)

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_parameter_values_come_from_the_fit(self, _mock):
        body = client.post("/analyze", json=VALID_PAYLOAD).json()
        by_outlet = {o["outlet"]: o["z"] for o in body["outlets"]}
        assert by_outlet["A"] == pytest.approx([1.0])
        assert by_outlet["B"] == pytest.approx([-0.5])

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_tensor_is_built_before_fitting(self, mock_fn):
        """The route must hand fit() a tensor, not the raw Pydantic list."""
        client.post("/analyze", json=VALID_PAYLOAD)
        tensor = mock_fn.call_args[0][0]
        assert tensor.shape == (2, 2, 3)
        assert tensor.outlets == ["A", "B"]
        assert tensor.n_mentions == 24

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_default_n_dimensions_forwarded(self, mock_fn):
        client.post("/analyze", json=VALID_PAYLOAD)
        assert mock_fn.call_args[1]["n_dimensions"] == 1

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_custom_n_dimensions_forwarded(self, mock_fn):
        client.post("/analyze", json={**VALID_PAYLOAD, "n_dimensions": 2})
        assert mock_fn.call_args[1]["n_dimensions"] == 2

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_default_n_restarts_forwarded(self, mock_fn):
        client.post("/analyze", json=VALID_PAYLOAD)
        assert mock_fn.call_args[1]["n_restarts"] == 1

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_custom_n_restarts_forwarded(self, mock_fn):
        client.post("/analyze", json={**VALID_PAYLOAD, "n_restarts": 10})
        assert mock_fn.call_args[1]["n_restarts"] == 10

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_default_sign_reference_forwarded(self, mock_fn):
        client.post("/analyze", json=VALID_PAYLOAD)
        kwargs = mock_fn.call_args[1]
        assert kwargs["sign_reference_outlet"] is None
        assert kwargs["sign_reference_positive"] is True

    @patch("sentiment_bias.api.app.fit", return_value=MOCK_FIT)
    def test_custom_sign_reference_forwarded(self, mock_fn):
        payload = {**VALID_PAYLOAD, "sign_reference_outlet": "A", "sign_reference_positive": False}
        client.post("/analyze", json=payload)
        kwargs = mock_fn.call_args[1]
        assert kwargs["sign_reference_outlet"] == "A"
        assert kwargs["sign_reference_positive"] is False


# ---------------------------------------------------------------------------
# POST /analyze — validation errors
# ---------------------------------------------------------------------------

class TestAnalyzeValidation:
    def test_missing_data_field_returns_422(self):
        assert client.post("/analyze", json={}).status_code == 422

    def test_invalid_mention_type_returns_422(self):
        payload = {"data": [{"outlet": "A", "subject": "X",
                             "mention_type": "UNKNOWN", "amount_of_mentions": 1}]}
        assert client.post("/analyze", json=payload).status_code == 422

    def test_negative_amount_returns_422(self):
        payload = {"data": [{"outlet": "A", "subject": "X",
                             "mention_type": "positive", "amount_of_mentions": -1}]}
        assert client.post("/analyze", json=payload).status_code == 422

    def test_missing_outlet_field_returns_422(self):
        payload = {"data": [{"subject": "X", "mention_type": "positive",
                             "amount_of_mentions": 1}]}
        assert client.post("/analyze", json=payload).status_code == 422

    def test_n_restarts_zero_returns_422(self):
        assert client.post("/analyze", json={**VALID_PAYLOAD, "n_restarts": 0}).status_code == 422

    def test_n_dimensions_three_returns_422(self):
        assert client.post("/analyze", json={**VALID_PAYLOAD, "n_dimensions": 3}).status_code == 422

    def test_unknown_sign_reference_outlet_returns_422(self):
        payload = {**VALID_PAYLOAD, "sign_reference_outlet": "NotAnOutlet"}
        assert client.post("/analyze", json=payload).status_code == 422

    def test_wrong_content_type_returns_422(self):
        response = client.post("/analyze", content="not json",
                               headers={"Content-Type": "text/plain"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /generate — shared fixtures
# ---------------------------------------------------------------------------

GENERATE_PAYLOAD = {
    "outlets": [
        {"outlet": "A", "z": [1.0]},
        {"outlet": "B", "z": [-1.0]},
    ],
    "subjects": [
        {"subject": "X", "a": [0.8], "b": 0.2},
        {"subject": "Y", "a": [-0.5], "b": -0.1},
    ],
    "amount_of_mentions": 50,
}


# ---------------------------------------------------------------------------
# POST /generate — happy path
# ---------------------------------------------------------------------------

class TestGenerateRoute:
    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_returns_200(self, _mock):
        assert client.post("/generate", json=GENERATE_PAYLOAD).status_code == 200

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_response_has_data_and_n_dimensions(self, _mock):
        body = client.post("/generate", json=GENERATE_PAYLOAD).json()
        assert "data" in body
        assert "n_dimensions" in body

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_n_dimensions_reflects_input_vectors(self, _mock):
        body = client.post("/generate", json=GENERATE_PAYLOAD).json()
        assert body["n_dimensions"] == 1

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_data_entries_have_correct_fields(self, _mock):
        entries = client.post("/generate", json=GENERATE_PAYLOAD).json()["data"]
        for entry in entries:
            assert set(entry) == {"outlet", "subject", "mention_type", "amount_of_mentions"}

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_data_entry_count_is_pairs_times_three(self, _mock):
        body = client.post("/generate", json=GENERATE_PAYLOAD).json()
        assert len(body["data"]) == 1 * 1 * 3   # MOCK_TENSOR is 1 outlet x 1 subject

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_counts_come_from_the_tensor(self, _mock):
        entries = client.post("/generate", json=GENERATE_PAYLOAD).json()["data"]
        lookup = {e["mention_type"]: e["amount_of_mentions"] for e in entries}
        assert lookup == {"negative": 8, "neutral": 12, "positive": 30}

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_parameters_forwarded_as_arrays(self, mock_fn):
        client.post("/generate", json=GENERATE_PAYLOAD)
        z, a, b = mock_fn.call_args[0]
        assert z.shape == (2, 1)
        assert a.shape == (2, 1)
        assert b.shape == (2,)
        np.testing.assert_allclose(z, [[1.0], [-1.0]])

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_names_forwarded(self, mock_fn):
        client.post("/generate", json=GENERATE_PAYLOAD)
        kwargs = mock_fn.call_args[1]
        assert kwargs["outlets"] == ["A", "B"]
        assert kwargs["subjects"] == ["X", "Y"]

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_fixed_n_is_the_default(self, mock_fn):
        client.post("/generate", json=GENERATE_PAYLOAD)
        kwargs = mock_fn.call_args[1]
        assert kwargs["n_mentions"] == 50
        assert kwargs["mean_mentions"] is None

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_poisson_flag_switches_mode(self, mock_fn):
        client.post("/generate", json={**GENERATE_PAYLOAD, "poisson": True})
        kwargs = mock_fn.call_args[1]
        assert kwargs["n_mentions"] == "poisson"
        assert kwargs["mean_mentions"] == 50

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_poisson_returns_200(self, _mock):
        payload = {**GENERATE_PAYLOAD, "poisson": True}
        assert client.post("/generate", json=payload).status_code == 200

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_default_amount_of_mentions_accepted(self, mock_fn):
        payload = {k: v for k, v in GENERATE_PAYLOAD.items() if k != "amount_of_mentions"}
        assert client.post("/generate", json=payload).status_code == 200
        assert mock_fn.call_args[1]["n_mentions"] == 100

    @patch("sentiment_bias.api.app.simulate", return_value=MOCK_TENSOR)
    def test_two_dimensional_input(self, mock_fn):
        payload = {
            "outlets": [{"outlet": "A", "z": [1.0, 0.5]}],
            "subjects": [{"subject": "X", "a": [0.8, 0.3], "b": 0.2}],
            "amount_of_mentions": 10,
        }
        body = client.post("/generate", json=payload).json()
        assert body["n_dimensions"] == 2
        assert mock_fn.call_args[0][0].shape == (1, 2)


# ---------------------------------------------------------------------------
# POST /generate — validation errors
# ---------------------------------------------------------------------------

class TestGenerateValidation:
    def test_missing_outlets_returns_422(self):
        payload = {k: v for k, v in GENERATE_PAYLOAD.items() if k != "outlets"}
        assert client.post("/generate", json=payload).status_code == 422

    def test_missing_subjects_returns_422(self):
        payload = {k: v for k, v in GENERATE_PAYLOAD.items() if k != "subjects"}
        assert client.post("/generate", json=payload).status_code == 422

    def test_negative_amount_of_mentions_returns_422(self):
        payload = {**GENERATE_PAYLOAD, "amount_of_mentions": -1}
        assert client.post("/generate", json=payload).status_code == 422

    def test_outlet_missing_z_returns_422(self):
        payload = {**GENERATE_PAYLOAD, "outlets": [{"outlet": "A"}]}
        assert client.post("/generate", json=payload).status_code == 422

    def test_subject_missing_a_returns_422(self):
        payload = {**GENERATE_PAYLOAD, "subjects": [{"subject": "X", "b": 0.1}]}
        assert client.post("/generate", json=payload).status_code == 422

    def test_subject_missing_b_returns_422(self):
        payload = {**GENERATE_PAYLOAD, "subjects": [{"subject": "X", "a": [0.5]}]}
        assert client.post("/generate", json=payload).status_code == 422

    def test_non_boolean_poisson_returns_422(self):
        payload = {**GENERATE_PAYLOAD, "poisson": "maybe"}
        assert client.post("/generate", json=payload).status_code == 422

    def test_empty_body_returns_422(self):
        assert client.post("/generate", json={}).status_code == 422


# ---------------------------------------------------------------------------
# Unmocked round trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_generate_output_is_accepted_by_analyze(self):
        """/generate must emit exactly the shape /analyze consumes."""
        generated = client.post("/generate", json={**GENERATE_PAYLOAD, "amount_of_mentions": 20})
        assert generated.status_code == 200
        replayed = client.post("/analyze", json=generated.json())
        assert replayed.status_code == 200
        body = replayed.json()
        assert {o["outlet"] for o in body["outlets"]} == {"A", "B"}
        assert {s["subject"] for s in body["subjects"]} == {"X", "Y"}

    def test_generate_respects_requested_total(self):
        payload = {**GENERATE_PAYLOAD, "amount_of_mentions": 20}
        entries = client.post("/generate", json=payload).json()["data"]
        assert sum(e["amount_of_mentions"] for e in entries) == 20 * 2 * 2
