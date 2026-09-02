"""
Integration tests for the FastAPI application (src.app).

Uses FastAPI's TestClient to exercise the HTTP layer without spinning up
a real server.  The analysis service is patched where noted so that tests
remain fast and deterministic.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.schemas import AnalysisInput, AnalysisOutput, Mention, OutletScore, SubjectScore

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "data": [
        {"outlet": "A", "subject": "X", "mention_type": "positive", "amount_of_mentions": 10},
        {"outlet": "A", "subject": "X", "mention_type": "negative", "amount_of_mentions": 3},
        {"outlet": "B", "subject": "X", "mention_type": "neutral",  "amount_of_mentions": 6},
        {"outlet": "B", "subject": "Y", "mention_type": "positive", "amount_of_mentions": 5},
    ]
}

MOCK_OUTPUT = AnalysisOutput(
    outlets=[OutletScore(outlet="A", z=[1.0]), OutletScore(outlet="B", z=[-0.5])],
    subjects=[SubjectScore(subject="X", a=[0.8], b=0.2), SubjectScore(subject="Y", a=[-0.3], b=0.1)],
    loss=12.34,
    bic=20.5,
)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndexRoute:
    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_html(self):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /analyze — happy path
# ---------------------------------------------------------------------------

class TestAnalyzeRoute:
    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_returns_200(self, _mock):
        response = client.post("/analyze", json=VALID_PAYLOAD)
        assert response.status_code == 200

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_response_has_outlets_and_subjects(self, _mock):
        body = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert "outlets"  in body
        assert "subjects" in body

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_outlet_schema(self, _mock):
        outlets = client.post("/analyze", json=VALID_PAYLOAD).json()["outlets"]
        assert all("outlet" in o and "z" in o for o in outlets)

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_subject_schema(self, _mock):
        subjects = client.post("/analyze", json=VALID_PAYLOAD).json()["subjects"]
        assert all("subject" in s and "a" in s and "b" in s for s in subjects)

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_outlet_count_matches_mock(self, _mock):
        body = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert len(body["outlets"]) == 2

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_subject_count_matches_mock(self, _mock):
        body = client.post("/analyze", json=VALID_PAYLOAD).json()
        assert len(body["subjects"]) == 2

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_z_values_are_lists_of_numbers(self, _mock):
        outlets = client.post("/analyze", json=VALID_PAYLOAD).json()["outlets"]
        assert all(isinstance(o["z"], list) for o in outlets)
        assert all(isinstance(v, (int, float)) for o in outlets for v in o["z"])

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_a_b_values_correct_types(self, _mock):
        subjects = client.post("/analyze", json=VALID_PAYLOAD).json()["subjects"]
        assert all(isinstance(s["a"], list) for s in subjects)
        assert all(isinstance(v, (int, float)) for s in subjects for v in s["a"])
        assert all(isinstance(s["b"], (int, float)) for s in subjects)

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_default_n_restarts_forwarded_to_service(self, mock_fn):
        client.post("/analyze", json=VALID_PAYLOAD)
        _, call_kwargs = mock_fn.call_args
        assert call_kwargs["n_restarts"] == 1

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_custom_n_restarts_forwarded_to_service(self, mock_fn):
        payload = {**VALID_PAYLOAD, "n_restarts": 10}
        client.post("/analyze", json=payload)
        _, call_kwargs = mock_fn.call_args
        assert call_kwargs["n_restarts"] == 10


# ---------------------------------------------------------------------------
# POST /analyze — validation errors
# ---------------------------------------------------------------------------

class TestAnalyzeValidation:
    def test_missing_data_field_returns_422(self):
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_invalid_mention_type_returns_422(self):
        payload = {
            "data": [{"outlet": "A", "subject": "X",
                      "mention_type": "UNKNOWN", "amount_of_mentions": 1}]
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_negative_amount_returns_422(self):
        payload = {
            "data": [{"outlet": "A", "subject": "X",
                      "mention_type": "positive", "amount_of_mentions": -1}]
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_missing_outlet_field_returns_422(self):
        payload = {
            "data": [{"subject": "X", "mention_type": "positive", "amount_of_mentions": 1}]
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_n_restarts_zero_returns_422(self):
        payload = {**VALID_PAYLOAD, "n_restarts": 0}
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422

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

MOCK_GENERATE_OUTPUT = AnalysisInput(
    data=[
        Mention(outlet="A", subject="X", mention_type="positive", amount_of_mentions=30),
        Mention(outlet="A", subject="X", mention_type="neutral",  amount_of_mentions=12),
        Mention(outlet="A", subject="X", mention_type="negative", amount_of_mentions=8),
    ],
    n_dimensions=1,
)


# ---------------------------------------------------------------------------
# POST /generate — happy path
# ---------------------------------------------------------------------------

class TestGenerateRoute:
    @patch("src.app.generate_data", return_value=MOCK_GENERATE_OUTPUT)
    def test_returns_200(self, _mock):
        response = client.post("/generate", json=GENERATE_PAYLOAD)
        assert response.status_code == 200

    @patch("src.app.generate_data", return_value=MOCK_GENERATE_OUTPUT)
    def test_response_has_data_and_n_dimensions(self, _mock):
        body = client.post("/generate", json=GENERATE_PAYLOAD).json()
        assert "data" in body
        assert "n_dimensions" in body

    @patch("src.app.generate_data", return_value=MOCK_GENERATE_OUTPUT)
    def test_n_dimensions_matches_mock(self, _mock):
        body = client.post("/generate", json=GENERATE_PAYLOAD).json()
        assert body["n_dimensions"] == MOCK_GENERATE_OUTPUT.n_dimensions

    @patch("src.app.generate_data", return_value=MOCK_GENERATE_OUTPUT)
    def test_data_entries_have_correct_fields(self, _mock):
        entries = client.post("/generate", json=GENERATE_PAYLOAD).json()["data"]
        for entry in entries:
            assert "outlet" in entry
            assert "subject" in entry
            assert "mention_type" in entry
            assert "amount_of_mentions" in entry

    @patch("src.app.generate_data", return_value=MOCK_GENERATE_OUTPUT)
    def test_data_entry_count_matches_mock(self, _mock):
        body = client.post("/generate", json=GENERATE_PAYLOAD).json()
        assert len(body["data"]) == len(MOCK_GENERATE_OUTPUT.data)

    @patch("src.app.generate_data", return_value=MOCK_GENERATE_OUTPUT)
    def test_generate_data_called_with_correct_args(self, mock_fn):
        client.post("/generate", json=GENERATE_PAYLOAD)
        mock_fn.assert_called_once()
        _, kwargs = mock_fn.call_args
        assert kwargs["amount_of_mentions"] == GENERATE_PAYLOAD["amount_of_mentions"]
        assert len(kwargs["outlets"]) == len(GENERATE_PAYLOAD["outlets"])
        assert len(kwargs["subjects"]) == len(GENERATE_PAYLOAD["subjects"])

    @patch("src.app.generate_data", return_value=MOCK_GENERATE_OUTPUT)
    def test_default_amount_of_mentions_accepted(self, _mock):
        payload = {k: v for k, v in GENERATE_PAYLOAD.items() if k != "amount_of_mentions"}
        response = client.post("/generate", json=payload)
        assert response.status_code == 200


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

    def test_empty_body_returns_422(self):
        assert client.post("/generate", json={}).status_code == 422
