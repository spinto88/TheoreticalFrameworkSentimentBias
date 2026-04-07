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
from src.schemas import AnalysisOutput, OutletScore, SubjectScore

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
    outlets=[OutletScore(outlet="A", z=1.0), OutletScore(outlet="B", z=-0.5)],
    subjects=[SubjectScore(subject="X", a=0.8, b=0.2), SubjectScore(subject="Y", a=-0.3, b=0.1)],
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
    def test_z_values_are_numeric(self, _mock):
        outlets = client.post("/analyze", json=VALID_PAYLOAD).json()["outlets"]
        assert all(isinstance(o["z"], (int, float)) for o in outlets)

    @patch("src.app.run_analysis", return_value=MOCK_OUTPUT)
    def test_a_b_values_are_numeric(self, _mock):
        subjects = client.post("/analyze", json=VALID_PAYLOAD).json()["subjects"]
        assert all(isinstance(s["a"], (int, float)) for s in subjects)
        assert all(isinstance(s["b"], (int, float)) for s in subjects)


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

    def test_wrong_content_type_returns_422(self):
        response = client.post("/analyze", content="not json",
                               headers={"Content-Type": "text/plain"})
        assert response.status_code == 422
