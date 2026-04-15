"""
Unit tests for src.schemas.

Verifies that Pydantic models accept valid input and reject invalid input
with appropriate validation errors.
"""

import pytest
from pydantic import ValidationError

from src.schemas import (
    AnalysisInput,
    AnalysisOutput,
    Mention,
    OutletScore,
    SubjectScore,
)


# ---------------------------------------------------------------------------
# Mention
# ---------------------------------------------------------------------------

class TestMention:
    def test_valid_positive(self):
        m = Mention(outlet="CNN", subject="Economy", mention_type="positive", amount_of_mentions=5)
        assert m.outlet == "CNN"
        assert m.mention_type == "positive"
        assert m.amount_of_mentions == 5

    def test_valid_neutral(self):
        m = Mention(outlet="A", subject="B", mention_type="neutral", amount_of_mentions=0)
        assert m.mention_type == "neutral"

    def test_valid_negative(self):
        m = Mention(outlet="A", subject="B", mention_type="negative", amount_of_mentions=1)
        assert m.mention_type == "negative"

    def test_invalid_mention_type_raises(self):
        with pytest.raises(ValidationError):
            Mention(outlet="A", subject="B", mention_type="mixed", amount_of_mentions=1)

    def test_negative_amount_raises(self):
        with pytest.raises(ValidationError):
            Mention(outlet="A", subject="B", mention_type="positive", amount_of_mentions=-1)

    def test_zero_amount_is_valid(self):
        m = Mention(outlet="A", subject="B", mention_type="neutral", amount_of_mentions=0)
        assert m.amount_of_mentions == 0

    def test_missing_outlet_raises(self):
        with pytest.raises(ValidationError):
            Mention(subject="B", mention_type="positive", amount_of_mentions=1)

    def test_missing_subject_raises(self):
        with pytest.raises(ValidationError):
            Mention(outlet="A", mention_type="positive", amount_of_mentions=1)

    def test_missing_mention_type_raises(self):
        with pytest.raises(ValidationError):
            Mention(outlet="A", subject="B", amount_of_mentions=1)

    def test_missing_amount_raises(self):
        with pytest.raises(ValidationError):
            Mention(outlet="A", subject="B", mention_type="positive")


# ---------------------------------------------------------------------------
# AnalysisInput
# ---------------------------------------------------------------------------

class TestAnalysisInput:
    def test_valid_single_mention(self):
        m = Mention(outlet="A", subject="X", mention_type="positive", amount_of_mentions=3)
        inp = AnalysisInput(data=[m])
        assert len(inp.data) == 1

    def test_valid_multiple_mentions(self):
        mentions = [
            Mention(outlet="A", subject="X", mention_type="positive", amount_of_mentions=i)
            for i in range(5)
        ]
        inp = AnalysisInput(data=mentions)
        assert len(inp.data) == 5

    def test_empty_list_is_valid(self):
        """Pydantic accepts an empty list; business logic constraints are
        handled at the service layer."""
        inp = AnalysisInput(data=[])
        assert inp.data == []

    def test_invalid_mention_inside_list_raises(self):
        with pytest.raises(ValidationError):
            AnalysisInput(data=[{"outlet": "A", "subject": "X",
                                  "mention_type": "UNKNOWN", "amount_of_mentions": 1}])

    def test_missing_data_field_raises(self):
        with pytest.raises(ValidationError):
            AnalysisInput()


# ---------------------------------------------------------------------------
# OutletScore
# ---------------------------------------------------------------------------

class TestOutletScore:
    def test_valid(self):
        score = OutletScore(outlet="Reuters", z=[1.23])
        assert score.outlet == "Reuters"
        assert score.z == pytest.approx([1.23])

    def test_negative_z_is_valid(self):
        score = OutletScore(outlet="A", z=[-3.5])
        assert score.z == pytest.approx([-3.5])

    def test_multidimensional_z_is_valid(self):
        score = OutletScore(outlet="A", z=[1.0, -0.5, 0.3])
        assert score.z == pytest.approx([1.0, -0.5, 0.3])

    def test_missing_z_raises(self):
        with pytest.raises(ValidationError):
            OutletScore(outlet="A")


# ---------------------------------------------------------------------------
# SubjectScore
# ---------------------------------------------------------------------------

class TestSubjectScore:
    def test_valid(self):
        score = SubjectScore(subject="Economy", a=[0.8], b=-0.2)
        assert score.subject == "Economy"
        assert score.a == pytest.approx([0.8])
        assert score.b == pytest.approx(-0.2)

    def test_multidimensional_a_is_valid(self):
        score = SubjectScore(subject="X", a=[0.8, -0.2], b=0.1)
        assert score.a == pytest.approx([0.8, -0.2])

    def test_missing_a_raises(self):
        with pytest.raises(ValidationError):
            SubjectScore(subject="X", b=0.0)

    def test_missing_b_raises(self):
        with pytest.raises(ValidationError):
            SubjectScore(subject="X", a=[0.0])


# ---------------------------------------------------------------------------
# AnalysisOutput
# ---------------------------------------------------------------------------

class TestAnalysisOutput:
    def test_valid(self):
        output = AnalysisOutput(
            outlets=[OutletScore(outlet="A", z=[1.0])],
            subjects=[SubjectScore(subject="X", a=[0.5], b=-0.1)],
        )
        assert len(output.outlets) == 1
        assert len(output.subjects) == 1

    def test_empty_lists_are_valid(self):
        output = AnalysisOutput(outlets=[], subjects=[])
        assert output.outlets == []
        assert output.subjects == []
