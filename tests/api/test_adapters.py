"""
Unit tests for sentiment_bias.api.adapters.

The adapters are the only bridge between the Pydantic wire format and the
core's numpy/pandas types, so these tests focus on round-trips: a payload
converted in and back out must survive unchanged.
"""

import numpy as np
import pandas as pd
import pytest

from sentiment_bias.api.adapters import (
    fit_to_output,
    scores_to_arrays,
    tensor_from_mentions,
    tensor_to_input,
)
from sentiment_bias.api.schemas import (
    AnalysisInput,
    AnalysisOutput,
    Mention,
    OutletScore,
    SubjectScore,
)
from sentiment_bias.data import MentionTensor, from_array
from sentiment_bias.results import BiasFit, dimension_columns


def mention(outlet: str, subject: str, mention_type: str, n: int) -> Mention:
    return Mention(
        outlet=outlet, subject=subject, mention_type=mention_type, amount_of_mentions=n
    )


def make_fit(z, a, b, outlets, subjects, *, loss=1.5, bic=7.5, n_restarts=1,
             log_likelihood=None) -> BiasFit:
    n_dimensions = z.shape[1]
    columns = dimension_columns(n_dimensions)
    counts = np.zeros((len(outlets), len(subjects), 3), dtype=np.int64)
    return BiasFit(
        z=pd.DataFrame(z, index=pd.Index(outlets, name="outlet"), columns=columns),
        a=pd.DataFrame(a, index=pd.Index(subjects, name="subject"), columns=columns),
        b=pd.Series(b, index=pd.Index(subjects, name="subject"), name="b"),
        loss=loss,
        log_likelihood=-loss if log_likelihood is None else log_likelihood,
        bic=bic,
        n_dimensions=n_dimensions,
        n_restarts=n_restarts,
        converged=True,
        losses=[loss],
        data=from_array(counts, outlets=outlets, subjects=subjects),
        x=np.concatenate([z.ravel(), a.ravel(), b]),
    )


MENTIONS = [
    mention("A", "X", "positive", 10),
    mention("A", "X", "negative", 3),
    mention("B", "X", "neutral", 6),
    mention("B", "Y", "positive", 5),
]


# ---------------------------------------------------------------------------
# tensor_from_mentions
# ---------------------------------------------------------------------------

class TestTensorFromMentions:
    def test_returns_mention_tensor(self):
        assert isinstance(tensor_from_mentions(MENTIONS), MentionTensor)

    def test_shape(self):
        assert tensor_from_mentions(MENTIONS).shape == (2, 2, 3)

    def test_names_sorted(self):
        tensor = tensor_from_mentions(MENTIONS)
        assert tensor.outlets == ["A", "B"]
        assert tensor.subjects == ["X", "Y"]

    def test_counts_placed_correctly(self):
        tensor = tensor_from_mentions(MENTIONS)
        assert tensor.counts[0, 0, 2] == 10   # A/X positive
        assert tensor.counts[0, 0, 0] == 3    # A/X negative
        assert tensor.counts[1, 0, 1] == 6    # B/X neutral

    def test_total_matches(self):
        assert tensor_from_mentions(MENTIONS).n_mentions == 24

    def test_pydantic_objects_are_read_by_attribute(self):
        """Mention is a BaseModel, not a Mapping — the adapter must not need keys."""
        tensor = tensor_from_mentions([mention("Solo", "Topic", "neutral", 7)])
        assert tensor.counts[0, 0, 1] == 7


# ---------------------------------------------------------------------------
# fit_to_output
# ---------------------------------------------------------------------------

class TestFitToOutput:
    @pytest.fixture
    def fit_1d(self):
        return make_fit(
            z=np.array([[1.0], [-0.5]]),
            a=np.array([[0.8], [-0.3]]),
            b=np.array([0.1, -0.2]),
            outlets=["A", "B"],
            subjects=["X", "Y"],
            loss=12.5,
            bic=30.25,
            n_restarts=3,
        )

    def test_returns_analysis_output(self, fit_1d):
        assert isinstance(fit_to_output(fit_1d), AnalysisOutput)

    def test_outlet_names_preserved_in_order(self, fit_1d):
        output = fit_to_output(fit_1d)
        assert [o.outlet for o in output.outlets] == ["A", "B"]

    def test_subject_names_preserved_in_order(self, fit_1d):
        output = fit_to_output(fit_1d)
        assert [s.subject for s in output.subjects] == ["X", "Y"]

    def test_z_values(self, fit_1d):
        output = fit_to_output(fit_1d)
        assert output.outlets[0].z == pytest.approx([1.0])
        assert output.outlets[1].z == pytest.approx([-0.5])

    def test_a_and_b_values(self, fit_1d):
        output = fit_to_output(fit_1d)
        assert output.subjects[0].a == pytest.approx([0.8])
        assert output.subjects[0].b == pytest.approx(0.1)
        assert output.subjects[1].a == pytest.approx([-0.3])
        assert output.subjects[1].b == pytest.approx(-0.2)

    def test_b_is_matched_by_name_not_position(self):
        """b is a Series; the adapter must align it by subject, not by index."""
        fit = make_fit(
            z=np.array([[1.0]]),
            a=np.array([[0.5], [0.6]]),
            b=np.array([0.1, 0.2]),
            outlets=["A"],
            subjects=["X", "Y"],
        )
        shuffled = fit.b.reindex(["Y", "X"])
        reordered = make_fit(
            z=fit.z.to_numpy(), a=fit.a.to_numpy(), b=shuffled.to_numpy(),
            outlets=["A"], subjects=["Y", "X"],
        )
        output = fit_to_output(reordered)
        by_name = {s.subject: s.b for s in output.subjects}
        assert by_name["X"] == pytest.approx(0.1)
        assert by_name["Y"] == pytest.approx(0.2)

    def test_scalars_carried_through(self, fit_1d):
        output = fit_to_output(fit_1d)
        assert output.loss == pytest.approx(12.5)
        assert output.bic == pytest.approx(30.25)
        assert output.n_restarts == 3

    def test_two_dimensional_vectors(self):
        fit = make_fit(
            z=np.array([[1.0, 0.5], [-0.5, 0.3]]),
            a=np.array([[0.8, 0.2]]),
            b=np.array([-0.1]),
            outlets=["A", "B"],
            subjects=["X"],
        )
        output = fit_to_output(fit)
        assert output.outlets[0].z == pytest.approx([1.0, 0.5])
        assert output.subjects[0].a == pytest.approx([0.8, 0.2])
        assert output.subjects[0].b == pytest.approx(-0.1)

    def test_z_is_list_of_floats(self, fit_1d):
        output = fit_to_output(fit_1d)
        assert isinstance(output.outlets[0].z, list)
        assert all(isinstance(v, float) for v in output.outlets[0].z)

    def test_b_is_python_float(self, fit_1d):
        assert type(fit_to_output(fit_1d).subjects[0].b) is float

    def test_agrees_with_biasfit_to_dict(self, fit_1d):
        """The adapter and BiasFit.to_dict must describe the same parameters."""
        output = fit_to_output(fit_1d)
        payload = fit_1d.to_dict()
        assert [o.outlet for o in output.outlets] == [o["outlet"] for o in payload["outlets"]]
        assert output.outlets[0].z == pytest.approx(payload["outlets"][0]["z"])
        assert output.subjects[0].b == pytest.approx(payload["subjects"][0]["b"])


# ---------------------------------------------------------------------------
# tensor_to_input
# ---------------------------------------------------------------------------

class TestTensorToInput:
    @pytest.fixture
    def tensor(self):
        counts = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int64)   # 1 outlet, 2 subjects
        return from_array(counts, outlets=["A"], subjects=["X", "Y"])

    def test_returns_analysis_input(self, tensor):
        assert isinstance(tensor_to_input(tensor, n_dimensions=1), AnalysisInput)

    def test_row_count_is_pairs_times_three(self, tensor):
        result = tensor_to_input(tensor, n_dimensions=1)
        assert len(result.data) == 1 * 2 * 3

    def test_n_dimensions_is_carried(self, tensor):
        assert tensor_to_input(tensor, n_dimensions=2).n_dimensions == 2

    def test_all_mention_types_present(self, tensor):
        result = tensor_to_input(tensor, n_dimensions=1)
        assert {m.mention_type for m in result.data} == {"negative", "neutral", "positive"}

    def test_all_names_present(self, tensor):
        result = tensor_to_input(tensor, n_dimensions=1)
        assert {m.outlet for m in result.data} == {"A"}
        assert {m.subject for m in result.data} == {"X", "Y"}

    def test_counts_survive(self, tensor):
        result = tensor_to_input(tensor, n_dimensions=1)
        lookup = {(m.outlet, m.subject, m.mention_type): m.amount_of_mentions for m in result.data}
        assert lookup[("A", "X", "negative")] == 1
        assert lookup[("A", "X", "neutral")] == 2
        assert lookup[("A", "X", "positive")] == 3
        assert lookup[("A", "Y", "positive")] == 6

    def test_total_preserved(self, tensor):
        result = tensor_to_input(tensor, n_dimensions=1)
        assert sum(m.amount_of_mentions for m in result.data) == tensor.n_mentions

    def test_round_trip_back_to_tensor(self, tensor):
        """tensor -> AnalysisInput -> tensor must be lossless."""
        result = tensor_to_input(tensor, n_dimensions=1)
        rebuilt = tensor_from_mentions(result.data)
        np.testing.assert_array_equal(rebuilt.counts, tensor.counts)
        assert rebuilt.outlets == tensor.outlets
        assert rebuilt.subjects == tensor.subjects

    def test_amounts_are_non_negative(self, tensor):
        result = tensor_to_input(tensor, n_dimensions=1)
        assert all(m.amount_of_mentions >= 0 for m in result.data)

    def test_zero_counts_are_emitted(self):
        """Empty cells still produce rows, so the shape is recoverable."""
        empty = from_array(np.zeros((1, 1, 3), dtype=np.int64), outlets=["A"], subjects=["X"])
        result = tensor_to_input(empty, n_dimensions=1)
        assert len(result.data) == 3
        assert all(m.amount_of_mentions == 0 for m in result.data)


# ---------------------------------------------------------------------------
# scores_to_arrays
# ---------------------------------------------------------------------------

class TestScoresToArrays:
    def test_shapes_1d(self):
        outlets = [OutletScore(outlet="A", z=[1.0]), OutletScore(outlet="B", z=[-1.0])]
        subjects = [SubjectScore(subject="X", a=[0.8], b=0.2)]
        z, a, b, names_o, names_s = scores_to_arrays(outlets, subjects)
        assert z.shape == (2, 1)
        assert a.shape == (1, 1)
        assert b.shape == (1,)
        assert names_o == ["A", "B"]
        assert names_s == ["X"]

    def test_shapes_2d(self):
        outlets = [OutletScore(outlet="A", z=[1.0, 0.5])]
        subjects = [SubjectScore(subject="X", a=[0.8, 0.3], b=0.2)]
        z, a, b, _, _ = scores_to_arrays(outlets, subjects)
        assert z.shape == (1, 2)
        assert a.shape == (1, 2)
        assert b.shape == (1,)

    def test_values(self):
        outlets = [OutletScore(outlet="A", z=[1.5]), OutletScore(outlet="B", z=[-2.5])]
        subjects = [SubjectScore(subject="X", a=[0.7], b=0.3)]
        z, a, b, _, _ = scores_to_arrays(outlets, subjects)
        np.testing.assert_allclose(z, [[1.5], [-2.5]])
        np.testing.assert_allclose(a, [[0.7]])
        np.testing.assert_allclose(b, [0.3])

    def test_dtype_is_float(self):
        z, a, b, _, _ = scores_to_arrays(
            [OutletScore(outlet="A", z=[1])], [SubjectScore(subject="X", a=[2], b=3)]
        )
        assert z.dtype == np.float64
        assert a.dtype == np.float64
        assert b.dtype == np.float64

    def test_order_is_preserved_not_sorted(self):
        """simulate() is handed the arrays positionally, so order must be kept."""
        outlets = [OutletScore(outlet="Z", z=[1.0]), OutletScore(outlet="A", z=[2.0])]
        subjects = [SubjectScore(subject="X", a=[0.5], b=0.0)]
        z, _, _, names, _ = scores_to_arrays(outlets, subjects)
        assert names == ["Z", "A"]
        np.testing.assert_allclose(z, [[1.0], [2.0]])
