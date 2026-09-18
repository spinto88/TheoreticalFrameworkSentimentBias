"""
Unit tests for sentiment_bias.results.BiasFit.

BiasFit is built by fit(), but it is a plain frozen dataclass, so these tests
construct it directly with known parameters — no optimiser involved.
"""

import numpy as np
import pandas as pd
import pytest

from sentiment_bias.data import from_array
from sentiment_bias.model import log_odds
from sentiment_bias.results import BiasFit, dimension_columns


def make_fit(
    z: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    outlets: list[str],
    subjects: list[str],
    *,
    loss: float = 1.0,
    log_likelihood: float | None = None,
    bic: float = 2.0,
    n_restarts: int = 1,
    converged: bool = True,
    losses: list[float] | None = None,
) -> BiasFit:
    """Assemble a BiasFit from raw arrays, the way fit() does."""
    n_dimensions = z.shape[1]
    columns = dimension_columns(n_dimensions)
    counts = np.zeros((len(outlets), len(subjects), 3), dtype=np.int64)
    counts[..., 1] = 10   # 10 neutral mentions in every cell
    return BiasFit(
        z=pd.DataFrame(z, index=pd.Index(outlets, name="outlet"), columns=columns),
        a=pd.DataFrame(a, index=pd.Index(subjects, name="subject"), columns=columns),
        b=pd.Series(b, index=pd.Index(subjects, name="subject"), name="b"),
        loss=loss,
        log_likelihood=-loss if log_likelihood is None else log_likelihood,
        bic=bic,
        n_dimensions=n_dimensions,
        n_restarts=n_restarts,
        converged=converged,
        losses=losses if losses is not None else [loss],
        data=from_array(counts, outlets=outlets, subjects=subjects),
        x=np.concatenate([z.ravel(), a.ravel(), b]),
    )


@pytest.fixture
def fit_1d():
    return make_fit(
        z=np.array([[1.0], [-0.5], [0.2]]),
        a=np.array([[0.8], [-0.3]]),
        b=np.array([0.1, -0.2]),
        outlets=["A", "B", "C"],
        subjects=["X", "Y"],
        loss=12.5,
        bic=30.25,
    )


@pytest.fixture
def fit_2d():
    return make_fit(
        z=np.array([[1.0, 0.5], [-0.5, 0.3]]),
        a=np.array([[0.8, 0.2]]),
        b=np.array([-0.1]),
        outlets=["A", "B"],
        subjects=["X"],
    )


# ---------------------------------------------------------------------------
# dimension_columns
# ---------------------------------------------------------------------------

class TestDimensionColumns:
    def test_one_dimension(self):
        assert dimension_columns(1) == ["dim_1"]

    def test_two_dimensions(self):
        assert dimension_columns(2) == ["dim_1", "dim_2"]

    def test_labels_are_one_based(self):
        assert dimension_columns(3) == ["dim_1", "dim_2", "dim_3"]


# ---------------------------------------------------------------------------
# Shapes and indices
# ---------------------------------------------------------------------------

class TestStructure:
    def test_z_is_frame_indexed_by_outlet(self, fit_1d):
        assert isinstance(fit_1d.z, pd.DataFrame)
        assert list(fit_1d.z.index) == ["A", "B", "C"]
        assert fit_1d.z.index.name == "outlet"

    def test_a_is_frame_indexed_by_subject(self, fit_1d):
        assert isinstance(fit_1d.a, pd.DataFrame)
        assert list(fit_1d.a.index) == ["X", "Y"]
        assert fit_1d.a.index.name == "subject"

    def test_b_is_series_not_frame(self, fit_1d):
        assert isinstance(fit_1d.b, pd.Series)
        assert fit_1d.b.name == "b"

    def test_b_lookup_by_subject_name(self, fit_1d):
        assert fit_1d.b["X"] == pytest.approx(0.1)
        assert fit_1d.b["Y"] == pytest.approx(-0.2)

    def test_z_lookup_by_name_and_dimension(self, fit_1d):
        assert fit_1d.z.loc["A", "dim_1"] == pytest.approx(1.0)
        assert fit_1d.z.loc["B", "dim_1"] == pytest.approx(-0.5)

    def test_columns_are_dim_labels_1d(self, fit_1d):
        assert list(fit_1d.z.columns) == ["dim_1"]
        assert list(fit_1d.a.columns) == ["dim_1"]

    def test_columns_are_dim_labels_2d(self, fit_2d):
        assert list(fit_2d.z.columns) == ["dim_1", "dim_2"]
        assert list(fit_2d.a.columns) == ["dim_1", "dim_2"]

    def test_outlets_property_matches_z_index(self, fit_1d):
        assert fit_1d.outlets == list(fit_1d.z.index)

    def test_is_frozen(self, fit_1d):
        with pytest.raises(Exception):
            fit_1d.loss = 0.0

    def test_data_is_the_fitted_tensor(self, fit_1d):
        assert fit_1d.data.outlets == ["A", "B", "C"]
        assert fit_1d.data.subjects == ["X", "Y"]

    def test_x_length_matches_parameter_count(self, fit_1d):
        assert len(fit_1d.x) == fit_1d.n_parameters

    def test_x_length_matches_parameter_count_2d(self, fit_2d):
        assert len(fit_2d.x) == fit_2d.n_parameters


# ---------------------------------------------------------------------------
# subjects
# ---------------------------------------------------------------------------

class TestSubjects:
    def test_joins_a_and_b(self, fit_1d):
        frame = fit_1d.subjects
        assert list(frame.columns) == ["dim_1", "b"]

    def test_one_row_per_subject(self, fit_1d):
        assert list(fit_1d.subjects.index) == ["X", "Y"]

    def test_values_align(self, fit_1d):
        frame = fit_1d.subjects
        assert frame.loc["X", "dim_1"] == pytest.approx(0.8)
        assert frame.loc["X", "b"] == pytest.approx(0.1)
        assert frame.loc["Y", "dim_1"] == pytest.approx(-0.3)
        assert frame.loc["Y", "b"] == pytest.approx(-0.2)

    def test_two_dimensional(self, fit_2d):
        assert list(fit_2d.subjects.columns) == ["dim_1", "dim_2", "b"]


# ---------------------------------------------------------------------------
# n_parameters
# ---------------------------------------------------------------------------

class TestNParameters:
    def test_one_dimension(self, fit_1d):
        # m=3, k=2, D=1 -> (3+2)*1 + 2 = 7
        assert fit_1d.n_parameters == 7

    def test_two_dimensions(self, fit_2d):
        # m=2, k=1, D=2 -> (2+1)*2 + 1 = 7
        assert fit_2d.n_parameters == 7

    def test_grows_with_dimensions(self):
        one = make_fit(
            np.zeros((4, 1)), np.zeros((2, 1)), np.zeros(2), ["A", "B", "C", "D"], ["X", "Y"]
        )
        two = make_fit(
            np.zeros((4, 2)), np.zeros((2, 2)), np.zeros(2), ["A", "B", "C", "D"], ["X", "Y"]
        )
        assert two.n_parameters > one.n_parameters


# ---------------------------------------------------------------------------
# log_odds
# ---------------------------------------------------------------------------

class TestLogOdds:
    def test_shape(self, fit_1d):
        assert fit_1d.log_odds().shape == (3, 2)

    def test_index_and_columns(self, fit_1d):
        frame = fit_1d.log_odds()
        assert list(frame.index) == ["A", "B", "C"]
        assert list(frame.columns) == ["X", "Y"]

    def test_matches_model_log_odds(self, fit_1d):
        expected = log_odds(
            fit_1d.z.to_numpy(), fit_1d.a.to_numpy(), fit_1d.b.to_numpy()
        )
        np.testing.assert_allclose(fit_1d.log_odds().to_numpy(), expected)

    def test_known_value(self, fit_1d):
        # q[A, X] = 1.0 * 0.8 + 0.1 = 0.9
        assert fit_1d.log_odds().loc["A", "X"] == pytest.approx(0.9)

    def test_two_dimensional_sums_over_dimensions(self, fit_2d):
        # q[A, X] = 1.0*0.8 + 0.5*0.2 - 0.1 = 0.8
        assert fit_2d.log_odds().loc["A", "X"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_top_level_keys(self, fit_1d):
        assert set(fit_1d.to_dict()) == {"outlets", "subjects", "loss", "bic", "n_restarts"}

    def test_outlet_entries(self, fit_1d):
        entries = fit_1d.to_dict()["outlets"]
        assert [e["outlet"] for e in entries] == ["A", "B", "C"]
        assert entries[0]["z"] == pytest.approx([1.0])

    def test_subject_entries(self, fit_1d):
        entries = fit_1d.to_dict()["subjects"]
        assert [e["subject"] for e in entries] == ["X", "Y"]
        assert entries[0]["a"] == pytest.approx([0.8])
        assert entries[0]["b"] == pytest.approx(0.1)

    def test_b_is_python_float(self, fit_1d):
        assert type(fit_1d.to_dict()["subjects"][0]["b"]) is float

    def test_z_is_list(self, fit_1d):
        assert isinstance(fit_1d.to_dict()["outlets"][0]["z"], list)

    def test_two_dimensional_vectors_have_length_two(self, fit_2d):
        payload = fit_2d.to_dict()
        assert len(payload["outlets"][0]["z"]) == 2
        assert len(payload["subjects"][0]["a"]) == 2

    def test_scalars_are_carried_through(self, fit_1d):
        payload = fit_1d.to_dict()
        assert payload["loss"] == pytest.approx(12.5)
        assert payload["bic"] == pytest.approx(30.25)
        assert payload["n_restarts"] == 1


# ---------------------------------------------------------------------------
# repr / summary
# ---------------------------------------------------------------------------

class TestReprAndSummary:
    def test_repr_is_single_line(self, fit_1d):
        assert "\n" not in repr(fit_1d)

    def test_repr_reports_dimensions_and_counts(self, fit_1d):
        text = repr(fit_1d)
        assert "D=1" in text
        assert "outlets=3" in text
        assert "subjects=2" in text

    def test_repr_reports_convergence(self, fit_1d):
        assert "converged" in repr(fit_1d)

    def test_repr_reports_non_convergence(self):
        fit = make_fit(
            np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), ["A"], ["X"], converged=False
        )
        assert "not-converged" in repr(fit)

    def test_summary_is_multiline(self, fit_1d):
        assert "\n" in fit_1d.summary()

    def test_summary_mentions_every_outlet_when_few(self, fit_1d):
        text = fit_1d.summary()
        for outlet in ["A", "B", "C"]:
            assert outlet in text

    def test_summary_includes_subject_table(self, fit_1d):
        assert "Subjects:" in fit_1d.summary()

    def test_summary_reports_mention_and_parameter_counts(self, fit_1d):
        text = fit_1d.summary()
        assert str(fit_1d.data.n_mentions) in text
        assert str(fit_1d.n_parameters) in text

    def test_summary_caps_outlet_listing(self):
        """With many outlets the summary shows only the extremes, not all of them."""
        n = 40
        fit = make_fit(
            z=np.linspace(-2, 2, n).reshape(-1, 1),
            a=np.array([[1.0]]),
            b=np.array([0.0]),
            outlets=[f"outlet_{i:02d}" for i in range(n)],
            subjects=["X"],
        )
        listed = sum(1 for name in fit.outlets if name in fit.summary())
        assert listed == 10


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

class TestLosses:
    def test_one_entry_per_restart(self):
        fit = make_fit(
            np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), ["A"], ["X"],
            n_restarts=3, losses=[9.0, 2.0, 5.0], loss=2.0,
        )
        assert len(fit.losses) == fit.n_restarts

    def test_loss_is_the_minimum_of_losses(self):
        fit = make_fit(
            np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), ["A"], ["X"],
            n_restarts=3, losses=[9.0, 2.0, 5.0], loss=2.0,
        )
        assert fit.loss == pytest.approx(min(fit.losses))


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    """Unlike loss, this one corresponds to the parameters actually reported."""

    def test_is_carried_through(self):
        fit = make_fit(
            np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), ["A"], ["X"],
            loss=5.0, log_likelihood=-4.25,
        )
        assert fit.log_likelihood == pytest.approx(-4.25)

    @pytest.mark.slow
    def test_matches_a_recomputation_from_the_reported_parameters(self):
        """The whole point: loss cannot be reproduced from z/a/b, this can."""
        import sentiment_bias as sb
        from sentiment_bias import model

        tensor = sb.simulate(
            np.linspace(-1.5, 1.5, 12), np.array([1.0, -0.6]), np.array([0.1, -0.1]),
            n_mentions=200, seed=4,
        )
        fit = sb.fit(tensor, n_dimensions=1, n_restarts=4, seed=0)
        recomputed = model.log_likelihood(
            model.pack(fit.z.to_numpy(), fit.a.to_numpy(), fit.b.to_numpy()),
            tensor.counts, 1, penalized=False,
        )
        assert fit.log_likelihood == pytest.approx(recomputed)

        # loss, by contrast, is evaluated at the optimiser's point.
        penalised_here = model.negative_log_likelihood(fit.x, tensor.counts, 1)
        assert penalised_here > fit.loss

    @pytest.mark.slow
    def test_is_invariant_under_rotation(self):
        import sentiment_bias as sb
        from sentiment_bias import model

        rng = np.random.default_rng(0)
        z = rng.normal(size=(15, 2))
        a = rng.normal(size=(4, 2))
        b = rng.normal(0, 0.1, 4)
        tensor = sb.simulate(z, a, b, n_mentions=200, seed=2)
        fit = sb.fit(tensor, n_dimensions=2, n_restarts=4, seed=0)

        angle = 0.9
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin], [sin, cos]])
        rotated = model.pack(
            fit.z.to_numpy() @ rotation.T, fit.a.to_numpy() @ rotation.T, fit.b.to_numpy()
        )
        assert model.log_likelihood(
            rotated, tensor.counts, 2, penalized=False
        ) == pytest.approx(fit.log_likelihood)
