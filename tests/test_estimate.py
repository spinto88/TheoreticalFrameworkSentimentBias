"""
Unit tests for sentiment_bias.estimate.fit.

`minimize` is patched out throughout, so these tests check the orchestration
around the solver — tensor building, parameter unpacking, canonicalization,
restart selection and BIC — without paying for a real optimisation.

The one test that actually optimises lives in test_simulate.py, marked slow.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sentiment_bias import fit
from sentiment_bias.data import from_array
from sentiment_bias.results import BiasFit


def n_params(m: int, k: int, n_dimensions: int = 1) -> int:
    return (m + k) * n_dimensions + k


def record(outlet: str, subject: str, mention_type: str, n: int) -> dict:
    return {
        "outlet": outlet,
        "subject": subject,
        "mention_type": mention_type,
        "amount_of_mentions": n,
    }


def solution(
    m: int,
    k: int,
    n_dimensions: int = 1,
    *,
    fun: float = 5.0,
    x: np.ndarray | None = None,
    success: bool = True,
) -> MagicMock:
    """A stand-in for scipy's OptimizeResult."""
    mock = MagicMock()
    mock.x = (
        x if x is not None
        else np.random.default_rng(42).uniform(-1, 1, n_params(m, k, n_dimensions))
    )
    mock.fun = fun
    mock.success = success
    return mock


SIMPLE_RECORDS = [
    record("A", "X", "positive", 5),
    record("B", "X", "negative", 3),
    record("C", "X", "neutral", 2),
]

ONE_RECORD = [record("A", "X", "positive", 1)]


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    @patch("sentiment_bias.estimate.minimize")
    def test_returns_bias_fit(self, mock_min):
        mock_min.return_value = solution(1, 1)
        assert isinstance(fit(ONE_RECORD), BiasFit)

    @patch("sentiment_bias.estimate.minimize")
    def test_outlet_count(self, mock_min):
        mock_min.return_value = solution(3, 1)
        assert len(fit(SIMPLE_RECORDS).z) == 3

    @patch("sentiment_bias.estimate.minimize")
    def test_subject_count(self, mock_min):
        data = [
            record("A", "X", "positive", 5),
            record("A", "Y", "negative", 3),
            record("A", "Z", "neutral", 2),
        ]
        mock_min.return_value = solution(1, 3)
        result = fit(data)
        assert len(result.a) == 3
        assert len(result.b) == 3

    @patch("sentiment_bias.estimate.minimize")
    def test_outlet_names_are_sorted(self, mock_min):
        data = [record(o, "X", "positive", 1) for o in ["Z", "A", "M"]]
        mock_min.return_value = solution(3, 1)
        assert list(fit(data).z.index) == ["A", "M", "Z"]

    @patch("sentiment_bias.estimate.minimize")
    def test_subject_names_are_sorted(self, mock_min):
        data = [record("A", s, "positive", 1) for s in ["Zebra", "Apple", "Mango"]]
        mock_min.return_value = solution(1, 3)
        result = fit(data)
        assert list(result.a.index) == ["Apple", "Mango", "Zebra"]
        assert list(result.b.index) == ["Apple", "Mango", "Zebra"]

    @patch("sentiment_bias.estimate.minimize")
    def test_z_and_a_are_frames_b_is_series(self, mock_min):
        mock_min.return_value = solution(1, 1)
        result = fit(ONE_RECORD)
        assert isinstance(result.z, pd.DataFrame)
        assert isinstance(result.a, pd.DataFrame)
        assert isinstance(result.b, pd.Series)

    @patch("sentiment_bias.estimate.minimize")
    def test_default_n_dimensions_gives_one_column(self, mock_min):
        mock_min.return_value = solution(1, 1)
        result = fit(ONE_RECORD)
        assert list(result.z.columns) == ["dim_1"]
        assert result.n_dimensions == 1

    @patch("sentiment_bias.estimate.minimize")
    def test_two_dimensions_give_two_columns(self, mock_min):
        mock_min.return_value = solution(1, 1, n_dimensions=2)
        result = fit(ONE_RECORD, n_dimensions=2)
        assert list(result.z.columns) == ["dim_1", "dim_2"]
        assert list(result.a.columns) == ["dim_1", "dim_2"]
        assert result.n_dimensions == 2

    @patch("sentiment_bias.estimate.minimize")
    def test_b_stays_scalar_per_subject_in_2d(self, mock_min):
        mock_min.return_value = solution(1, 1, n_dimensions=2)
        result = fit(ONE_RECORD, n_dimensions=2)
        assert result.b.shape == (1,)

    @patch("sentiment_bias.estimate.minimize")
    def test_data_carries_the_fitted_tensor(self, mock_min):
        mock_min.return_value = solution(3, 1)
        result = fit(SIMPLE_RECORDS)
        assert result.data.outlets == ["A", "B", "C"]
        assert result.data.n_mentions == 10

    @patch("sentiment_bias.estimate.minimize")
    def test_converged_reflects_solution_success(self, mock_min):
        mock_min.return_value = solution(1, 1, success=False)
        assert fit(ONE_RECORD).converged is False

    @patch("sentiment_bias.estimate.minimize")
    def test_x_matches_reported_parameters(self, mock_min):
        mock_min.return_value = solution(1, 1, n_dimensions=2)
        result = fit(ONE_RECORD, n_dimensions=2)
        expected = np.concatenate(
            [result.z.to_numpy().ravel(), result.a.to_numpy().ravel(), result.b.to_numpy()]
        )
        np.testing.assert_allclose(result.x, expected)


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------

class TestInputCoercion:
    @patch("sentiment_bias.estimate.minimize")
    def test_accepts_mention_tensor(self, mock_min):
        tensor = from_array(np.ones((2, 1, 3), dtype=np.int64))
        mock_min.return_value = solution(2, 1)
        assert list(fit(tensor).z.index) == tensor.outlets

    @patch("sentiment_bias.estimate.minimize")
    def test_accepts_wide_dataframe(self, mock_min):
        df = pd.DataFrame([
            {"outlet": "A", "subject": "X", "negative": 1, "neutral": 2, "positive": 3},
            {"outlet": "B", "subject": "X", "negative": 4, "neutral": 5, "positive": 6},
        ])
        mock_min.return_value = solution(2, 1)
        assert list(fit(df).z.index) == ["A", "B"]

    @patch("sentiment_bias.estimate.minimize")
    def test_accepts_long_dataframe(self, mock_min):
        mock_min.return_value = solution(3, 1)
        assert list(fit(pd.DataFrame(SIMPLE_RECORDS)).z.index) == ["A", "B", "C"]

    @patch("sentiment_bias.estimate.minimize")
    def test_accepts_raw_array(self, mock_min):
        mock_min.return_value = solution(2, 1)
        result = fit(np.ones((2, 1, 3), dtype=np.int64))
        assert list(result.z.index) == ["outlet_00", "outlet_01"]

    @patch("sentiment_bias.estimate.minimize")
    def test_counts_are_passed_to_the_objective(self, mock_min):
        mock_min.return_value = solution(3, 1)
        fit(SIMPLE_RECORDS)
        _, kwargs = mock_min.call_args
        counts = kwargs["args"][0]
        assert counts.shape == (3, 1, 3)
        assert counts.sum() == 10

    @patch("sentiment_bias.estimate.minimize")
    def test_n_dimensions_is_passed_to_the_objective(self, mock_min):
        mock_min.return_value = solution(1, 1, n_dimensions=2)
        fit(ONE_RECORD, n_dimensions=2)
        _, kwargs = mock_min.call_args
        assert kwargs["args"][1] == 2


# ---------------------------------------------------------------------------
# Solver invocation
# ---------------------------------------------------------------------------

class TestSolverInvocation:
    @patch("sentiment_bias.estimate.minimize")
    def test_called_once_by_default(self, mock_min):
        mock_min.return_value = solution(1, 1)
        fit(ONE_RECORD)
        mock_min.assert_called_once()

    @patch("sentiment_bias.estimate.minimize")
    def test_uses_lbfgsb_with_jacobian(self, mock_min):
        mock_min.return_value = solution(1, 1)
        fit(ONE_RECORD)
        _, kwargs = mock_min.call_args
        assert kwargs["method"] == "L-BFGS-B"
        assert kwargs["jac"] is not None

    @patch("sentiment_bias.estimate.minimize")
    def test_x0_length_1d(self, mock_min):
        mock_min.return_value = solution(3, 1)
        fit(SIMPLE_RECORDS)
        _, kwargs = mock_min.call_args
        assert len(kwargs["x0"]) == n_params(3, 1, 1)

    @patch("sentiment_bias.estimate.minimize")
    def test_x0_length_2d(self, mock_min):
        mock_min.return_value = solution(1, 1, n_dimensions=2)
        fit(ONE_RECORD, n_dimensions=2)
        _, kwargs = mock_min.call_args
        assert len(kwargs["x0"]) == n_params(1, 1, 2)

    @patch("sentiment_bias.estimate.minimize")
    def test_default_bounds_are_plus_minus_five(self, mock_min):
        mock_min.return_value = solution(1, 1)
        fit(ONE_RECORD)
        _, kwargs = mock_min.call_args
        bounds = kwargs["bounds"]
        assert list(bounds.lb) == [-5.0] * n_params(1, 1, 1)
        assert list(bounds.ub) == [5.0] * n_params(1, 1, 1)

    @patch("sentiment_bias.estimate.minimize")
    def test_custom_bounds_are_forwarded(self, mock_min):
        mock_min.return_value = solution(1, 1)
        fit(ONE_RECORD, bounds=(-2.0, 3.0))
        _, kwargs = mock_min.call_args
        assert list(kwargs["bounds"].lb) == [-2.0] * n_params(1, 1, 1)
        assert list(kwargs["bounds"].ub) == [3.0] * n_params(1, 1, 1)

    @patch("sentiment_bias.estimate.minimize")
    def test_custom_tol_is_forwarded(self, mock_min):
        mock_min.return_value = solution(1, 1)
        fit(ONE_RECORD, tol=1e-3)
        _, kwargs = mock_min.call_args
        assert kwargs["tol"] == pytest.approx(1e-3)

    @patch("sentiment_bias.estimate.minimize")
    def test_seed_makes_initialisation_reproducible(self, mock_min):
        mock_min.return_value = solution(3, 1)
        fit(SIMPLE_RECORDS, seed=123)
        first = mock_min.call_args[1]["x0"].copy()
        fit(SIMPLE_RECORDS, seed=123)
        np.testing.assert_allclose(mock_min.call_args[1]["x0"], first)

    @patch("sentiment_bias.estimate.minimize")
    def test_different_seeds_give_different_initialisation(self, mock_min):
        mock_min.return_value = solution(3, 1)
        fit(SIMPLE_RECORDS, seed=1)
        first = mock_min.call_args[1]["x0"].copy()
        fit(SIMPLE_RECORDS, seed=2)
        assert not np.allclose(mock_min.call_args[1]["x0"], first)

    @patch("sentiment_bias.estimate.minimize")
    def test_restarts_use_independent_starting_points(self, mock_min):
        mock_min.return_value = solution(3, 1)
        fit(SIMPLE_RECORDS, n_restarts=3, seed=0)
        starts = [call[1]["x0"] for call in mock_min.call_args_list]
        assert not np.allclose(starts[0], starts[1])
        assert not np.allclose(starts[1], starts[2])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_zero_dimensions_raises(self):
        with pytest.raises(ValueError, match="n_dimensions"):
            fit(ONE_RECORD, n_dimensions=0)

    def test_negative_dimensions_raises(self):
        with pytest.raises(ValueError, match="n_dimensions"):
            fit(ONE_RECORD, n_dimensions=-1)

    def test_zero_restarts_raises(self):
        with pytest.raises(ValueError, match="n_restarts"):
            fit(ONE_RECORD, n_restarts=0)

    def test_negative_restarts_raises(self):
        with pytest.raises(ValueError, match="n_restarts"):
            fit(ONE_RECORD, n_restarts=-3)


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

class TestCanonicalization:
    @patch("sentiment_bias.estimate.minimize")
    def test_1d_output_is_canonicalized(self, mock_min):
        """n_dimensions=1 output has sum(z) == 0 and sum(z**2) == 1."""
        data = [record(o, "X", "positive", 5) for o in ["A", "B", "C"]]
        # x = [z_A, z_B, z_C, a_X, b_X]
        mock_min.return_value = solution(
            3, 1, x=np.array([0.0, 3.0, 6.0, 0.5, 0.2])
        )
        z = fit(data).z["dim_1"].to_numpy()
        assert z.sum() == pytest.approx(0.0, abs=1e-9)
        assert (z ** 2).sum() == pytest.approx(1.0)

    @patch("sentiment_bias.estimate.minimize")
    def test_sign_reference_outlet_controls_sign(self, mock_min):
        data = [record(o, "X", "positive", 5) for o in ["A", "B"]]
        # x = [z_A, z_B, a_X, b_X]. Unit norm sends [1, -1] to [1, -1]/sqrt(2).
        mock_min.return_value = solution(2, 1, x=np.array([1.0, -1.0, 0.5, 0.2]))
        half = 1.0 / np.sqrt(2.0)

        default = fit(data).z["dim_1"]
        assert default["A"] == pytest.approx(half)
        assert default["B"] == pytest.approx(-half)

        flipped = fit(data, sign_reference_outlet="B", sign_reference_positive=True).z["dim_1"]
        assert flipped["B"] == pytest.approx(half)
        assert flipped["A"] == pytest.approx(-half)

    @patch("sentiment_bias.estimate.minimize")
    def test_sign_reference_positive_false(self, mock_min):
        data = [record(o, "X", "positive", 5) for o in ["A", "B"]]
        mock_min.return_value = solution(2, 1, x=np.array([1.0, -1.0, 0.5, 0.2]))
        z = fit(data, sign_reference_positive=False).z["dim_1"]
        assert z["A"] < 0

    @patch("sentiment_bias.estimate.minimize")
    def test_two_dimensions_not_canonicalized(self, mock_min):
        """n_dimensions=2 output is returned as-is, without standardization."""
        data = [record(o, "X", "positive", 1) for o in ["A", "B"]]
        # x = [z_A(2), z_B(2), a_X(2), b_X]
        mock_min.return_value = solution(
            2, 1, n_dimensions=2, x=np.array([0.0, 1.0, 3.0, -2.0, 0.5, 0.1, 0.2])
        )
        z = fit(data, n_dimensions=2).z
        np.testing.assert_allclose(z.loc["A"].to_numpy(), [0.0, 1.0])
        np.testing.assert_allclose(z.loc["B"].to_numpy(), [3.0, -2.0])

    @patch("sentiment_bias.estimate.minimize")
    def test_parameters_within_bounds_2d(self, mock_min):
        """n_dimensions=2 skips 1D canonicalization, so the raw box bounds still hold."""
        data = [record("A", "X", "positive", 8), record("B", "Y", "negative", 4)]
        mock_min.return_value = solution(
            2, 2, n_dimensions=2,
            x=np.random.default_rng(0).uniform(-5, 5, n_params(2, 2, 2)),
        )
        result = fit(data, n_dimensions=2)
        assert ((result.z.to_numpy() >= -5.0) & (result.z.to_numpy() <= 5.0)).all()
        assert ((result.a.to_numpy() >= -5.0) & (result.a.to_numpy() <= 5.0)).all()
        assert ((result.b.to_numpy() >= -5.0) & (result.b.to_numpy() <= 5.0)).all()


# ---------------------------------------------------------------------------
# Restarts
# ---------------------------------------------------------------------------

class TestRestarts:
    @patch("sentiment_bias.estimate.minimize")
    def test_calls_minimize_that_many_times(self, mock_min):
        mock_min.return_value = solution(1, 1)
        fit(ONE_RECORD, n_restarts=4)
        assert mock_min.call_count == 4

    @patch("sentiment_bias.estimate.minimize")
    def test_keeps_lowest_loss_solution(self, mock_min):
        solutions = [
            solution(1, 1, fun=9.0, x=np.array([1.0, 0.1, 0.1])),
            solution(1, 1, fun=2.0, x=np.array([2.0, 0.2, 0.2])),   # best
            solution(1, 1, fun=5.0, x=np.array([3.0, 0.3, 0.3])),
        ]
        mock_min.side_effect = solutions
        result = fit(ONE_RECORD, n_restarts=3)
        assert result.loss == pytest.approx(2.0)
        # single outlet -> canonicalization is a no-op, so a/b come from the winner
        assert result.a.loc["X", "dim_1"] == pytest.approx(0.2)
        assert result.b["X"] == pytest.approx(0.2)

    @patch("sentiment_bias.estimate.minimize")
    def test_best_found_regardless_of_position(self, mock_min):
        mock_min.side_effect = [
            solution(1, 1, fun=1.0),
            solution(1, 1, fun=4.0),
            solution(1, 1, fun=3.0),
        ]
        assert fit(ONE_RECORD, n_restarts=3).loss == pytest.approx(1.0)

    @patch("sentiment_bias.estimate.minimize")
    def test_n_restarts_reported(self, mock_min):
        mock_min.return_value = solution(1, 1)
        assert fit(ONE_RECORD, n_restarts=7).n_restarts == 7

    @patch("sentiment_bias.estimate.minimize")
    def test_default_n_restarts_is_one(self, mock_min):
        mock_min.return_value = solution(1, 1)
        assert fit(ONE_RECORD).n_restarts == 1

    @patch("sentiment_bias.estimate.minimize")
    def test_losses_records_every_restart_in_order(self, mock_min):
        mock_min.side_effect = [
            solution(1, 1, fun=9.0),
            solution(1, 1, fun=2.0),
            solution(1, 1, fun=5.0),
        ]
        assert fit(ONE_RECORD, n_restarts=3).losses == pytest.approx([9.0, 2.0, 5.0])

    @patch("sentiment_bias.estimate.minimize")
    def test_loss_is_the_minimum_of_losses(self, mock_min):
        mock_min.side_effect = [
            solution(1, 1, fun=9.0),
            solution(1, 1, fun=2.0),
            solution(1, 1, fun=5.0),
        ]
        result = fit(ONE_RECORD, n_restarts=3)
        assert result.loss == pytest.approx(min(result.losses))

    @patch("sentiment_bias.estimate.minimize")
    def test_losses_has_one_entry_by_default(self, mock_min):
        mock_min.return_value = solution(1, 1, fun=42.0)
        assert fit(ONE_RECORD).losses == pytest.approx([42.0])


# ---------------------------------------------------------------------------
# loss / bic
# ---------------------------------------------------------------------------

class TestLossAndBIC:
    @patch("sentiment_bias.estimate.minimize")
    def test_loss_comes_from_solution_fun(self, mock_min):
        mock_min.return_value = solution(1, 1, fun=42.0)
        result = fit(ONE_RECORD)
        assert result.loss == pytest.approx(42.0)
        assert isinstance(result.loss, float)

    @patch("sentiment_bias.estimate.minimize")
    def test_bic_is_finite_float(self, mock_min):
        mock_min.return_value = solution(1, 1, fun=5.0)
        result = fit([record("A", "X", "positive", 10)])
        assert isinstance(result.bic, float)
        assert np.isfinite(result.bic)

    @patch("sentiment_bias.estimate.minimize")
    def test_bic_matches_formula(self, mock_min):
        """BIC = 2*loss + n_params*ln(n_mentions)."""
        data = [record("A", "X", "positive", 10), record("A", "X", "negative", 5)]
        # m=1, k=1, D=1 -> n_params=3 ; n_mentions=15
        mock_min.return_value = solution(1, 1, fun=7.0)
        expected = 2 * 7.0 + 3 * np.log(15)
        assert fit(data).bic == pytest.approx(expected)

    @patch("sentiment_bias.estimate.minimize")
    def test_bic_larger_with_more_dimensions(self, mock_min):
        data = [record("A", "X", "positive", 8), record("B", "Y", "negative", 4)]
        mock_min.return_value = solution(2, 2, n_dimensions=1, fun=5.0)
        one = fit(data, n_dimensions=1).bic
        mock_min.return_value = solution(2, 2, n_dimensions=2, fun=5.0)
        two = fit(data, n_dimensions=2).bic
        assert two > one

    @patch("sentiment_bias.estimate.minimize")
    def test_bic_increases_with_higher_loss(self, mock_min):
        data = [record("A", "X", "positive", 10)]
        mock_min.return_value = solution(1, 1, fun=3.0)
        low = fit(data).bic
        mock_min.return_value = solution(1, 1, fun=20.0)
        assert fit(data).bic > low

    @patch("sentiment_bias.estimate.minimize")
    def test_bic_uses_best_restart_loss(self, mock_min):
        mock_min.side_effect = [solution(1, 1, fun=9.0), solution(1, 1, fun=2.0)]
        data = [record("A", "X", "positive", 10)]
        expected = 2 * 2.0 + n_params(1, 1, 1) * np.log(10)
        assert fit(data, n_restarts=2).bic == pytest.approx(expected)

    @patch("sentiment_bias.estimate.minimize")
    def test_n_parameters_matches_layout(self, mock_min):
        data = [record("A", "X", "positive", 8), record("B", "Y", "negative", 4)]
        mock_min.return_value = solution(2, 2, n_dimensions=1)
        assert fit(data).n_parameters == n_params(2, 2, 1)
