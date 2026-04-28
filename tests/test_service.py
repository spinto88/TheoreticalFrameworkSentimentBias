"""
Unit tests for src.service.

Covers:
- build_tensor  : tensor shape, sorting, index mapping, count accumulation
- build_output  : AnalysisOutput assembly from raw parameter arrays (2-D)
- log_likelihood / negative_log_likelihood : return types and relationship
- grad_negative_log_likelihood : finite-difference check
- run_analysis  : output structure and parameter bounds
                  (minimize is patched to keep tests fast)
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.schemas import AnalysisOutput, Mention
from src.service import (
    build_output,
    build_tensor,
    grad_negative_log_likelihood,
    log_likelihood,
    negative_log_likelihood,
    run_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mention(outlet: str, subject: str, mtype: str, n: int) -> Mention:
    return Mention(
        outlet=outlet,
        subject=subject,
        mention_type=mtype,
        amount_of_mentions=n,
    )


SIMPLE_DATA = [
    make_mention("A", "X", "positive", 10),
    make_mention("A", "X", "negative", 3),
    make_mention("B", "X", "neutral",  6),
    make_mention("B", "Y", "positive", 5),
]


# ---------------------------------------------------------------------------
# build_tensor
# ---------------------------------------------------------------------------

class TestBuildTensor:
    def test_output_shape(self):
        matrix, outlets, subjects = build_tensor(SIMPLE_DATA)
        assert matrix.shape == (len(outlets), len(subjects), 3)

    def test_outlets_are_sorted(self):
        data = [
            make_mention("C", "X", "positive", 1),
            make_mention("A", "X", "positive", 1),
            make_mention("B", "X", "positive", 1),
        ]
        _, outlets, _ = build_tensor(data)
        assert outlets == ["A", "B", "C"]

    def test_subjects_are_sorted(self):
        data = [
            make_mention("A", "Z", "positive", 1),
            make_mention("A", "M", "positive", 1),
            make_mention("A", "A", "positive", 1),
        ]
        _, _, subjects = build_tensor(data)
        assert subjects == ["A", "M", "Z"]

    def test_positive_count_placed_at_index_2(self):
        data = [make_mention("A", "X", "positive", 7)]
        matrix, _, _ = build_tensor(data)
        assert matrix[0, 0, 2] == 7
        assert matrix[0, 0, 0] == 0  # negative
        assert matrix[0, 0, 1] == 0  # neutral

    def test_negative_count_placed_at_index_0(self):
        data = [make_mention("A", "X", "negative", 4)]
        matrix, _, _ = build_tensor(data)
        assert matrix[0, 0, 0] == 4

    def test_neutral_count_placed_at_index_1(self):
        data = [make_mention("A", "X", "neutral", 9)]
        matrix, _, _ = build_tensor(data)
        assert matrix[0, 0, 1] == 9

    def test_counts_accumulate_for_same_cell(self):
        """Two rows with the same (outlet, subject, type) must be summed."""
        data = [
            make_mention("A", "X", "positive", 5),
            make_mention("A", "X", "positive", 3),
        ]
        matrix, _, _ = build_tensor(data)
        assert matrix[0, 0, 2] == 8

    def test_multiple_outlets_and_subjects(self):
        matrix, outlets, subjects = build_tensor(SIMPLE_DATA)
        # outlets: A, B  —  subjects: X, Y
        assert outlets == ["A", "B"]
        assert subjects == ["X", "Y"]
        # A-X positive = 10, A-X negative = 3
        a_idx, x_idx = outlets.index("A"), subjects.index("X")
        assert matrix[a_idx, x_idx, 2] == 10
        assert matrix[a_idx, x_idx, 0] == 3
        # B-Y positive = 5
        b_idx, y_idx = outlets.index("B"), subjects.index("Y")
        assert matrix[b_idx, y_idx, 2] == 5

    def test_unobserved_cell_is_zero(self):
        """A-Y combination is absent from SIMPLE_DATA — its cell must be 0."""
        matrix, outlets, subjects = build_tensor(SIMPLE_DATA)
        a_idx, y_idx = outlets.index("A"), subjects.index("Y")
        assert matrix[a_idx, y_idx].sum() == 0

    def test_single_outlet_single_subject(self):
        data = [make_mention("Solo", "Topic", "neutral", 1)]
        matrix, outlets, subjects = build_tensor(data)
        assert matrix.shape == (1, 1, 3)
        assert outlets == ["Solo"]
        assert subjects == ["Topic"]


# ---------------------------------------------------------------------------
# build_output
# ---------------------------------------------------------------------------

class TestBuildOutput:
    """build_output receives 2-D arrays (m, D) / (k, D) and returns lists."""

    def test_outlet_names_match(self):
        outlets = ["A", "B", "C"]
        subjects = ["X"]
        z = np.array([[1.0], [-0.5], [0.2]])  # (3, 1)
        a = np.array([[0.8]])                  # (1, 1)
        b = np.array([[-0.3]])                 # (1, 1)
        result = build_output(outlets, subjects, z, a, b)
        assert [o.outlet for o in result.outlets] == outlets

    def test_subject_names_match(self):
        outlets = ["A"]
        subjects = ["P", "Q"]
        z = np.array([[0.5]])                   # (1, 1)
        a = np.array([[1.0], [-1.0]])           # (2, 1)
        b = np.array([[0.1], [0.2]])            # (2, 1)
        result = build_output(outlets, subjects, z, a, b)
        assert [s.subject for s in result.subjects] == subjects

    def test_z_values_assigned_correctly(self):
        outlets = ["A", "B"]
        z = np.array([[2.5], [-1.1]])           # (2, 1)
        result = build_output(outlets, ["X"], z, np.array([[0.0]]), np.array([[0.0]]))
        assert result.outlets[0].z == pytest.approx([2.5])
        assert result.outlets[1].z == pytest.approx([-1.1])

    def test_a_b_values_assigned_correctly(self):
        subjects = ["X", "Y"]
        a = np.array([[0.7], [-0.3]])           # (2, 1)
        b = np.array([[1.2], [0.4]])            # (2, 1)
        result = build_output(["A"], subjects, np.array([[0.0]]), a, b)
        assert result.subjects[0].a == pytest.approx([0.7])
        assert result.subjects[0].b == pytest.approx([1.2])
        assert result.subjects[1].a == pytest.approx([-0.3])
        assert result.subjects[1].b == pytest.approx([0.4])

    def test_two_dimensional_output(self):
        outlets = ["A", "B"]
        z = np.array([[1.0, 0.5], [-0.5, 0.3]])   # (2, 2)
        a = np.array([[0.8, 0.2]])                  # (1, 2)
        b = np.array([[-0.1, 0.4]])                 # (1, 2)
        result = build_output(outlets, ["X"], z, a, b)
        assert result.outlets[0].z == pytest.approx([1.0, 0.5])
        assert result.outlets[1].z == pytest.approx([-0.5, 0.3])
        assert result.subjects[0].a == pytest.approx([0.8, 0.2])
        assert result.subjects[0].b == pytest.approx([-0.1, 0.4])

    def test_returns_analysis_output_instance(self):
        result = build_output(["A"], ["X"], np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]]))
        assert isinstance(result, AnalysisOutput)

    def test_z_values_are_lists_of_python_floats(self):
        result = build_output(["A"], ["X"], np.array([[1.0]]), np.array([[0.5]]), np.array([[-0.5]]))
        assert isinstance(result.outlets[0].z, list)
        assert all(type(v) is float for v in result.outlets[0].z)
        assert isinstance(result.subjects[0].a, list)
        assert all(type(v) is float for v in result.subjects[0].a)
        assert isinstance(result.subjects[0].b, list)
        assert all(type(v) is float for v in result.subjects[0].b)


# ---------------------------------------------------------------------------
# log_likelihood / negative_log_likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    @pytest.fixture
    def small_matrix(self):
        """2 outlets × 1 subject × 3 sentiment types."""
        m = np.zeros((2, 1, 3), dtype=np.int64)
        m[0, 0, 2] = 10   # outlet 0, subject 0, positive
        m[1, 0, 0] = 5    # outlet 1, subject 0, negative
        return m

    def test_returns_finite_float(self, small_matrix):
        # n_dims=1: x has (2 + 2*1)*1 = 4 elements
        x = np.zeros(4)
        result = log_likelihood(x, small_matrix)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_negative_log_likelihood_is_negation(self, small_matrix):
        x = np.array([0.5, -0.5, 1.0, 0.2])
        assert negative_log_likelihood(x, small_matrix) == pytest.approx(
            -log_likelihood(x, small_matrix)
        )

    def test_l2_penalty_lowers_likelihood_for_large_params(self, small_matrix):
        """Large parameter values should score lower than moderate ones due
        to the Gaussian regularisation term."""
        x_moderate = np.array([1.0, -1.0, 0.5, 0.5])
        x_extreme  = np.array([4.9, -4.9, 4.9, 4.9])
        assert log_likelihood(x_moderate, small_matrix) > log_likelihood(x_extreme, small_matrix)

    def test_zero_counts_matrix(self):
        """All-zero counts should still return a finite likelihood."""
        matrix = np.zeros((2, 2, 3), dtype=np.int64)
        x = np.zeros(6)   # (2 + 2*2)*1 = 6
        assert np.isfinite(log_likelihood(x, matrix))

    def test_two_dimensional_returns_finite(self, small_matrix):
        # n_dims=2: x has (2 + 2*1)*2 = 8 elements
        x = np.zeros(8)
        assert np.isfinite(log_likelihood(x, small_matrix, n_dims=2))

    def test_two_dimensional_negation(self, small_matrix):
        x = np.random.default_rng(7).uniform(-1, 1, 8)
        assert negative_log_likelihood(x, small_matrix, n_dims=2) == pytest.approx(
            -log_likelihood(x, small_matrix, n_dims=2)
        )


# ---------------------------------------------------------------------------
# grad_negative_log_likelihood
# ---------------------------------------------------------------------------

class TestGradNegativeLogLikelihood:
    """Verify the analytical gradient against finite differences."""

    @pytest.fixture
    def small_matrix(self):
        m = np.zeros((2, 2, 3), dtype=np.int64)
        m[0, 0, 2] = 8
        m[0, 1, 0] = 3
        m[1, 0, 1] = 5
        m[1, 1, 2] = 2
        return m

    def _finite_diff_grad(self, x, matrix, n_dims, eps=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (negative_log_likelihood(xp, matrix, n_dims) -
                       negative_log_likelihood(xm, matrix, n_dims)) / (2 * eps)
        return grad

    def test_gradient_matches_finite_diff_1d(self, small_matrix):
        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, (2 + 2 * 2) * 1)
        analytical = grad_negative_log_likelihood(x, small_matrix, n_dims=1)
        numerical  = self._finite_diff_grad(x, small_matrix, n_dims=1)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)

    def test_gradient_matches_finite_diff_2d(self, small_matrix):
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, (2 + 2 * 2) * 2)
        analytical = grad_negative_log_likelihood(x, small_matrix, n_dims=2)
        numerical  = self._finite_diff_grad(x, small_matrix, n_dims=2)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)

    def test_gradient_shape_1d(self, small_matrix):
        x = np.zeros((2 + 2 * 2) * 1)
        g = grad_negative_log_likelihood(x, small_matrix, n_dims=1)
        assert g.shape == x.shape

    def test_gradient_shape_2d(self, small_matrix):
        x = np.zeros((2 + 2 * 2) * 2)
        g = grad_negative_log_likelihood(x, small_matrix, n_dims=2)
        assert g.shape == x.shape


# ---------------------------------------------------------------------------
# run_analysis
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    """Tests for run_analysis with minimize patched out.

    The patch makes tests deterministic and fast while still exercising
    build_tensor, the parameter slicing logic, and build_output.
    """

    def _mock_solution(self, m: int, k: int, n_dims: int = 1) -> MagicMock:
        """Return a mock OptimizeResult with plausible parameter values."""
        mock = MagicMock()
        rng = np.random.default_rng(42)
        mock.x = rng.uniform(-1, 1, (m + 2 * k) * n_dims)
        return mock

    @patch("src.service.minimize")
    def test_output_outlet_count(self, mock_min):
        data = [
            make_mention("A", "X", "positive", 5),
            make_mention("B", "X", "negative", 3),
            make_mention("C", "X", "neutral",  2),
        ]
        mock_min.return_value = self._mock_solution(3, 1)
        result = run_analysis(data)
        assert len(result.outlets) == 3

    @patch("src.service.minimize")
    def test_output_subject_count(self, mock_min):
        data = [
            make_mention("A", "X", "positive", 5),
            make_mention("A", "Y", "negative", 3),
            make_mention("A", "Z", "neutral",  2),
        ]
        mock_min.return_value = self._mock_solution(1, 3)
        result = run_analysis(data)
        assert len(result.subjects) == 3

    @patch("src.service.minimize")
    def test_outlet_names_are_sorted(self, mock_min):
        data = [
            make_mention("Z", "X", "positive", 1),
            make_mention("A", "X", "positive", 1),
            make_mention("M", "X", "positive", 1),
        ]
        mock_min.return_value = self._mock_solution(3, 1)
        result = run_analysis(data)
        assert [o.outlet for o in result.outlets] == ["A", "M", "Z"]

    @patch("src.service.minimize")
    def test_subject_names_are_sorted(self, mock_min):
        data = [
            make_mention("A", "Zebra",  "positive", 1),
            make_mention("A", "Apple",  "negative", 1),
            make_mention("A", "Mango",  "neutral",  1),
        ]
        mock_min.return_value = self._mock_solution(1, 3)
        result = run_analysis(data)
        assert [s.subject for s in result.subjects] == ["Apple", "Mango", "Zebra"]

    @patch("src.service.minimize")
    def test_parameters_within_bounds(self, mock_min):
        """Solver bounds are [-5, 5]; the solution must respect them."""
        data = [
            make_mention("A", "X", "positive", 8),
            make_mention("B", "Y", "negative", 4),
        ]
        rng = np.random.default_rng(0)
        mock = MagicMock()
        mock.x = rng.uniform(-5, 5, 6)   # m=2, k=2, n_dims=1 → 6 params
        mock_min.return_value = mock
        result = run_analysis(data)
        for o in result.outlets:
            assert all(-5.0 <= v <= 5.0 for v in o.z)
        for s in result.subjects:
            assert all(-5.0 <= v <= 5.0 for v in s.a)
            assert all(-5.0 <= v <= 5.0 for v in s.b)

    @patch("src.service.minimize")
    def test_minimize_called_once(self, mock_min):
        mock_min.return_value = self._mock_solution(1, 1)
        run_analysis([make_mention("A", "X", "positive", 1)])
        mock_min.assert_called_once()

    @patch("src.service.minimize")
    def test_returns_analysis_output(self, mock_min):
        mock_min.return_value = self._mock_solution(1, 1)
        result = run_analysis([make_mention("A", "X", "positive", 1)])
        assert isinstance(result, AnalysisOutput)

    @patch("src.service.minimize")
    def test_z_is_list_of_floats(self, mock_min):
        mock_min.return_value = self._mock_solution(1, 1)
        result = run_analysis([make_mention("A", "X", "positive", 1)])
        assert isinstance(result.outlets[0].z, list)

    @patch("src.service.minimize")
    def test_two_dimensions_doubles_param_count(self, mock_min):
        """With n_dims=2, minimize must be called with twice as many parameters."""
        data = [make_mention("A", "X", "positive", 1)]
        mock_min.return_value = self._mock_solution(1, 1, n_dims=2)
        run_analysis(data, n_dims=2)
        _, call_kwargs = mock_min.call_args
        # x0 should have (1 + 2*1)*2 = 6 elements
        assert len(call_kwargs["x0"]) == 6

    @patch("src.service.minimize")
    def test_two_dimensions_output_z_length(self, mock_min):
        """With n_dims=2, each outlet z vector must have length 2."""
        data = [make_mention("A", "X", "positive", 1)]
        mock_min.return_value = self._mock_solution(1, 1, n_dims=2)
        result = run_analysis(data, n_dims=2)
        assert len(result.outlets[0].z) == 2
        assert len(result.subjects[0].a) == 2
        assert len(result.subjects[0].b) == 2
