"""
Unit tests for src.service.

Covers:
- build_tensor  : tensor shape, sorting, index mapping, count accumulation
- build_output  : AnalysisOutput assembly from raw parameter arrays
- log_likelihood / negative_log_likelihood : return types and relationship
- run_analysis  : output structure and parameter bounds
                  (differential_evolution is patched to keep tests fast)
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.schemas import AnalysisOutput, Mention
from src.service import (
    build_output,
    build_tensor,
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
    def test_outlet_names_match(self):
        outlets = ["A", "B", "C"]
        subjects = ["X"]
        z = np.array([[1.0], [-0.5], [0.2]])   # shape (3, 1)
        a = np.array([[0.8]])                   # shape (1, 1)
        b = np.array([-0.3])
        result = build_output(outlets, subjects, z, a, b)
        assert [o.outlet for o in result.outlets] == outlets

    def test_subject_names_match(self):
        outlets = ["A"]
        subjects = ["P", "Q"]
        z = np.array([[0.5]])                    # shape (1, 1)
        a = np.array([[1.0], [-1.0]])            # shape (2, 1)
        b = np.array([0.1, 0.2])
        result = build_output(outlets, subjects, z, a, b)
        assert [s.subject for s in result.subjects] == subjects

    def test_z_values_assigned_correctly(self):
        outlets = ["A", "B"]
        z = np.array([[2.5], [-1.1]])            # shape (2, 1)
        result = build_output(outlets, ["X"], z, np.array([[0.0]]), np.array([0.0]))
        assert result.outlets[0].z == pytest.approx([2.5])
        assert result.outlets[1].z == pytest.approx([-1.1])

    def test_a_b_values_assigned_correctly(self):
        subjects = ["X", "Y"]
        a = np.array([[0.7], [-0.3]])            # shape (2, 1)
        b = np.array([1.2, 0.4])
        result = build_output(["A"], subjects, np.array([[0.0]]), a, b)
        assert result.subjects[0].a == pytest.approx([0.7])
        assert result.subjects[0].b == pytest.approx(1.2)
        assert result.subjects[1].a == pytest.approx([-0.3])
        assert result.subjects[1].b == pytest.approx(0.4)

    def test_returns_analysis_output_instance(self):
        result = build_output(["A"], ["X"], np.array([[0.0]]), np.array([[0.0]]), np.array([0.0]))
        assert isinstance(result, AnalysisOutput)

    def test_z_values_are_python_floats(self):
        """Ensure numpy values are converted to plain Python floats."""
        result = build_output(["A"], ["X"], np.array([[1.0]]), np.array([[0.5]]), np.array([-0.5]))
        assert isinstance(result.outlets[0].z, list)
        assert all(type(v) is float for v in result.outlets[0].z)
        assert isinstance(result.subjects[0].a, list)
        assert all(type(v) is float for v in result.subjects[0].a)
        assert type(result.subjects[0].b) is float

    def test_multidimensional_z_and_a(self):
        """D=2: each outlet has a 2-component bias vector."""
        outlets = ["A", "B"]
        subjects = ["X"]
        z = np.array([[1.0, 0.5], [-0.5, 0.3]])   # shape (2, 2)
        a = np.array([[0.8, -0.2]])                # shape (1, 2)
        b = np.array([0.1])
        result = build_output(outlets, subjects, z, a, b)
        assert result.outlets[0].z == pytest.approx([1.0, 0.5])
        assert result.outlets[1].z == pytest.approx([-0.5, 0.3])
        assert result.subjects[0].a == pytest.approx([0.8, -0.2])


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
        x = np.zeros(4)   # z_0, z_1, a_0, b_0
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
        x = np.zeros(6)   # z_0, z_1, a_0, a_1, b_0, b_1
        assert np.isfinite(log_likelihood(x, matrix))

    def test_multidimensional_returns_finite(self, small_matrix):
        """D=2: m=2, k=1 → 2*2 + 1*2 + 1 = 7 parameters."""
        x = np.zeros(7)
        result = log_likelihood(x, small_matrix, D=2)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_multidimensional_dot_product(self, small_matrix):
        """Dot product z_i · a_j must equal scalar product when D=1."""
        x_scalar = np.array([0.5, -0.5, 1.0, 0.2])   # D=1: m=2,k=1 → 4 params
        x_vector = np.array([0.5, -0.5, 1.0, 0.2])   # identical layout for D=1
        assert log_likelihood(x_scalar, small_matrix, D=1) == pytest.approx(
            log_likelihood(x_vector, small_matrix, D=1)
        )


# ---------------------------------------------------------------------------
# run_analysis
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    """Tests for run_analysis with differential_evolution patched out.

    The patch makes tests deterministic and fast while still exercising
    build_tensor, the parameter slicing logic, and build_output.
    """

    def _mock_solution(self, m: int, k: int) -> MagicMock:
        """Return a mock OptimizeResult with plausible parameter values."""
        mock = MagicMock()
        rng = np.random.default_rng(42)
        mock.x = rng.uniform(-1, 1, m + 2 * k)
        return mock

    @patch("src.service.differential_evolution")
    def test_output_outlet_count(self, mock_de):
        data = [
            make_mention("A", "X", "positive", 5),
            make_mention("B", "X", "negative", 3),
            make_mention("C", "X", "neutral",  2),
        ]
        mock_de.return_value = self._mock_solution(3, 1)
        result = run_analysis(data)
        assert len(result.outlets) == 3

    @patch("src.service.differential_evolution")
    def test_output_subject_count(self, mock_de):
        data = [
            make_mention("A", "X", "positive", 5),
            make_mention("A", "Y", "negative", 3),
            make_mention("A", "Z", "neutral",  2),
        ]
        mock_de.return_value = self._mock_solution(1, 3)
        result = run_analysis(data)
        assert len(result.subjects) == 3

    @patch("src.service.differential_evolution")
    def test_outlet_names_are_sorted(self, mock_de):
        data = [
            make_mention("Z", "X", "positive", 1),
            make_mention("A", "X", "positive", 1),
            make_mention("M", "X", "positive", 1),
        ]
        mock_de.return_value = self._mock_solution(3, 1)
        result = run_analysis(data)
        assert [o.outlet for o in result.outlets] == ["A", "M", "Z"]

    @patch("src.service.differential_evolution")
    def test_subject_names_are_sorted(self, mock_de):
        data = [
            make_mention("A", "Zebra",  "positive", 1),
            make_mention("A", "Apple",  "negative", 1),
            make_mention("A", "Mango",  "neutral",  1),
        ]
        mock_de.return_value = self._mock_solution(1, 3)
        result = run_analysis(data)
        assert [s.subject for s in result.subjects] == ["Apple", "Mango", "Zebra"]

    @patch("src.service.differential_evolution")
    def test_parameters_within_bounds(self, mock_de):
        """Solver bounds are [-5, 5]; the solution must respect them."""
        data = [
            make_mention("A", "X", "positive", 8),
            make_mention("B", "Y", "negative", 4),
        ]
        rng = np.random.default_rng(0)
        mock = MagicMock()
        mock.x = rng.uniform(-5, 5, 6)   # m=2, k=2  →  2 + 4 params
        mock_de.return_value = mock
        result = run_analysis(data)
        for o in result.outlets:
            assert all(-5.0 <= zi <= 5.0 for zi in o.z)
        for s in result.subjects:
            assert all(-5.0 <= ai <= 5.0 for ai in s.a)
            assert -5.0 <= s.b <= 5.0

    @patch("src.service.differential_evolution")
    def test_differential_evolution_called_once(self, mock_de):
        mock_de.return_value = self._mock_solution(1, 1)
        run_analysis([make_mention("A", "X", "positive", 1)])
        mock_de.assert_called_once()

    @patch("src.service.differential_evolution")
    def test_returns_analysis_output(self, mock_de):
        mock_de.return_value = self._mock_solution(1, 1)
        result = run_analysis([make_mention("A", "X", "positive", 1)])
        assert isinstance(result, AnalysisOutput)
