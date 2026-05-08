"""
Unit tests for src.service.

Covers:
- build_tensor  : tensor shape, sorting, index mapping, count accumulation
- build_output  : AnalysisOutput assembly — z/a as (m/k, D) arrays, b as (k,)
- log_likelihood / negative_log_likelihood : return types and relationship
- grad_negative_log_likelihood : finite-difference verification
- run_analysis  : output structure and parameter bounds
                  (minimize is patched to keep tests fast)

Parameter vector layout: x = [z (m*D), a (k*D), b (k)]
Total length: (m + k) * n_dims + k
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.schemas import AnalysisInput, AnalysisOutput, Mention, OutletScore, SubjectScore
from src.service import (
    aproximate_bayesian_information_criteria,
    build_output,
    build_tensor,
    generate_data,
    generate_mentions,
    grad_negative_log_likelihood,
    log_likelihood,
    negative_log_likelihood,
    run_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mention(outlet: str, subject: str, mtype: str, n: int) -> Mention:
    return Mention(
        outlet=outlet,
        subject=subject,
        mention_type=mtype,
        amount_of_mentions=n,
    )


def n_params(m: int, k: int, n_dims: int = 1) -> int:
    """Total parameter count for the flat optimisation vector."""
    return (m + k) * n_dims + k


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
        assert matrix[0, 0, 0] == 0
        assert matrix[0, 0, 1] == 0

    def test_negative_count_placed_at_index_0(self):
        data = [make_mention("A", "X", "negative", 4)]
        matrix, _, _ = build_tensor(data)
        assert matrix[0, 0, 0] == 4

    def test_neutral_count_placed_at_index_1(self):
        data = [make_mention("A", "X", "neutral", 9)]
        matrix, _, _ = build_tensor(data)
        assert matrix[0, 0, 1] == 9

    def test_counts_accumulate_for_same_cell(self):
        data = [
            make_mention("A", "X", "positive", 5),
            make_mention("A", "X", "positive", 3),
        ]
        matrix, _, _ = build_tensor(data)
        assert matrix[0, 0, 2] == 8

    def test_multiple_outlets_and_subjects(self):
        matrix, outlets, subjects = build_tensor(SIMPLE_DATA)
        assert outlets == ["A", "B"]
        assert subjects == ["X", "Y"]
        a_idx, x_idx = outlets.index("A"), subjects.index("X")
        assert matrix[a_idx, x_idx, 2] == 10
        assert matrix[a_idx, x_idx, 0] == 3
        b_idx, y_idx = outlets.index("B"), subjects.index("Y")
        assert matrix[b_idx, y_idx, 2] == 5

    def test_unobserved_cell_is_zero(self):
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
# aproximate_bayesian_information_criteria
# ---------------------------------------------------------------------------

class TestApproximateBIC:
    """Unit tests for the BIC formula: 2*neglogl + n_params*ln(n_data)."""

    def test_known_value_zero_penalty(self):
        # ln(1) == 0, so penalty term vanishes
        assert aproximate_bayesian_information_criteria(5.0, 4, 1) == pytest.approx(10.0)

    def test_known_value_e_data(self):
        # ln(e) == 1, so bic = 2*neglogl + n_params
        assert aproximate_bayesian_information_criteria(3.0, 4, int(np.e * 1e6) // int(1e6)) == pytest.approx(
            2 * 3.0 + 4 * np.log(int(np.e * 1e6) // int(1e6)), rel=1e-3
        )

    def test_known_value_explicit(self):
        neglogl, n_params, n_data = 7.0, 3, 100
        expected = 2 * 7.0 + 3 * np.log(100)
        assert aproximate_bayesian_information_criteria(neglogl, n_params, n_data) == pytest.approx(expected)

    def test_returns_float(self):
        result = aproximate_bayesian_information_criteria(1.0, 2, 10)
        assert isinstance(result, float)

    def test_result_is_finite(self):
        assert np.isfinite(aproximate_bayesian_information_criteria(5.0, 6, 50))

    def test_increases_with_more_parameters(self):
        # Same neglogl and n_data, more parameters → larger BIC
        bic_few = aproximate_bayesian_information_criteria(5.0, 3, 100)
        bic_many = aproximate_bayesian_information_criteria(5.0, 10, 100)
        assert bic_many > bic_few

    def test_increases_with_higher_neglogl(self):
        bic_low = aproximate_bayesian_information_criteria(2.0, 4, 50)
        bic_high = aproximate_bayesian_information_criteria(10.0, 4, 50)
        assert bic_high > bic_low

    def test_zero_neglogl_equals_penalty_term(self):
        n_params, n_data = 5, 20
        assert aproximate_bayesian_information_criteria(0.0, n_params, n_data) == pytest.approx(
            n_params * np.log(n_data)
        )


# ---------------------------------------------------------------------------
# build_output
# ---------------------------------------------------------------------------

class TestBuildOutput:
    """build_output receives z/a as (m/k, D) arrays and b as (k,) scalar array."""

    def _call(self, outlets, subjects, z, a, b, loss=0.0, bic=0.0):
        return build_output(outlets, subjects, z, a, b, loss=loss, bic=bic)

    def test_outlet_names_match(self):
        outlets = ["A", "B", "C"]
        subjects = ["X"]
        z = np.array([[1.0], [-0.5], [0.2]])   # (3, 1)
        a = np.array([[0.8]])                   # (1, 1)
        b = np.array([-0.3])                    # (1,)
        result = self._call(outlets, subjects, z, a, b)
        assert [o.outlet for o in result.outlets] == outlets

    def test_subject_names_match(self):
        outlets = ["A"]
        subjects = ["P", "Q"]
        z = np.array([[0.5]])                   # (1, 1)
        a = np.array([[1.0], [-1.0]])           # (2, 1)
        b = np.array([0.1, 0.2])                # (2,)
        result = self._call(outlets, subjects, z, a, b)
        assert [s.subject for s in result.subjects] == subjects

    def test_z_values_assigned_correctly(self):
        outlets = ["A", "B"]
        z = np.array([[2.5], [-1.1]])           # (2, 1)
        result = self._call(outlets, ["X"], z, np.array([[0.0]]), np.array([0.0]))
        assert result.outlets[0].z == pytest.approx([2.5])
        assert result.outlets[1].z == pytest.approx([-1.1])

    def test_a_b_values_assigned_correctly(self):
        subjects = ["X", "Y"]
        a = np.array([[0.7], [-0.3]])           # (2, 1)
        b = np.array([1.2, 0.4])                # (2,)
        result = self._call(["A"], subjects, np.array([[0.0]]), a, b)
        assert result.subjects[0].a == pytest.approx([0.7])
        assert result.subjects[0].b == pytest.approx(1.2)
        assert result.subjects[1].a == pytest.approx([-0.3])
        assert result.subjects[1].b == pytest.approx(0.4)

    def test_two_dimensional_output(self):
        outlets = ["A", "B"]
        z = np.array([[1.0, 0.5], [-0.5, 0.3]])   # (2, 2)
        a = np.array([[0.8, 0.2]])                  # (1, 2)
        b = np.array([-0.1])                        # (1,) scalar
        result = self._call(outlets, ["X"], z, a, b)
        assert result.outlets[0].z == pytest.approx([1.0, 0.5])
        assert result.outlets[1].z == pytest.approx([-0.5, 0.3])
        assert result.subjects[0].a == pytest.approx([0.8, 0.2])
        assert result.subjects[0].b == pytest.approx(-0.1)

    def test_b_is_scalar_float(self):
        result = self._call(["A"], ["X"], np.array([[0.0]]), np.array([[0.5]]), np.array([-0.5]))
        assert isinstance(result.subjects[0].b, float)

    def test_z_and_a_are_lists_of_python_floats(self):
        result = self._call(["A"], ["X"], np.array([[1.0]]), np.array([[0.5]]), np.array([-0.5]))
        assert isinstance(result.outlets[0].z, list)
        assert all(type(v) is float for v in result.outlets[0].z)
        assert isinstance(result.subjects[0].a, list)
        assert all(type(v) is float for v in result.subjects[0].a)

    def test_loss_is_stored_as_float(self):
        result = self._call(["A"], ["X"], np.array([[0.0]]), np.array([[0.0]]), np.array([0.0]), loss=7.5)
        assert result.loss == pytest.approx(7.5)
        assert isinstance(result.loss, float)

    def test_bic_is_stored_as_float(self):
        result = self._call(["A"], ["X"], np.array([[0.0]]), np.array([[0.0]]), np.array([0.0]), bic=12.3)
        assert result.bic == pytest.approx(12.3)
        assert isinstance(result.bic, float)

    def test_bic_independent_of_loss(self):
        r1 = self._call(["A"], ["X"], np.array([[0.0]]), np.array([[0.0]]), np.array([0.0]), loss=1.0, bic=5.0)
        r2 = self._call(["A"], ["X"], np.array([[0.0]]), np.array([[0.0]]), np.array([0.0]), loss=9.0, bic=5.0)
        assert r1.bic == r2.bic

    def test_returns_analysis_output_instance(self):
        result = self._call(["A"], ["X"], np.array([[0.0]]), np.array([[0.0]]), np.array([0.0]))
        assert isinstance(result, AnalysisOutput)


# ---------------------------------------------------------------------------
# log_likelihood / negative_log_likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    @pytest.fixture
    def small_matrix(self):
        """2 outlets × 1 subject × 3 sentiment types."""
        m = np.zeros((2, 1, 3), dtype=np.int64)
        m[0, 0, 2] = 10   # outlet 0, positive
        m[1, 0, 0] = 5    # outlet 1, negative
        return m

    def test_returns_finite_float(self, small_matrix):
        # m=2, k=1, n_dims=1 → (2+1)*1 + 1 = 4 params
        x = np.zeros(n_params(2, 1, 1))
        result = log_likelihood(x, small_matrix)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_negative_log_likelihood_is_negation(self, small_matrix):
        x = np.array([0.5, -0.5, 1.0, 0.2])   # 4 params for m=2, k=1, D=1
        assert negative_log_likelihood(x, small_matrix) == pytest.approx(
            -log_likelihood(x, small_matrix)
        )

    def test_l2_penalty_lowers_likelihood_for_large_params(self, small_matrix):
        x_moderate = np.array([1.0, -1.0, 0.5, 0.5])
        x_extreme  = np.array([4.9, -4.9, 4.9, 4.9])
        assert log_likelihood(x_moderate, small_matrix) > log_likelihood(x_extreme, small_matrix)

    def test_zero_counts_matrix(self):
        matrix = np.zeros((2, 2, 3), dtype=np.int64)
        # m=2, k=2, n_dims=1 → (2+2)*1 + 2 = 6 params
        x = np.zeros(n_params(2, 2, 1))
        assert np.isfinite(log_likelihood(x, matrix))

    def test_two_dimensional_returns_finite(self, small_matrix):
        # m=2, k=1, n_dims=2 → (2+1)*2 + 1 = 7 params
        x = np.zeros(n_params(2, 1, 2))
        assert np.isfinite(log_likelihood(x, small_matrix, n_dims=2))

    def test_two_dimensional_negation(self, small_matrix):
        x = np.random.default_rng(7).uniform(-1, 1, n_params(2, 1, 2))
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

    def _finite_diff(self, x, matrix, n_dims, eps=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (negative_log_likelihood(xp, matrix, n_dims) -
                       negative_log_likelihood(xm, matrix, n_dims)) / (2 * eps)
        return grad

    def test_gradient_matches_finite_diff_1d(self, small_matrix):
        # m=2, k=2, n_dims=1 → 6 params
        x = np.random.default_rng(42).uniform(-1, 1, n_params(2, 2, 1))
        np.testing.assert_allclose(
            grad_negative_log_likelihood(x, small_matrix, n_dims=1),
            self._finite_diff(x, small_matrix, n_dims=1),
            rtol=1e-4, atol=1e-6,
        )

    def test_gradient_matches_finite_diff_2d(self, small_matrix):
        # m=2, k=2, n_dims=2 → (2+2)*2 + 2 = 10 params
        x = np.random.default_rng(0).uniform(-1, 1, n_params(2, 2, 2))
        np.testing.assert_allclose(
            grad_negative_log_likelihood(x, small_matrix, n_dims=2),
            self._finite_diff(x, small_matrix, n_dims=2),
            rtol=1e-4, atol=1e-6,
        )

    def test_gradient_shape_1d(self, small_matrix):
        x = np.zeros(n_params(2, 2, 1))
        assert grad_negative_log_likelihood(x, small_matrix, n_dims=1).shape == x.shape

    def test_gradient_shape_2d(self, small_matrix):
        x = np.zeros(n_params(2, 2, 2))
        assert grad_negative_log_likelihood(x, small_matrix, n_dims=2).shape == x.shape


# ---------------------------------------------------------------------------
# run_analysis
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    """Tests for run_analysis with minimize patched out."""

    def _mock_solution(self, m: int, k: int, n_dims: int = 1, fun: float = 5.0) -> MagicMock:
        mock = MagicMock()
        mock.x = np.random.default_rng(42).uniform(-1, 1, n_params(m, k, n_dims))
        mock.fun = fun
        return mock

    @patch("src.service.minimize")
    def test_output_outlet_count(self, mock_min):
        data = [
            make_mention("A", "X", "positive", 5),
            make_mention("B", "X", "negative", 3),
            make_mention("C", "X", "neutral",  2),
        ]
        mock_min.return_value = self._mock_solution(3, 1)
        assert len(run_analysis(data).outlets) == 3

    @patch("src.service.minimize")
    def test_output_subject_count(self, mock_min):
        data = [
            make_mention("A", "X", "positive", 5),
            make_mention("A", "Y", "negative", 3),
            make_mention("A", "Z", "neutral",  2),
        ]
        mock_min.return_value = self._mock_solution(1, 3)
        assert len(run_analysis(data).subjects) == 3

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
        data = [
            make_mention("A", "X", "positive", 8),
            make_mention("B", "Y", "negative", 4),
        ]
        # m=2, k=2, n_dims=1 → 6 params
        mock = MagicMock()
        mock.x = np.random.default_rng(0).uniform(-5, 5, n_params(2, 2, 1))
        mock_min.return_value = mock
        result = run_analysis(data)
        for o in result.outlets:
            assert all(-5.0 <= v <= 5.0 for v in o.z)
        for s in result.subjects:
            assert all(-5.0 <= v <= 5.0 for v in s.a)
            assert -5.0 <= s.b <= 5.0

    @patch("src.service.minimize")
    def test_minimize_called_once(self, mock_min):
        mock_min.return_value = self._mock_solution(1, 1)
        run_analysis([make_mention("A", "X", "positive", 1)])
        mock_min.assert_called_once()

    @patch("src.service.minimize")
    def test_returns_analysis_output(self, mock_min):
        mock_min.return_value = self._mock_solution(1, 1)
        assert isinstance(run_analysis([make_mention("A", "X", "positive", 1)]), AnalysisOutput)

    @patch("src.service.minimize")
    def test_z_is_list_a_is_list_b_is_float(self, mock_min):
        mock_min.return_value = self._mock_solution(1, 1)
        result = run_analysis([make_mention("A", "X", "positive", 1)])
        assert isinstance(result.outlets[0].z, list)
        assert isinstance(result.subjects[0].a, list)
        assert isinstance(result.subjects[0].b, float)

    @patch("src.service.minimize")
    def test_two_dimensions_param_count(self, mock_min):
        """With n_dims=2, minimize receives (1+1)*2 + 1 = 5 parameters."""
        mock_min.return_value = self._mock_solution(1, 1, n_dims=2)
        run_analysis([make_mention("A", "X", "positive", 1)], n_dims=2)
        _, call_kwargs = mock_min.call_args
        assert len(call_kwargs["x0"]) == n_params(1, 1, 2)

    @patch("src.service.minimize")
    def test_two_dimensions_output_lengths(self, mock_min):
        """With n_dims=2, z and a have length 2; b is still a scalar."""
        mock_min.return_value = self._mock_solution(1, 1, n_dims=2)
        result = run_analysis([make_mention("A", "X", "positive", 1)], n_dims=2)
        assert len(result.outlets[0].z) == 2
        assert len(result.subjects[0].a) == 2
        assert isinstance(result.subjects[0].b, float)

    @patch("src.service.minimize")
    def test_loss_comes_from_solution_fun(self, mock_min):
        mock_min.return_value = self._mock_solution(1, 1, fun=42.0)
        result = run_analysis([make_mention("A", "X", "positive", 1)])
        assert result.loss == pytest.approx(42.0)
        assert isinstance(result.loss, float)

    @patch("src.service.minimize")
    def test_bic_is_float_in_output(self, mock_min):
        mock_min.return_value = self._mock_solution(1, 1, fun=5.0)
        result = run_analysis([make_mention("A", "X", "positive", 10)])
        assert isinstance(result.bic, float)
        assert np.isfinite(result.bic)

    @patch("src.service.minimize")
    def test_bic_matches_formula(self, mock_min):
        """BIC = 2*loss + n_params*ln(n_mentions)."""
        data = [
            make_mention("A", "X", "positive", 10),
            make_mention("A", "X", "negative", 5),
        ]
        # m=1, k=1, n_dims=1 → n_params=3; n_mentions=15
        mock_min.return_value = self._mock_solution(1, 1, fun=7.0)
        result = run_analysis(data)
        expected_bic = 2 * 7.0 + 3 * np.log(15)
        assert result.bic == pytest.approx(expected_bic)

    @patch("src.service.minimize")
    def test_bic_larger_with_more_dimensions(self, mock_min):
        """For same data and loss, 2D model has more parameters so BIC is larger."""
        data = [
            make_mention("A", "X", "positive", 8),
            make_mention("B", "Y", "negative", 4),
        ]
        # m=2, k=2: n_params_1d=(2+2)*1+2=6, n_params_2d=(2+2)*2+2=10
        mock_min.return_value = self._mock_solution(2, 2, n_dims=1, fun=5.0)
        result_1d = run_analysis(data, n_dims=1)

        mock_min.return_value = self._mock_solution(2, 2, n_dims=2, fun=5.0)
        result_2d = run_analysis(data, n_dims=2)

        assert result_2d.bic > result_1d.bic

    @patch("src.service.minimize")
    def test_bic_increases_with_higher_loss(self, mock_min):
        data = [make_mention("A", "X", "positive", 10)]
        mock_min.return_value = self._mock_solution(1, 1, fun=3.0)
        result_low = run_analysis(data)

        mock_min.return_value = self._mock_solution(1, 1, fun=20.0)
        result_high = run_analysis(data)

        assert result_high.bic > result_low.bic


# ---------------------------------------------------------------------------
# generate_mentions
# ---------------------------------------------------------------------------

OUTLETS_1D = [
    OutletScore(outlet="A", z=[1.0]),
    OutletScore(outlet="B", z=[-1.0]),
]

SUBJECTS_1D = [
    SubjectScore(subject="X", a=[0.8], b=0.2),
    SubjectScore(subject="Y", a=[-0.5], b=-0.1),
]

OUTLETS_2D = [
    OutletScore(outlet="A", z=[1.0, 0.5]),
    OutletScore(outlet="B", z=[-1.0, -0.5]),
]

SUBJECTS_2D = [
    SubjectScore(subject="X", a=[0.8, 0.3], b=0.2),
    SubjectScore(subject="Y", a=[-0.5, 0.1], b=-0.1),
]


class TestGenerateMentions:
    def test_output_shape(self):
        assert generate_mentions(q=0.0, n=100).shape == (3,)

    def test_counts_sum_to_n(self):
        for q in [-2.0, 0.0, 2.0]:
            assert generate_mentions(q=q, n=200).sum() == 200

    def test_all_counts_non_negative(self):
        assert (generate_mentions(q=0.5, n=100) >= 0).all()

    def test_positive_q_biases_toward_positive_mentions(self):
        """With large positive q, positive count should dominate negative."""
        result = generate_mentions(q=5.0, n=500)
        assert result[2] > result[0]

    def test_negative_q_biases_toward_negative_mentions(self):
        """With large negative q, negative count should dominate positive."""
        result = generate_mentions(q=-5.0, n=500)
        assert result[0] > result[2]

    def test_zero_n_returns_all_zeros(self):
        result = generate_mentions(q=1.0, n=0)
        assert result.sum() == 0
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# generate_data
# ---------------------------------------------------------------------------

class TestGenerateData:
    def test_returns_analysis_input(self):
        assert isinstance(generate_data(OUTLETS_1D, SUBJECTS_1D, amount_of_mentions=10), AnalysisInput)

    def test_n_dimensions_matches_z_length_1d(self):
        result = generate_data(OUTLETS_1D, SUBJECTS_1D, amount_of_mentions=10)
        assert result.n_dimensions == 1

    def test_n_dimensions_matches_z_length_2d(self):
        result = generate_data(OUTLETS_2D, SUBJECTS_2D, amount_of_mentions=10)
        assert result.n_dimensions == 2

    def test_data_row_count(self):
        """n_outlets × n_subjects × 3 sentiment types."""
        result = generate_data(OUTLETS_1D, SUBJECTS_1D, amount_of_mentions=10)
        assert len(result.data) == len(OUTLETS_1D) * len(SUBJECTS_1D) * 3

    def test_all_mention_types_present(self):
        result = generate_data(OUTLETS_1D, SUBJECTS_1D, amount_of_mentions=10)
        assert {m.mention_type for m in result.data} == {"negative", "neutral", "positive"}

    def test_all_outlet_names_appear(self):
        result = generate_data(OUTLETS_1D, SUBJECTS_1D, amount_of_mentions=10)
        assert {m.outlet for m in result.data} == {o.outlet for o in OUTLETS_1D}

    def test_all_subject_names_appear(self):
        result = generate_data(OUTLETS_1D, SUBJECTS_1D, amount_of_mentions=10)
        assert {m.subject for m in result.data} == {s.subject for s in SUBJECTS_1D}

    def test_per_pair_totals_equal_amount_of_mentions(self):
        from collections import defaultdict
        n = 50
        result = generate_data(OUTLETS_1D, SUBJECTS_1D, amount_of_mentions=n)
        totals: dict = defaultdict(int)
        for m in result.data:
            totals[(m.outlet, m.subject)] += m.amount_of_mentions
        for outlet in OUTLETS_1D:
            for subject in SUBJECTS_1D:
                assert totals[(outlet.outlet, subject.subject)] == n

    def test_all_amounts_non_negative(self):
        result = generate_data(OUTLETS_1D, SUBJECTS_1D, amount_of_mentions=20)
        assert all(m.amount_of_mentions >= 0 for m in result.data)

    def test_zero_amount_of_mentions(self):
        result = generate_data(OUTLETS_1D, SUBJECTS_1D, amount_of_mentions=0)
        assert all(m.amount_of_mentions == 0 for m in result.data)

    def test_single_outlet_single_subject(self):
        outlets = [OutletScore(outlet="Solo", z=[0.0])]
        subjects = [SubjectScore(subject="Topic", a=[1.0], b=0.0)]
        result = generate_data(outlets, subjects, amount_of_mentions=10)
        assert len(result.data) == 3
        assert {m.outlet for m in result.data} == {"Solo"}
        assert {m.subject for m in result.data} == {"Topic"}
