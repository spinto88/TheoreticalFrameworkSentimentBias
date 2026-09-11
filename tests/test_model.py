"""
Unit tests for sentiment_bias.model — the pure-numpy core.

Covers:
- pack / unpack / log_odds : parameter algebra round-trips and shapes
- log_likelihood / negative_log_likelihood : return types and relationship
- grad_negative_log_likelihood : finite-difference verification
- bic : the formula 2*neglogl + n_params*ln(n_data)
- canonicalize_1d : standardization and q-invariance

Parameter vector layout: x = [z (m*D), a (k*D), b (k)]
Total length: (m + k) * D + k
"""

import numpy as np
import pytest

from sentiment_bias.model import (
    bic,
    canonicalize_1d,
    grad_negative_log_likelihood,
    log_likelihood,
    log_odds,
    negative_log_likelihood,
    pack,
    unpack,
)


def n_params(m: int, k: int, n_dimensions: int = 1) -> int:
    """Total parameter count for the flat optimisation vector."""
    return (m + k) * n_dimensions + k


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------

class TestPackUnpack:
    def test_pack_length_1d(self):
        z = np.array([[1.0], [2.0], [3.0]])   # (3, 1)
        a = np.array([[0.5], [-0.5]])          # (2, 1)
        b = np.array([0.1, 0.2])               # (2,)
        assert pack(z, a, b).shape == (n_params(3, 2, 1),)

    def test_pack_length_2d(self):
        z = np.zeros((3, 2))
        a = np.zeros((2, 2))
        b = np.zeros(2)
        assert pack(z, a, b).shape == (n_params(3, 2, 2),)

    def test_pack_layout_is_z_then_a_then_b(self):
        z = np.array([[1.0], [2.0]])
        a = np.array([[3.0]])
        b = np.array([4.0])
        np.testing.assert_allclose(pack(z, a, b), [1.0, 2.0, 3.0, 4.0])

    def test_unpack_shapes_1d(self):
        x = np.arange(n_params(3, 2, 1), dtype=float)
        z, a, b = unpack(x, 3, 2, 1)
        assert z.shape == (3, 1)
        assert a.shape == (2, 1)
        assert b.shape == (2,)

    def test_unpack_shapes_2d(self):
        x = np.arange(n_params(3, 2, 2), dtype=float)
        z, a, b = unpack(x, 3, 2, 2)
        assert z.shape == (3, 2)
        assert a.shape == (2, 2)
        assert b.shape == (2,)

    def test_unpack_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])   # m=2, k=1, D=1
        z, a, b = unpack(x, 2, 1, 1)
        np.testing.assert_allclose(z, [[1.0], [2.0]])
        np.testing.assert_allclose(a, [[3.0]])
        np.testing.assert_allclose(b, [4.0])

    def test_round_trip_1d(self):
        x = np.random.default_rng(0).normal(size=n_params(4, 3, 1))
        np.testing.assert_allclose(pack(*unpack(x, 4, 3, 1)), x)

    def test_round_trip_2d(self):
        x = np.random.default_rng(1).normal(size=n_params(4, 3, 2))
        np.testing.assert_allclose(pack(*unpack(x, 4, 3, 2)), x)

    def test_unpack_default_is_one_dimension(self):
        x = np.arange(n_params(2, 1, 1), dtype=float)
        assert unpack(x, 2, 1)[0].shape == (2, 1)


# ---------------------------------------------------------------------------
# log_odds
# ---------------------------------------------------------------------------

class TestLogOdds:
    def test_shape(self):
        z = np.zeros((4, 2))
        a = np.zeros((3, 2))
        b = np.zeros(3)
        assert log_odds(z, a, b).shape == (4, 3)

    def test_values_1d(self):
        z = np.array([[2.0], [-1.0]])
        a = np.array([[3.0], [0.5]])
        b = np.array([1.0, -1.0])
        # q[0,0] = 2*3 + 1 = 7 ; q[0,1] = 2*0.5 - 1 = 0
        # q[1,0] = -1*3 + 1 = -2 ; q[1,1] = -1*0.5 - 1 = -1.5
        np.testing.assert_allclose(log_odds(z, a, b), [[7.0, 0.0], [-2.0, -1.5]])

    def test_values_2d_sums_over_dimensions(self):
        z = np.array([[1.0, 2.0]])
        a = np.array([[3.0, 4.0]])
        b = np.array([0.5])
        # 1*3 + 2*4 + 0.5 = 11.5
        np.testing.assert_allclose(log_odds(z, a, b), [[11.5]])

    def test_baseline_shared_across_dimensions(self):
        z = np.zeros((2, 2))
        a = np.zeros((2, 2))
        b = np.array([0.7, -0.3])
        np.testing.assert_allclose(log_odds(z, a, b), [[0.7, -0.3], [0.7, -0.3]])


# ---------------------------------------------------------------------------
# bic
# ---------------------------------------------------------------------------

class TestBIC:
    """Unit tests for the BIC formula: 2*neglogl + n_params*ln(n_data)."""

    def test_known_value_zero_penalty(self):
        # ln(1) == 0, so the penalty term vanishes
        assert bic(5.0, 4, 1) == pytest.approx(10.0)

    def test_known_value_explicit(self):
        expected = 2 * 7.0 + 3 * np.log(100)
        assert bic(7.0, 3, 100) == pytest.approx(expected)

    def test_returns_float(self):
        assert isinstance(bic(1.0, 2, 10), float)

    def test_result_is_finite(self):
        assert np.isfinite(bic(5.0, 6, 50))

    def test_increases_with_more_parameters(self):
        assert bic(5.0, 10, 100) > bic(5.0, 3, 100)

    def test_increases_with_higher_neglogl(self):
        assert bic(10.0, 4, 50) > bic(2.0, 4, 50)

    def test_zero_neglogl_equals_penalty_term(self):
        assert bic(0.0, 5, 20) == pytest.approx(5 * np.log(20))


# ---------------------------------------------------------------------------
# log_likelihood / negative_log_likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    @pytest.fixture
    def small_counts(self):
        """2 outlets x 1 subject x 3 sentiment types."""
        counts = np.zeros((2, 1, 3), dtype=np.int64)
        counts[0, 0, 2] = 10   # outlet 0, positive
        counts[1, 0, 0] = 5    # outlet 1, negative
        return counts

    def test_returns_finite_float(self, small_counts):
        x = np.zeros(n_params(2, 1, 1))
        result = log_likelihood(x, small_counts)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_negative_log_likelihood_is_negation(self, small_counts):
        x = np.array([0.5, -0.5, 1.0, 0.2])   # 4 params for m=2, k=1, D=1
        assert negative_log_likelihood(x, small_counts) == pytest.approx(
            -log_likelihood(x, small_counts)
        )

    def test_l2_penalty_lowers_likelihood_for_large_params(self, small_counts):
        x_moderate = np.array([1.0, -1.0, 0.5, 0.5])
        x_extreme = np.array([4.9, -4.9, 4.9, 4.9])
        assert log_likelihood(x_moderate, small_counts) > log_likelihood(x_extreme, small_counts)

    def test_zero_counts_matrix(self):
        counts = np.zeros((2, 2, 3), dtype=np.int64)
        x = np.zeros(n_params(2, 2, 1))
        assert np.isfinite(log_likelihood(x, counts))

    def test_two_dimensional_returns_finite(self, small_counts):
        x = np.zeros(n_params(2, 1, 2))
        assert np.isfinite(log_likelihood(x, small_counts, n_dimensions=2))

    def test_two_dimensional_negation(self, small_counts):
        x = np.random.default_rng(7).uniform(-1, 1, n_params(2, 1, 2))
        assert negative_log_likelihood(x, small_counts, n_dimensions=2) == pytest.approx(
            -log_likelihood(x, small_counts, n_dimensions=2)
        )

    def test_zero_parameters_give_pure_penalty_free_value(self):
        """At x = 0 every q_ij is 0, so logl = -N * ln(3) with no penalty."""
        counts = np.zeros((1, 1, 3), dtype=np.int64)
        counts[0, 0, 1] = 9   # 9 neutral mentions
        x = np.zeros(n_params(1, 1, 1))
        assert log_likelihood(x, counts) == pytest.approx(-9 * np.log(3.0))


# ---------------------------------------------------------------------------
# grad_negative_log_likelihood
# ---------------------------------------------------------------------------

class TestGradNegativeLogLikelihood:
    """Verify the analytical gradient against finite differences."""

    @pytest.fixture
    def small_counts(self):
        counts = np.zeros((2, 2, 3), dtype=np.int64)
        counts[0, 0, 2] = 8
        counts[0, 1, 0] = 3
        counts[1, 0, 1] = 5
        counts[1, 1, 2] = 2
        return counts

    def _finite_diff(self, x, counts, n_dimensions, eps=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (negative_log_likelihood(xp, counts, n_dimensions) -
                       negative_log_likelihood(xm, counts, n_dimensions)) / (2 * eps)
        return grad

    def test_gradient_matches_finite_diff_1d(self, small_counts):
        x = np.random.default_rng(42).uniform(-1, 1, n_params(2, 2, 1))
        np.testing.assert_allclose(
            grad_negative_log_likelihood(x, small_counts, n_dimensions=1),
            self._finite_diff(x, small_counts, n_dimensions=1),
            rtol=1e-4, atol=1e-6,
        )

    def test_gradient_matches_finite_diff_2d(self, small_counts):
        x = np.random.default_rng(0).uniform(-1, 1, n_params(2, 2, 2))
        np.testing.assert_allclose(
            grad_negative_log_likelihood(x, small_counts, n_dimensions=2),
            self._finite_diff(x, small_counts, n_dimensions=2),
            rtol=1e-4, atol=1e-6,
        )

    def test_gradient_shape_1d(self, small_counts):
        x = np.zeros(n_params(2, 2, 1))
        assert grad_negative_log_likelihood(x, small_counts, n_dimensions=1).shape == x.shape

    def test_gradient_shape_2d(self, small_counts):
        x = np.zeros(n_params(2, 2, 2))
        assert grad_negative_log_likelihood(x, small_counts, n_dimensions=2).shape == x.shape


# ---------------------------------------------------------------------------
# canonicalize_1d
# ---------------------------------------------------------------------------

class TestCanonicalize1d:
    """z/a/b are only identified up to z -> (z-c)/s, a -> a*s, b -> b + c*a."""

    def _q(self, z, a, b):
        """q_ij = dot(z_i, a_j) + b_j for every outlet/subject pair."""
        return z @ a.T + b[np.newaxis, :]

    def test_z_has_zero_sum_and_unit_norm(self):
        z = np.array([[3.0], [1.0], [-2.0]])
        a = np.array([[0.6], [-1.4]])
        b = np.array([0.3, -0.1])
        z_new, _, _ = canonicalize_1d(z, a, b)
        assert z_new.sum() == pytest.approx(0.0, abs=1e-12)
        assert (z_new ** 2).sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("m", [2, 3, 10, 50])
    def test_unit_norm_holds_for_any_number_of_outlets(self, m):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(m, 1))
        a = np.array([[0.6], [-1.4]])
        b = np.array([0.3, -0.1])
        z_new, _, _ = canonicalize_1d(z, a, b)
        assert (z_new ** 2).sum() == pytest.approx(1.0)
        # Unit norm, not unit variance: the two differ by sqrt(m).
        assert z_new.var() == pytest.approx(1.0 / m)

    def test_a_absorbs_the_full_scale(self):
        z = np.array([[3.0], [1.0], [-2.0]])
        a = np.array([[0.6], [-1.4]])
        b = np.array([0.3, -0.1])
        scale = np.sqrt(((z[:, 0] - z[:, 0].mean()) ** 2).sum())
        _, a_new, _ = canonicalize_1d(z, a, b)
        np.testing.assert_allclose(np.abs(a_new[:, 0]), np.abs(a[:, 0] * scale))

    def test_q_is_invariant(self):
        z = np.array([[3.0], [1.0], [-2.0]])
        a = np.array([[0.6], [-1.4]])
        b = np.array([0.3, -0.1])
        q_before = self._q(z, a, b)
        z_new, a_new, b_new = canonicalize_1d(z, a, b)
        assert self._q(z_new, a_new, b_new) == pytest.approx(q_before)

    def test_default_reference_is_outlet_zero_positive(self):
        z = np.array([[-3.0], [1.0], [2.0]])
        a = np.array([[0.6]])
        b = np.array([0.3])
        z_new, _, _ = canonicalize_1d(z, a, b)
        assert z_new[0, 0] > 0

    def test_sign_reference_negative(self):
        z = np.array([[-3.0], [1.0], [2.0]])
        a = np.array([[0.6]])
        b = np.array([0.3])
        z_new, _, _ = canonicalize_1d(z, a, b, sign_reference_idx=0, sign_reference_positive=False)
        assert z_new[0, 0] < 0

    def test_sign_reference_selects_other_outlet(self):
        z = np.array([[-3.0], [1.0], [2.0]])
        a = np.array([[0.6]])
        b = np.array([0.3])
        z_new, _, _ = canonicalize_1d(z, a, b, sign_reference_idx=2, sign_reference_positive=False)
        assert z_new[2, 0] < 0

    def test_degenerate_single_outlet_returns_unchanged(self):
        z = np.array([[2.0]])
        a = np.array([[0.6], [-1.4]])
        b = np.array([0.3, -0.1])
        z_new, a_new, b_new = canonicalize_1d(z, a, b)
        assert z_new is z
        assert a_new is a
        assert b_new is b

    def test_degenerate_identical_outlets_returns_unchanged(self):
        z = np.array([[2.0], [2.0]])
        a = np.array([[0.6]])
        b = np.array([0.3])
        z_new, a_new, b_new = canonicalize_1d(z, a, b)
        assert z_new is z
        assert a_new is a
        assert b_new is b

    def test_output_shapes_preserved(self):
        z = np.array([[3.0], [1.0], [-2.0]])
        a = np.array([[0.6], [-1.4]])
        b = np.array([0.3, -0.1])
        z_new, a_new, b_new = canonicalize_1d(z, a, b)
        assert z_new.shape == (3, 1)
        assert a_new.shape == (2, 1)
        assert b_new.shape == (2,)
