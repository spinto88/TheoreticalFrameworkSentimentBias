"""
Unit tests for sentiment_bias.simulate — the generative direction.

Covers the three ways of choosing N per pair (fixed int, Poisson draw, or an
explicit (m, k) array), reproducibility under `seed`, and `thin`.

The end-to-end recovery test at the bottom is the only test in the suite that
really runs the optimiser; it is marked `slow`.
"""

import numpy as np
import pytest

from sentiment_bias import fit, simulate, thin
from sentiment_bias.data import MentionTensor, from_array

Z_1D = np.array([1.0, -1.0])
A_1D = np.array([0.8, -0.5])
B_1D = np.array([0.2, -0.1])

Z_2D = np.array([[1.0, 0.5], [-1.0, -0.5]])
A_2D = np.array([[0.8, 0.3], [-0.5, 0.1]])


# ---------------------------------------------------------------------------
# Shape, names and basic invariants
# ---------------------------------------------------------------------------

class TestStructure:
    def test_returns_mention_tensor(self):
        assert isinstance(simulate(Z_1D, A_1D, B_1D, n_mentions=10, seed=0), MentionTensor)

    def test_shape_is_outlets_by_subjects_by_three(self):
        assert simulate(Z_1D, A_1D, B_1D, n_mentions=10, seed=0).shape == (2, 2, 3)

    def test_generated_names(self):
        tensor = simulate(Z_1D, A_1D, B_1D, n_mentions=10, seed=0)
        assert tensor.outlets == ["outlet_00", "outlet_01"]
        assert tensor.subjects == ["subject_00", "subject_01"]

    def test_explicit_names(self):
        tensor = simulate(
            Z_1D, A_1D, B_1D, n_mentions=10, seed=0,
            outlets=["CNN", "Fox"], subjects=["Economy", "Politics"],
        )
        assert tensor.outlets == ["CNN", "Fox"]
        assert tensor.subjects == ["Economy", "Politics"]

    def test_counts_are_non_negative(self):
        counts = simulate(Z_1D, A_1D, B_1D, n_mentions=50, seed=0).counts
        assert (counts >= 0).all()

    def test_counts_are_integers(self):
        assert simulate(Z_1D, A_1D, B_1D, n_mentions=10, seed=0).counts.dtype == np.int64

    def test_one_dimensional_input_is_promoted(self):
        """z/a given as (m,)/(k,) vectors mean D=1."""
        tensor = simulate(Z_1D, A_1D, B_1D, n_mentions=10, seed=0)
        assert tensor.shape == (2, 2, 3)

    def test_two_dimensional_input(self):
        tensor = simulate(Z_2D, A_2D, B_1D, n_mentions=10, seed=0)
        assert tensor.shape == (2, 2, 3)

    def test_single_outlet_single_subject(self):
        tensor = simulate(np.array([0.0]), np.array([1.0]), np.array([0.0]),
                          n_mentions=10, seed=0)
        assert tensor.shape == (1, 1, 3)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            simulate(Z_2D, A_1D, B_1D, n_mentions=10, seed=0)

    def test_wrong_b_length_raises(self):
        with pytest.raises(ValueError, match=r"b must have shape"):
            simulate(Z_1D, A_1D, np.array([0.0, 0.0, 0.0]), n_mentions=10, seed=0)


# ---------------------------------------------------------------------------
# n_mentions as a fixed int
# ---------------------------------------------------------------------------

class TestFixedN:
    def test_every_pair_totals_exactly_n(self):
        counts = simulate(Z_1D, A_1D, B_1D, n_mentions=37, seed=0).counts
        np.testing.assert_array_equal(counts.sum(axis=2), np.full((2, 2), 37))

    def test_total_is_n_times_pairs(self):
        tensor = simulate(Z_1D, A_1D, B_1D, n_mentions=25, seed=0)
        assert tensor.n_mentions == 25 * 2 * 2

    def test_default_n_is_one_hundred(self):
        counts = simulate(Z_1D, A_1D, B_1D, seed=0).counts
        np.testing.assert_array_equal(counts.sum(axis=2), np.full((2, 2), 100))

    def test_zero_mentions_gives_all_zeros(self):
        tensor = simulate(Z_1D, A_1D, B_1D, n_mentions=0, seed=0)
        assert tensor.n_mentions == 0
        assert tensor.shape == (2, 2, 3)

    def test_positive_q_favours_positive_mentions(self):
        """A large positive q_ij should put more mass on positive than negative."""
        tensor = simulate(np.array([5.0]), np.array([1.0]), np.array([0.0]),
                          n_mentions=2000, seed=0)
        negative, _, positive = tensor.counts[0, 0]
        assert positive > negative

    def test_negative_q_favours_negative_mentions(self):
        tensor = simulate(np.array([-5.0]), np.array([1.0]), np.array([0.0]),
                          n_mentions=2000, seed=0)
        negative, _, positive = tensor.counts[0, 0]
        assert negative > positive

    def test_zero_q_is_symmetric_on_average(self):
        tensor = simulate(np.array([0.0]), np.array([0.0]), np.array([0.0]),
                          n_mentions=20000, seed=0)
        negative, neutral, positive = tensor.counts[0, 0]
        # all three categories are equally likely at q = 0
        assert abs(int(negative) - int(positive)) < 0.05 * 20000
        assert abs(int(neutral) - int(positive)) < 0.05 * 20000


# ---------------------------------------------------------------------------
# n_mentions="poisson"
# ---------------------------------------------------------------------------

class TestPoissonN:
    def test_totals_vary_between_pairs(self):
        counts = simulate(
            np.linspace(-2, 2, 20), A_1D, B_1D,
            n_mentions="poisson", mean_mentions=100, seed=0,
        ).counts
        assert counts.sum(axis=2).std() > 0

    def test_mean_total_is_near_mu(self):
        counts = simulate(
            np.linspace(-2, 2, 50), np.linspace(-1, 1, 6), np.zeros(6),
            n_mentions="poisson", mean_mentions=100, seed=123457,
        ).counts
        assert counts.sum(axis=2).mean() == pytest.approx(100, rel=0.05)

    def test_reproducible_with_same_seed(self):
        first = simulate(Z_1D, A_1D, B_1D, n_mentions="poisson", mean_mentions=50, seed=7)
        second = simulate(Z_1D, A_1D, B_1D, n_mentions="poisson", mean_mentions=50, seed=7)
        np.testing.assert_array_equal(first.counts, second.counts)

    def test_different_seeds_differ(self):
        first = simulate(Z_1D, A_1D, B_1D, n_mentions="poisson", mean_mentions=50, seed=1)
        second = simulate(Z_1D, A_1D, B_1D, n_mentions="poisson", mean_mentions=50, seed=2)
        assert not np.array_equal(first.counts, second.counts)

    def test_counts_are_non_negative(self):
        counts = simulate(
            Z_1D, A_1D, B_1D, n_mentions="poisson", mean_mentions=20, seed=0
        ).counts
        assert (counts >= 0).all()

    def test_missing_mean_mentions_raises(self):
        with pytest.raises(ValueError, match="requires mean_mentions"):
            simulate(Z_1D, A_1D, B_1D, n_mentions="poisson", seed=0)

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="n_mentions must be"):
            simulate(Z_1D, A_1D, B_1D, n_mentions="binomial", mean_mentions=10, seed=0)


# ---------------------------------------------------------------------------
# n_mentions as an explicit (m, k) array
# ---------------------------------------------------------------------------

class TestArrayN:
    def test_respected_cell_by_cell(self):
        sizes = np.array([[7, 13], [0, 42]])
        counts = simulate(Z_1D, A_1D, B_1D, n_mentions=sizes, seed=0).counts
        np.testing.assert_array_equal(counts.sum(axis=2), sizes)

    def test_total_matches_array_sum(self):
        sizes = np.array([[7, 13], [1, 42]])
        tensor = simulate(Z_1D, A_1D, B_1D, n_mentions=sizes, seed=0)
        assert tensor.n_mentions == int(sizes.sum())

    def test_accepts_nested_lists(self):
        counts = simulate(Z_1D, A_1D, B_1D, n_mentions=[[5, 5], [5, 5]], seed=0).counts
        np.testing.assert_array_equal(counts.sum(axis=2), np.full((2, 2), 5))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"must have shape \(2, 2\)"):
            simulate(Z_1D, A_1D, B_1D, n_mentions=np.array([[1, 2, 3]]), seed=0)


# ---------------------------------------------------------------------------
# Contradictory arguments
# ---------------------------------------------------------------------------

class TestArgumentValidation:
    def test_int_with_mean_mentions_raises(self):
        with pytest.raises(ValueError, match="only valid with"):
            simulate(Z_1D, A_1D, B_1D, n_mentions=100, mean_mentions=50, seed=0)

    def test_array_with_mean_mentions_raises(self):
        with pytest.raises(ValueError, match="only valid with"):
            simulate(
                Z_1D, A_1D, B_1D,
                n_mentions=np.array([[1, 1], [1, 1]]), mean_mentions=50, seed=0,
            )


# ---------------------------------------------------------------------------
# Reproducibility of the fixed-N path
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_counts(self):
        first = simulate(Z_1D, A_1D, B_1D, n_mentions=100, seed=99)
        second = simulate(Z_1D, A_1D, B_1D, n_mentions=100, seed=99)
        np.testing.assert_array_equal(first.counts, second.counts)

    def test_generator_accepted_as_seed(self):
        rng = np.random.default_rng(5)
        tensor = simulate(Z_1D, A_1D, B_1D, n_mentions=10, seed=rng)
        assert tensor.n_mentions == 40

    def test_no_seed_still_works(self):
        assert simulate(Z_1D, A_1D, B_1D, n_mentions=10).n_mentions == 40


# ---------------------------------------------------------------------------
# thin
# ---------------------------------------------------------------------------

class TestThin:
    @pytest.fixture
    def tensor(self):
        return simulate(
            np.linspace(-2, 2, 10), np.linspace(-1, 1, 4), np.zeros(4),
            n_mentions=100, seed=0,
        )

    def test_total_is_exactly_the_requested_fraction(self, tensor):
        reduced = thin(tensor, 0.5, seed=0)
        assert reduced.n_mentions == round(0.5 * tensor.n_mentions)

    def test_total_for_other_fractions(self, tensor):
        for fraction in [0.1, 0.25, 0.37, 0.9]:
            reduced = thin(tensor, fraction, seed=0)
            assert reduced.n_mentions == round(fraction * tensor.n_mentions)

    def test_no_cell_exceeds_the_original(self, tensor):
        reduced = thin(tensor, 0.6, seed=0)
        assert (reduced.counts <= tensor.counts).all()

    def test_counts_are_non_negative(self, tensor):
        assert (thin(tensor, 0.3, seed=0).counts >= 0).all()

    def test_fraction_one_is_identity(self, tensor):
        np.testing.assert_array_equal(thin(tensor, 1.0, seed=0).counts, tensor.counts)

    def test_shape_and_names_preserved(self, tensor):
        reduced = thin(tensor, 0.5, seed=0)
        assert reduced.shape == tensor.shape
        assert reduced.outlets == tensor.outlets
        assert reduced.subjects == tensor.subjects

    def test_returns_a_new_tensor(self, tensor):
        reduced = thin(tensor, 0.5, seed=0)
        assert reduced is not tensor
        assert tensor.n_mentions == 100 * 10 * 4   # original untouched

    def test_reproducible_with_same_seed(self, tensor):
        first = thin(tensor, 0.5, seed=3)
        second = thin(tensor, 0.5, seed=3)
        np.testing.assert_array_equal(first.counts, second.counts)

    def test_different_seeds_differ(self, tensor):
        first = thin(tensor, 0.5, seed=1)
        second = thin(tensor, 0.5, seed=2)
        assert not np.array_equal(first.counts, second.counts)

    def test_dtype_is_int64(self, tensor):
        assert thin(tensor, 0.5, seed=0).counts.dtype == np.int64

    def test_zero_fraction_raises(self, tensor):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            thin(tensor, 0.0)

    def test_negative_fraction_raises(self, tensor):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            thin(tensor, -0.5)

    def test_fraction_above_one_raises(self, tensor):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            thin(tensor, 1.5)

    def test_all_zero_tensor(self):
        empty = from_array(np.zeros((2, 2, 3), dtype=np.int64))
        assert thin(empty, 0.5, seed=0).n_mentions == 0


# ---------------------------------------------------------------------------
# End-to-end recovery (really runs the optimiser)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRecovery:
    def test_recovers_known_parameters(self):
        """simulate with a known z, fit, and check the ranking is recovered.

        This mirrors the synthetic-data validation behind the paper: the sign
        and scale of z are not identified, but its ordering is, so Spearman
        correlation is the right measure.
        """
        from scipy.stats import spearmanr

        z = np.linspace(-2, 2, 30)
        a = np.linspace(-1, 1, 5)
        b = np.random.default_rng(0).normal(0, 0.05, 5)

        tensor = simulate(z, a, b, n_mentions="poisson", mean_mentions=100, seed=123457)
        result = fit(tensor, n_dimensions=1, n_restarts=3, seed=1)

        assert abs(spearmanr(z, result.z["dim_1"]).statistic) > 0.9
        assert abs(spearmanr(a, result.a["dim_1"]).statistic) > 0.9

    def test_recovers_log_odds(self):
        """q_ij is identified even though z and a individually are not."""
        from scipy.stats import spearmanr

        from sentiment_bias.model import log_odds

        z = np.linspace(-2, 2, 20)
        a = np.linspace(-1, 1, 4)
        b = np.zeros(4)

        tensor = simulate(z, a, b, n_mentions=200, seed=42)
        result = fit(tensor, n_dimensions=1, n_restarts=3, seed=1)

        expected = log_odds(z.reshape(-1, 1), a.reshape(-1, 1), b).ravel()
        recovered = result.log_odds().to_numpy().ravel()
        assert abs(spearmanr(expected, recovered).statistic) > 0.9
