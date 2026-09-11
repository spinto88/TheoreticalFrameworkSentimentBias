"""
The generative direction: draw synthetic mention counts from known parameters.

This is the other half of the validation loop — simulate with a known
``(z, a, b)``, fit, and check how well the parameters are recovered — and it
is what the model-consistency experiments rest on.
"""

from dataclasses import replace
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sentiment_bias import model
from sentiment_bias.data import MentionTensor, from_array

__all__ = ["simulate", "thin"]


def simulate(
    z: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    *,
    n_mentions: int | ArrayLike | Literal["poisson"] = 100,
    mean_mentions: float | None = None,
    outlets: Sequence[str] | None = None,
    subjects: Sequence[str] | None = None,
    seed: int | np.random.Generator | None = None,
) -> MentionTensor:
    """Draw synthetic mention counts from known model parameters.

    For every (outlet, subject) pair the log-odds ``q_ij = dot(z_i, a_j) +
    b_j`` fixes a three-category multinomial, from which ``N_ij`` mentions are
    drawn.

    Args:
        z: Outlet bias, shape ``(m,)`` or ``(m, D)``; 1-D input means ``D=1``.
        a: Subject discrimination, shape ``(k,)`` or ``(k, D)``.
        b: Subject baseline sentiment, shape ``(k,)``.
        n_mentions: How many mentions to draw per pair. An ``int`` uses the
            same count everywhere; ``"poisson"`` draws
            ``N_ij ~ Poisson(mean_mentions)`` independently per pair; an
            ``(m, k)`` array gives the count for each pair explicitly.
        mean_mentions: The Poisson mean; required by — and only valid with —
            ``n_mentions="poisson"``.
        outlets: Outlet names; defaults to ``outlet_00``, ``outlet_01``, …
        subjects: Subject names; defaults to ``subject_00``, ``subject_01``, …
        seed: Seed or generator, making the draw reproducible.

    Returns:
        A :class:`~sentiment_bias.data.MentionTensor` of the sampled counts.
    """
    z_array = np.atleast_2d(np.asarray(z, dtype=np.float64).T).T
    a_array = np.atleast_2d(np.asarray(a, dtype=np.float64).T).T
    b_array = np.asarray(b, dtype=np.float64)

    if z_array.shape[1] != a_array.shape[1]:
        raise ValueError(
            f"z has {z_array.shape[1]} dimensions but a has {a_array.shape[1]}"
        )
    if b_array.shape != (a_array.shape[0],):
        raise ValueError(
            f"b must have shape ({a_array.shape[0]},), got {b_array.shape}"
        )

    m, k = z_array.shape[0], a_array.shape[0]
    rng = np.random.default_rng(seed)
    sizes = _resolve_sizes(n_mentions, mean_mentions, m, k, rng)
    q = model.log_odds(z_array, a_array, b_array)

    counts = np.empty((m, k, 3), dtype=np.int64)
    for i in range(m):
        for j in range(k):
            counts[i, j] = _sample_counts(float(q[i, j]), int(sizes[i, j]), rng)

    return from_array(counts, outlets=outlets, subjects=subjects)


def _resolve_sizes(
    n_mentions: int | ArrayLike | Literal["poisson"],
    mean_mentions: float | None,
    m: int,
    k: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Turn the ``n_mentions``/``mean_mentions`` pair into an ``(m, k)`` array."""
    if isinstance(n_mentions, str):
        if n_mentions != "poisson":
            raise ValueError(
                f"n_mentions must be an int, an (m, k) array or 'poisson', got {n_mentions!r}"
            )
        if mean_mentions is None:
            raise ValueError("n_mentions='poisson' requires mean_mentions")
        return rng.poisson(mean_mentions, size=(m, k)).astype(np.int64)

    if mean_mentions is not None:
        raise ValueError("mean_mentions is only valid with n_mentions='poisson'")

    if np.isscalar(n_mentions):
        return np.full((m, k), int(n_mentions), dtype=np.int64)

    sizes = np.asarray(n_mentions, dtype=np.int64)
    if sizes.shape != (m, k):
        raise ValueError(f"n_mentions array must have shape ({m}, {k}), got {sizes.shape}")
    return sizes


def _sample_counts(q: float, n: int, rng: np.random.Generator) -> NDArray[np.int64]:
    """Draw one ``(3,)`` count vector for a pair with log-odds ``q``.

    The category probabilities follow the three-category model::

        P(negative) = exp(-q) / Z
        P(neutral)  = 1       / Z
        P(positive) = exp( q) / Z
        Z           = exp(-q) + 1 + exp(q)
    """
    weights = np.array([np.exp(-q), 1.0, np.exp(q)])
    return rng.multinomial(n, weights / weights.sum())


def thin(
    tensor: MentionTensor,
    fraction: float,
    *,
    seed: int | np.random.Generator | None = None,
) -> MentionTensor:
    """Subsample a tensor down to a fraction of its total mentions.

    Draws ``round(fraction * n_mentions)`` mentions without replacement from
    the pool of all observed mentions, which is how much data the model would
    have seen under a shorter collection window. Used to measure how the fit
    degrades as data thins out.

    Args:
        tensor: The counts to subsample.
        fraction: Fraction of total mentions to keep, in ``(0, 1]``.
        seed: Seed or generator, making the draw reproducible.

    Returns:
        A new :class:`~sentiment_bias.data.MentionTensor` with the same shape
        and names, where no cell exceeds its original count.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    rng = np.random.default_rng(seed)
    kept = rng.multivariate_hypergeometric(
        tensor.counts.ravel(), round(fraction * tensor.n_mentions)
    )
    return replace(tensor, counts=kept.reshape(tensor.counts.shape).astype(np.int64))
