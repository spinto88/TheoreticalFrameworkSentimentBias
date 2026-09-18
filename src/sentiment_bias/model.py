"""
Latent bias model: likelihood, gradient and parameter algebra.

Pure NumPy — no pandas, no Pydantic, no I/O. This is the only module that
needs to be read to verify the mathematics.

The model has D latent dimensions:

  - z[i, d] : d-th bias component for media outlet i
  - a[j, d] : d-th discrimination parameter for subject j
  - b[j]    : scalar baseline sentiment for subject j (always 1-D)

The log-odds of a positive vs. negative mention for outlet i covering
subject j is:

    q_ij = dot(z[i], a[j]) + b[j]
         = sum_d( z[i,d] * a[j,d] ) + b[j]

Each dimension d couples only z[i,d] with a[j,d].  The baseline b[j]
is shared across all dimensions.  Setting D=1 recovers the original
scalar model.

Mention counts for a pair (i, j) follow a three-category multinomial
(negative, neutral, positive) with probabilities derived from q_ij:

    P(negative) = exp(-q) / Z,  P(neutral) = 1 / Z,  P(positive) = exp(q) / Z
    Z = exp(-q) + 1 + exp(q)

Flat parameter vector layout for m outlets, k subjects, D dimensions
(total size (m + k) * D + k)::

    x = [z[0,0], …, z[0,D-1], z[1,0], …, z[m-1,D-1],   # m*D elements
         a[0,0], …, a[k-1,D-1],                        # k*D elements
         b[0],   …, b[k-1]]                            # k   elements
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "pack",
    "unpack",
    "log_odds",
    "log_likelihood",
    "negative_log_likelihood",
    "grad_negative_log_likelihood",
    "bic",
    "canonicalize",
]


def pack(
    z: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Flatten ``(z, a, b)`` into the solver's parameter vector.

    Args:
        z: Bias scores, shape ``(m, D)``.
        a: Discrimination parameters, shape ``(k, D)``.
        b: Baseline sentiment, shape ``(k,)``.

    Returns:
        Flat vector of length ``(m + k) * D + k``.
    """
    return np.concatenate([np.ravel(z), np.ravel(a), np.ravel(b)])


def unpack(
    x: NDArray[np.float64],
    n_outlets: int,
    n_subjects: int,
    n_dimensions: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Split the solver's parameter vector back into ``(z, a, b)``.

    Args:
        x: Flat parameter vector of length ``(m + k) * D + k``.
        n_outlets: Number of outlets *m*.
        n_subjects: Number of subjects *k*.
        n_dimensions: Number of latent dimensions D.

    Returns:
        ``(z, a, b)`` with shapes ``(m, D)``, ``(k, D)`` and ``(k,)``.
    """
    split_z = n_outlets * n_dimensions
    split_a = split_z + n_subjects * n_dimensions
    z = x[:split_z].reshape(n_outlets, n_dimensions)
    a = x[split_z:split_a].reshape(n_subjects, n_dimensions)
    b = x[split_a:]
    return z, a, b


def log_odds(
    z: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the log-odds matrix ``q_ij = dot(z[i], a[j]) + b[j]``.

    Args:
        z: Bias scores, shape ``(m, D)``.
        a: Discrimination parameters, shape ``(k, D)``.
        b: Baseline sentiment, shape ``(k,)``.

    Returns:
        Array of shape ``(m, k)``.
    """
    return z @ a.T + b


def log_likelihood(
    x: NDArray[np.float64],
    counts: NDArray[np.int_],
    n_dimensions: int = 1,
    *,
    penalized: bool = True,
) -> float:
    """Compute the log-likelihood of the bias model.

    Args:
        x: Flat parameter vector of length ``(m + k) * D + k``.
        counts: Mention counts, shape ``(m, k, 3)``.
        n_dimensions: Number of latent dimensions D (default 1).
        penalized: Whether to subtract the Gaussian (L2) penalty on every
            scalar parameter. The unpenalised value depends on the
            parameters only through ``q``, which makes it invariant under
            every reparametrisation the model admits — including
            :func:`canonicalize`.

    Returns:
        The log-likelihood value (higher is better).
    """
    n_rows, n_cols = counts.shape[0], counts.shape[1]
    z, a, b = unpack(x, n_rows, n_cols, n_dimensions)

    logl: float = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            q_ij: float = float(np.dot(z[i], a[j])) + float(b[j])
            N_ij: int = int(np.sum(counts[i, j]))
            logl += (
                (counts[i, j, 2] - counts[i, j, 0]) * q_ij
                - N_ij * np.log(np.exp(q_ij) + 1 + np.exp(-q_ij))
            )

    if penalized:
        logl -= 0.5 * float(np.sum(z ** 2))
        logl -= 0.5 * float(np.sum(a ** 2))
        logl -= 0.5 * float(np.sum(b ** 2))

    return logl


def negative_log_likelihood(
    x: NDArray[np.float64],
    counts: NDArray[np.int_],
    n_dimensions: int = 1,
) -> float:
    """Return the negated log-likelihood, for use with minimisation solvers.

    Args:
        x: Flat parameter vector of length ``(m + k) * D + k``.
        counts: Mention counts, shape ``(m, k, 3)``.
        n_dimensions: Number of latent dimensions D (default 1).

    Returns:
        ``-log_likelihood(x, counts, n_dimensions)``.
    """
    return -log_likelihood(x, counts, n_dimensions)


def grad_negative_log_likelihood(
    x: NDArray[np.float64],
    counts: NDArray[np.int_],
    n_dimensions: int = 1,
) -> NDArray[np.float64]:
    """Compute the gradient of the negative penalised log-likelihood.

    For each dimension d, the partial derivatives are:

    - ``dL/dz[i,d] = sum_j W_ij * a[j,d] - z[i,d]``
    - ``dL/da[j,d] = sum_i W_ij * z[i,d] - a[j,d]``
    - ``dL/db[j]   = sum_i W_ij           - b[j]``   (scalar)

    where ``W_ij = (pos_ij - neg_ij) - N_ij * (exp(q_ij) - exp(-q_ij))
    / (exp(q_ij) + 1 + exp(-q_ij))`` and ``q_ij = dot(z[i], a[j]) + b[j]``.

    Args:
        x: Flat parameter vector of length ``(m + k) * D + k``.
        counts: Mention counts, shape ``(m, k, 3)``.
        n_dimensions: Number of latent dimensions D (default 1).

    Returns:
        Gradient vector of length ``(m + k) * D + k``.
    """
    n_rows, n_cols = counts.shape[0], counts.shape[1]
    z, a, b = unpack(x, n_rows, n_cols, n_dimensions)

    grad_z: NDArray[np.float64] = np.zeros_like(z)
    grad_a: NDArray[np.float64] = np.zeros_like(a)
    grad_b: NDArray[np.float64] = np.zeros_like(b)

    for i in range(n_rows):
        for j in range(n_cols):
            q_ij: float = float(np.dot(z[i], a[j])) + float(b[j])
            N_ij: int = int(np.sum(counts[i, j]))
            W_ij: float = (
                float(counts[i, j, 2] - counts[i, j, 0])
                - N_ij * (np.exp(q_ij) - np.exp(-q_ij)) / (np.exp(q_ij) + 1 + np.exp(-q_ij))
            )
            grad_z[i] += W_ij * a[j]
            grad_a[j] += W_ij * z[i]
            grad_b[j] += W_ij  # scalar accumulation

    grad_z -= z
    grad_a -= a
    grad_b -= b

    return -pack(grad_z, grad_a, grad_b)


def bic(neg_log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Return the approximate Bayesian Information Criterion.

    It is approximate because the L2 penalty's effect on the log-posterior
    is treated as negligible, i.e. the log-posterior is taken to be close
    to the log-likelihood.  Lower is better.

    Args:
        neg_log_likelihood: Minimised negative log-likelihood.
        n_parameters: Number of free parameters, ``m * D + k * (D + 1)``.
        n_observations: Number of independent mentions.

    Returns:
        The BIC value.
    """
    return 2 * neg_log_likelihood + n_parameters * np.log(n_observations)


def canonicalize(
    z: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    sign_reference_idx: int = 0,
    sign_reference_positive: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Put the fitted parameters in a canonical gauge, for any D.

    The likelihood is invariant under the whole affine group
    ``z -> A(z - c)``, ``a -> A^-T a``, ``b -> b + c.a`` for any
    invertible ``A``, so the raw optimizer output has no fixed location,
    scale, orientation or sign. The L2 penalty removes most of that
    freedom but is itself invariant under orthogonal ``A``, leaving
    exactly ``O(D)`` — rotations and reflections — undetermined. This
    picks one representative, leaving every ``q_ij`` unchanged:

    1. ``sum(z) == 0`` along each dimension, the shift absorbed into ``b``.
    2. ``z`` rotated to its principal axes, so ``z.T @ z`` is diagonal with
       entries in decreasing order. Without this the orientation is
       arbitrary and a rerun from a different seed returns a different one.
    3. ``sum(z**2) == 1`` summed over all dimensions, via a single global
       factor absorbed into ``a``. Deliberately not one factor per column:
       the ratios between the eigenvalues of ``z.T @ z`` are rotation
       invariant, hence identified, and say how much each dimension
       carries — rescaling columns separately would destroy them.
    4. Reflections fixed one dimension at a time. Dimension 1 follows
       ``sign_reference_idx``; the rest put their largest-magnitude
       component positive, since the reference outlet may sit near zero
       on the higher dimensions.

    Note step 3 is unit *norm*, not unit variance: with zero mean the two
    are mutually exclusive, since ``sum(z**2) == m * var(z)``. For D=1
    that means ``var(z) == 1 / m`` and ``a`` is ``sqrt(m)`` times larger
    than it would be under unit variance.

    Two caveats worth knowing:

    - This runs *after* the optimisation, so it is purely a change of
      representation. The box constraints apply to the raw parameter
      vector, and the ``a`` returned here is not bounded by them. It also
      means the result is no longer at the penalised optimum, so
      :func:`negative_log_likelihood` evaluated here exceeds the
      optimiser's final value; the unpenalised likelihood is unchanged.
    - When two singular values of ``z`` coincide the rotation is not
      unique, and the result stops being deterministic within that
      subspace.

    Args:
        z: Estimated bias scores, shape ``(m, D)``.
        a: Estimated discrimination parameters, shape ``(k, D)``.
        b: Estimated baseline sentiment parameters, shape ``(k,)``.
        sign_reference_idx: Index of the outlet whose sign on dimension 1
            determines that dimension's sign convention.
        sign_reference_positive: If True, the reference outlet's first
            component is made positive; if False, negative.

    Returns:
        The canonicalized ``(z, a, b)``, shaped as the inputs.
    """
    centre = z.mean(axis=0)
    z_new = z - centre
    b_new = b + a @ centre

    scale = float(np.linalg.norm(z_new))
    if scale == 0:
        return z, a, b  # degenerate: all outlets identical (or a single outlet)

    # Rotate onto the principal axes of z. numpy orders the singular values
    # decreasingly, so dimension 1 is the one carrying the most variation.
    rotation = np.linalg.svd(z_new, full_matrices=False)[2].T
    z_new = z_new @ rotation
    a_new = (a @ rotation) * scale
    z_new = z_new / scale

    for d in range(z_new.shape[1]):
        column = z_new[:, d]
        if d == 0:
            reference = column[sign_reference_idx]
            flip = reference < 0 if sign_reference_positive else reference > 0
        else:
            flip = column[np.argmax(np.abs(column))] < 0
        if flip:
            z_new[:, d] = -column
            a_new[:, d] = -a_new[:, d]

    return z_new, a_new, b_new
