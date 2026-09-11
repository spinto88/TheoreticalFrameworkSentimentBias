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
    "canonicalize_1d",
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
) -> float:
    """Compute the penalised log-likelihood of the bias model.

    A Gaussian (L2) penalty is applied to every scalar parameter.

    Args:
        x: Flat parameter vector of length ``(m + k) * D + k``.
        counts: Mention counts, shape ``(m, k, 3)``.
        n_dimensions: Number of latent dimensions D (default 1).

    Returns:
        The penalised log-likelihood value (higher is better).
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


def canonicalize_1d(
    z: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    sign_reference_idx: int = 0,
    sign_reference_positive: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Standardize 1-D outlet bias to zero mean / unit variance.

    The likelihood is invariant under ``z -> (z - c) / s``,
    ``a -> a * s``, ``b -> b + c * a`` for any ``c``/``s``, so the raw
    optimizer output has no fixed scale or sign. This picks the
    representative with ``z`` at zero mean / unit variance (absorbing
    the rescaling into ``a`` and the mean shift into ``b`` so that
    ``q_ij = dot(z, a) + b`` is unchanged), then flips the sign of
    ``z`` and ``a`` if needed so that ``z[sign_reference_idx]`` has the
    requested sign.

    Args:
        z: Estimated bias scores, shape ``(m, 1)``.
        a: Estimated discrimination parameters, shape ``(k, 1)``.
        b: Estimated baseline sentiment parameters, shape ``(k,)``.
        sign_reference_idx: Index of the outlet whose sign after
            standardization determines the overall sign convention.
        sign_reference_positive: If True, the reference outlet's ``z``
            is made positive; if False, negative.

    Returns:
        The canonicalized ``(z, a, b)``, with ``z`` and ``a`` still
        shaped ``(m, 1)`` / ``(k, 1)``.
    """
    z_flat = z[:, 0]
    a_flat = a[:, 0]

    z_mean, z_std = z_flat.mean(), z_flat.std()
    if z_std == 0:
        return z, a, b  # degenerate: all outlets identical (or a single outlet)

    z_new = (z_flat - z_mean) / z_std
    a_new = a_flat * z_std
    b_new = b + z_mean * a_flat

    ref = z_new[sign_reference_idx]
    flip = (sign_reference_positive and ref < 0) or (not sign_reference_positive and ref > 0)
    if flip:
        z_new, a_new = -z_new, -a_new

    return z_new.reshape(-1, 1), a_new.reshape(-1, 1), b_new
