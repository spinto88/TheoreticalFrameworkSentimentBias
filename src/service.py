"""
Core analysis service.

Implements the latent bias estimation model with D latent dimensions:

  - z[i, d] : d-th bias component for media outlet i
  - a[j, d] : d-th discrimination parameter for subject j
  - b[j, d] : d-th baseline sentiment for subject j

The log-odds of a positive vs. negative mention for outlet i covering
subject j sums contributions across all D dimensions:

    log-odds(i, j) = sum_d( z[i,d] * a[j,d] + b[j,d] )

Each dimension d is independent: z[i,d] only interacts with a[j,d]
and b[j,d].  Setting D=1 recovers the original scalar model.

Parameters are estimated by maximising a penalised log-likelihood
(Gaussian regularisation on all parameters) using SciPy's
L-BFGS-B gradient descent solver with an analytical gradient.

Flat parameter vector layout for m outlets, k subjects, D dimensions
(total size (m + 2k) * D, C-order / row-major)::

    x = [z[0,0], …, z[0,D-1], z[1,0], …, z[m-1,D-1],
         a[0,0], …, a[k-1,D-1],
         b[0,0], …, b[k-1,D-1]]
"""

import json
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, Bounds, OptimizeResult
from typing import List

from src.schemas import AnalysisOutput, Mention, OutletScore, SubjectScore


def build_tensor(
    data: List[Mention],
) -> tuple[NDArray[np.int_], List[str], List[str]]:
    """Build a 3-D mention tensor from a flat list of Mention records.

    Each cell ``matrix[i, j, t]`` holds the number of times outlet *i*
    mentioned subject *j* with sentiment type *t*, where the sentiment
    index mapping is: 0 → negative, 1 → neutral, 2 → positive.

    Args:
        data: List of :class:`~src.schemas.Mention` objects.

    Returns:
        A tuple ``(matrix, outlets, subjects)`` where:

        - ``matrix`` is a NumPy array of shape ``(m, k, 3)`` with dtype
          ``int64``.
        - ``outlets`` is a sorted list of unique outlet names (length *m*).
        - ``subjects`` is a sorted list of unique subject names (length *k*).
    """
    outlets: List[str] = sorted({d.outlet for d in data})
    subjects: List[str] = sorted({d.subject for d in data})

    outlet_to_idx: dict[str, int] = {o: i for i, o in enumerate(outlets)}
    subject_to_idx: dict[str, int] = {s: j for j, s in enumerate(subjects)}

    m, k = len(outlets), len(subjects)
    matrix: NDArray[np.int_] = np.zeros((m, k, 3), dtype=np.int64)

    type_to_idx: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}

    for d in data:
        i = outlet_to_idx[d.outlet]
        j = subject_to_idx[d.subject]
        t = type_to_idx[d.mention_type]
        matrix[i, j, t] += d.amount_of_mentions

    return matrix, outlets, subjects


def build_output(
    outlets: List[str],
    subjects: List[str],
    z: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> AnalysisOutput:
    """Assemble the API response object from estimated parameters.

    Args:
        outlets: Ordered list of outlet names (length *m*).
        subjects: Ordered list of subject names (length *k*).
        z: Estimated bias scores, shape ``(m, D)``.
        a: Estimated discrimination parameters, shape ``(k, D)``.
        b: Estimated baseline sentiment parameters, shape ``(k, D)``.

    Returns:
        An :class:`~src.schemas.AnalysisOutput` instance ready for
        serialisation.
    """
    outlet_scores: List[OutletScore] = [
        OutletScore(outlet=outlets[i], z=z[i].tolist())
        for i in range(len(outlets))
    ]
    subject_scores: List[SubjectScore] = [
        SubjectScore(subject=subjects[j], a=a[j].tolist(), b=b[j].tolist())
        for j in range(len(subjects))
    ]
    return AnalysisOutput(outlets=outlet_scores, subjects=subject_scores)


def log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
    n_dims: int = 1,
) -> float:
    """Compute the penalised log-likelihood of the multidimensional bias model.

    The linear predictor for outlet *i* and subject *j* sums contributions
    from all D latent dimensions::

        q_ij = sum_d( z[i,d] * a[j,d] + b[j,d] )

    A Gaussian (L2) penalty is applied to every scalar parameter.

    Args:
        x: Flat parameter vector of length ``(m + 2k) * n_dims``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.
        n_dims: Number of latent dimensions D (default 1).

    Returns:
        The penalised log-likelihood value (higher is better).
    """
    n_rows: int = mentions_matrix.shape[0]
    n_cols: int = mentions_matrix.shape[1]

    z: NDArray[np.float64] = x[: n_rows * n_dims].reshape(n_rows, n_dims)
    a: NDArray[np.float64] = x[n_rows * n_dims : (n_rows + n_cols) * n_dims].reshape(n_cols, n_dims)
    b: NDArray[np.float64] = x[(n_rows + n_cols) * n_dims :].reshape(n_cols, n_dims)

    logl: float = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            q_ij: float = float(np.sum(z[i] * a[j] + b[j]))
            N_ij: int = int(np.sum(mentions_matrix[i, j]))
            logl += (
                (mentions_matrix[i, j, 2] - mentions_matrix[i, j, 0]) * q_ij
                - N_ij * np.log(np.exp(q_ij) + 1 + np.exp(-q_ij))
            )

    logl -= 0.5 * float(np.sum(z ** 2))
    logl -= 0.5 * float(np.sum(a ** 2 + b ** 2))

    return logl


def negative_log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
    n_dims: int = 1,
) -> float:
    """Return the negated log-likelihood for use with minimisation solvers.

    Args:
        x: Flat parameter vector of length ``(m + 2k) * n_dims``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.
        n_dims: Number of latent dimensions D (default 1).

    Returns:
        ``-log_likelihood(x, mentions_matrix, n_dims)``.
    """
    return -log_likelihood(x, mentions_matrix, n_dims)


def grad_negative_log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
    n_dims: int = 1,
) -> NDArray[np.float64]:
    """Compute the gradient of the negative penalised log-likelihood.

    For each dimension d, the partial derivatives are:

    - ``dL/dz[i,d] = sum_j W_ij * a[j,d] - z[i,d]``
    - ``dL/da[j,d] = sum_i W_ij * z[i,d] - a[j,d]``
    - ``dL/db[j,d] = sum_i W_ij            - b[j,d]``

    where ``W_ij = (pos_ij - neg_ij) - N_ij * tanh(q_ij / 2)``
    and ``q_ij = sum_d( z[i,d] * a[j,d] + b[j,d] )``.

    Since *W_ij* is scalar (independent of d), each dimension of b[j]
    receives the same data gradient — only the regularisation term
    ``-b[j,d]`` distinguishes components.

    Args:
        x: Flat parameter vector of length ``(m + 2k) * n_dims``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.
        n_dims: Number of latent dimensions D (default 1).

    Returns:
        Gradient vector of length ``(m + 2k) * n_dims``.
    """
    n_rows: int = mentions_matrix.shape[0]
    n_cols: int = mentions_matrix.shape[1]

    z: NDArray[np.float64] = x[: n_rows * n_dims].reshape(n_rows, n_dims)
    a: NDArray[np.float64] = x[n_rows * n_dims : (n_rows + n_cols) * n_dims].reshape(n_cols, n_dims)
    b: NDArray[np.float64] = x[(n_rows + n_cols) * n_dims :].reshape(n_cols, n_dims)

    grad_z: NDArray[np.float64] = np.zeros_like(z)
    grad_a: NDArray[np.float64] = np.zeros_like(a)
    grad_b: NDArray[np.float64] = np.zeros_like(b)

    for i in range(n_rows):
        for j in range(n_cols):
            q_ij: float = float(np.sum(z[i] * a[j] + b[j]))
            N_ij: int = int(np.sum(mentions_matrix[i, j]))
            W_ij: float = (
                float(mentions_matrix[i, j, 2] - mentions_matrix[i, j, 0])
                - N_ij * (np.exp(q_ij) - np.exp(-q_ij)) / (np.exp(q_ij) + 1 + np.exp(-q_ij))
            )
            grad_z[i] += W_ij * a[j]
            grad_a[j] += W_ij * z[i]
            grad_b[j] += W_ij  # W_ij is scalar — broadcasts to all D dims

    grad_z -= z
    grad_a -= a
    grad_b -= b

    return -np.concatenate([grad_z.ravel(), grad_a.ravel(), grad_b.ravel()])


def run_analysis(data: List[Mention], n_dims: int = 1) -> AnalysisOutput:
    """Estimate latent bias parameters from a list of mention records.

    Builds the mention tensor, sets symmetric box constraints
    ``[-5, 5]`` on all parameters, and runs L-BFGS-B to minimise the
    negative penalised log-likelihood using an analytical gradient.

    Args:
        data: List of :class:`~src.schemas.Mention` objects covering one
            or more outlet–subject–sentiment combinations.
        n_dims: Number of latent dimensions D to fit (default 1).

    Returns:
        An :class:`~src.schemas.AnalysisOutput` with estimated *z*
        vectors (length D) for each outlet and (*a*, *b*) vectors for
        each subject.
    """
    mentions_matrix, outlets, subjects = build_tensor(data)

    m: int = len(outlets)
    k: int = len(subjects)
    n_params: int = (m + 2 * k) * n_dims

    bounds: Bounds = Bounds([-5] * n_params, [5] * n_params)
    x0: NDArray[np.float64] = np.random.normal(size=n_params)

    solution: OptimizeResult = minimize(
        fun=negative_log_likelihood,
        x0=x0,
        args=(mentions_matrix, n_dims),
        method="L-BFGS-B",
        jac=grad_negative_log_likelihood,
        bounds=bounds,
        tol=1e-6,
    )

    z = solution.x[: m * n_dims].reshape(m, n_dims)
    a = solution.x[m * n_dims : (m + k) * n_dims].reshape(k, n_dims)
    b = solution.x[(m + k) * n_dims :].reshape(k, n_dims)

    return build_output(outlets, subjects, z, a, b)


if __name__ == "__main__":
    input_data = json.load(open("data/input.json", "r"))
    print(run_analysis(input_data["data"]))
