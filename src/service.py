"""
Core analysis service.

Implements the latent bias estimation model:

  - z_i  : bias vector of dimension D for media outlet i
  - a_j  : discrimination vector of dimension D for subject j
             (how strongly outlets polarise when covering this subject)
  - b[j] : baseline sentiment for subject j
             (overall media sentiment independent of outlet bias)

The model assumes the log-odds of positive vs. negative mentions for
outlet i covering subject j equal the dot product of their latent vectors
plus a subject-level intercept:

    log-odds(i, j) = dot(z_i, a_j) + b[j]

Setting D=1 recovers the original scalar model z[i] * a[j] + b[j].

Parameters are estimated by maximising a penalised log-likelihood
(Gaussian regularisation on all parameters) using SciPy's
differential evolution solver.
"""

import json
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, Bounds, OptimizeResult
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
        z: Estimated bias vectors, shape ``(m, D)``.
        a: Estimated discrimination vectors, shape ``(k, D)``.
        b: Estimated baseline sentiment parameters, shape ``(k,)``.

    Returns:
        An :class:`~src.schemas.AnalysisOutput` instance ready for
        serialisation.
    """
    outlet_scores: List[OutletScore] = [
        OutletScore(outlet=outlets[i], z=z[i].tolist())
        for i in range(len(outlets))
    ]
    subject_scores: List[SubjectScore] = [
        SubjectScore(subject=subjects[j], a=a[j].tolist(), b=float(b[j]))
        for j in range(len(subjects))
    ]
    return AnalysisOutput(outlets=outlet_scores, subjects=subject_scores)


def log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
    D: int = 1,
) -> float:
    """Compute the penalised log-likelihood of the bias model.

    The likelihood is derived from a symmetric ordinal model where, for
    outlet *i* and subject *j*, the log-odds of a positive vs. negative
    mention equals ``dot(z_i, a_j) + b[j]``.  The normalising denominator
    accounts for neutral mentions via a three-category softmax.

    A Gaussian (L2) penalty with unit variance is applied to all
    parameters.  For the vector parameters ``z_i`` and ``a_j`` the
    penalty is the sum of squares of their components.

    The parameter vector ``x`` is laid out as::

        x = [z_0, ..., z_{m-1},   # each z_i is a block of D values
             a_0, ..., a_{k-1},   # each a_j is a block of D values
             b_0, ..., b_{k-1}]

    Total length: ``m*D + k*D + k``.

    Args:
        x: Flat parameter vector of length ``m*D + k*D + k``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.
        D: Dimensionality of the latent space.  Defaults to 1.

    Returns:
        The penalised log-likelihood value (higher is better).
    """
    m: int = mentions_matrix.shape[0]
    k: int = mentions_matrix.shape[1]

    z: NDArray[np.float64] = x[: m * D].reshape(m, D)
    a: NDArray[np.float64] = x[m * D : m * D + k * D].reshape(k, D)
    b: NDArray[np.float64] = x[m * D + k * D :]

    logl: float = 0.0
    for i in range(m):
        for j in range(k):
            z_ij: float = float(np.dot(z[i], a[j])) + b[j]
            N_ij: int = int(np.sum(mentions_matrix[i, j]))
            logl += (
                (mentions_matrix[i, j, 2] - mentions_matrix[i, j, 0]) * z_ij
                - N_ij * np.log(np.exp(z_ij) + 1 + np.exp(-z_ij))
            )
            logl -= 0.5 * (np.sum(z[i] ** 2) + np.sum(a[j] ** 2) + b[j] ** 2)

    return float(logl)


def negative_log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
    D: int = 1,
) -> float:
    """Return the negated log-likelihood for use with minimisation solvers.

    Args:
        x: Flat parameter vector of length ``m*D + k*D + k``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.
        D: Dimensionality of the latent space.  Defaults to 1.

    Returns:
        ``-log_likelihood(x, mentions_matrix, D)``.
    """
    return -log_likelihood(x, mentions_matrix, D)


def run_analysis(data: List[Mention], D: int = 1) -> AnalysisOutput:
    """Estimate latent bias parameters from a list of mention records.

    Builds the mention tensor, sets symmetric box constraints
    ``[-5, 5]`` on all parameters, and runs differential evolution to
    minimise the negative penalised log-likelihood.

    Args:
        data: List of :class:`~src.schemas.Mention` objects covering one
            or more outlet–subject–sentiment combinations.
        D: Dimensionality of the latent space.  Each outlet bias vector
            ``z_i`` and each subject discrimination vector ``a_j`` will
            have *D* components.  Defaults to 1, which recovers the
            original scalar model.

    Returns:
        An :class:`~src.schemas.AnalysisOutput` with estimated *z*
        vectors for each outlet and (*a* vector, *b* scalar) parameters
        for each subject.
    """
    mentions_matrix, outlets, subjects = build_tensor(data)

    m: int = len(outlets)
    k: int = len(subjects)
    n_params: int = m * D + k * D + k

    bounds: Bounds = Bounds([-5] * n_params, [5] * n_params)

    solution: OptimizeResult = differential_evolution(
        negative_log_likelihood,
        args=(mentions_matrix, D),
        bounds=bounds,
        maxiter=1000,
        popsize=25,
    )

    z = solution.x[: m * D].reshape(m, D)
    a = solution.x[m * D : m * D + k * D].reshape(k, D)
    b = solution.x[m * D + k * D :]

    return build_output(outlets, subjects, z, a, b)


if __name__ == "__main__":
    input_data = json.load(open("data/input.json", "r"))
    print(run_analysis(input_data["data"]))
