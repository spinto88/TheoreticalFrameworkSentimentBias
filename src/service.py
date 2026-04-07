"""
Core analysis service.

Implements the latent bias estimation model:

  - z[i]  : bias score for media outlet i
  - a[j]  : discrimination parameter for subject j
             (how strongly outlets polarise when covering this subject)
  - b[j]  : baseline sentiment for subject j
             (overall media sentiment independent of outlet bias)

The model assumes the log-odds of positive vs. negative mentions for
outlet i covering subject j follow a linear function of z[i]:

    log-odds(i, j) = z[i] * a[j] + b[j]

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
        z: Estimated bias scores, shape ``(m,)``.
        a: Estimated discrimination parameters, shape ``(k,)``.
        b: Estimated baseline sentiment parameters, shape ``(k,)``.

    Returns:
        An :class:`~src.schemas.AnalysisOutput` instance ready for
        serialisation.
    """
    outlet_scores: List[OutletScore] = [
        OutletScore(outlet=outlets[i], z=float(z[i]))
        for i in range(len(outlets))
    ]
    subject_scores: List[SubjectScore] = [
        SubjectScore(subject=subjects[j], a=float(a[j]), b=float(b[j]))
        for j in range(len(subjects))
    ]
    return AnalysisOutput(outlets=outlet_scores, subjects=subject_scores)


def log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
) -> float:
    """Compute the penalised log-likelihood of the bias model.

    The likelihood is derived from a symmetric ordinal model where, for
    outlet *i* and subject *j*, the log-odds of a positive vs. negative
    mention equals ``z[i] * a[j] + b[j]``.  The normalising denominator
    accounts for neutral mentions via a three-category softmax.

    A Gaussian (L2) penalty with unit variance is applied to all
    parameters to regularise the solution.

    The parameter vector ``x`` is laid out as::

        x = [z_0, ..., z_{m-1}, a_0, ..., a_{k-1}, b_0, ..., b_{k-1}]

    Args:
        x: Flat parameter vector of length ``m + 2k``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.

    Returns:
        The penalised log-likelihood value (higher is better).
    """
    n_rows: int = mentions_matrix.shape[0]
    n_cols: int = mentions_matrix.shape[1]

    z: NDArray[np.float64] = x[:n_rows]
    a: NDArray[np.float64] = x[n_rows : n_rows + n_cols]
    b: NDArray[np.float64] = x[n_rows + n_cols :]

    logl: float = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            z_ij: float = z[i] * a[j] + b[j]
            N_ij: int = int(np.sum(mentions_matrix[i, j]))
            logl += (
                (mentions_matrix[i, j, 2] - mentions_matrix[i, j, 0]) * z_ij
                - N_ij * np.log(np.exp(z_ij) + 1 + np.exp(-z_ij))
            )
            logl -= 0.5 * (z[i] ** 2 + a[j] ** 2 + b[j] ** 2)

    return logl


def negative_log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
) -> float:
    """Return the negated log-likelihood for use with minimisation solvers.

    Args:
        x: Flat parameter vector of length ``m + 2k``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.

    Returns:
        ``-log_likelihood(x, mentions_matrix)``.
    """
    return -log_likelihood(x, mentions_matrix)


def run_analysis(data: List[Mention]) -> AnalysisOutput:
    """Estimate latent bias parameters from a list of mention records.

    Builds the mention tensor, sets symmetric box constraints
    ``[-5, 5]`` on all parameters, and runs differential evolution to
    minimise the negative penalised log-likelihood.

    Args:
        data: List of :class:`~src.schemas.Mention` objects covering one
            or more outlet–subject–sentiment combinations.

    Returns:
        An :class:`~src.schemas.AnalysisOutput` with estimated *z*
        scores for each outlet and (*a*, *b*) parameters for each
        subject.
    """
    mentions_matrix, outlets, subjects = build_tensor(data)

    m: int = len(outlets)
    k: int = len(subjects)

    bounds: Bounds = Bounds([-5] * (m + 2 * k), [5] * (m + 2 * k))

    solution: OptimizeResult = differential_evolution(
        negative_log_likelihood,
        args=(mentions_matrix,),
        bounds=bounds,
        maxiter=1000,
        popsize=25,
    )

    z = solution.x[:m]
    a = solution.x[m : m + k]
    b = solution.x[m + k :]

    return build_output(outlets, subjects, z, a, b)


if __name__ == "__main__":
    input_data = json.load(open("data/input.json", "r"))
    print(run_analysis(input_data["data"]))
