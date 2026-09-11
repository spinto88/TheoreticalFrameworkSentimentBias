"""
Core analysis service.

Implements the latent bias estimation model with D latent dimensions:

  - z[i, d] : d-th bias component for media outlet i
  - a[j, d] : d-th discrimination parameter for subject j
  - b[j]    : scalar baseline sentiment for subject j (always 1-D)

The log-odds of a positive vs. negative mention for outlet i covering
subject j is:

    log-odds(i, j) = dot(z[i], a[j]) + b[j]
                   = sum_d( z[i,d] * a[j,d] ) + b[j]

Each dimension d couples only z[i,d] with a[j,d].  The baseline b[j]
is shared across all dimensions.  Setting D=1 recovers the original
scalar model.

Parameters are estimated by maximising a penalised log-likelihood
(Gaussian regularisation on all parameters) using SciPy's
L-BFGS-B gradient descent solver with an analytical gradient.

Flat parameter vector layout for m outlets, k subjects, D dimensions
(total size (m + k) * D + k)::

    x = [z[0,0], …, z[0,D-1], z[1,0], …, z[m-1,D-1],   # m*D elements
         a[0,0], …, a[k-1,D-1],                           # k*D elements
         b[0],   …, b[k-1]]                               # k   elements
"""

import json
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, Bounds, OptimizeResult
from scipy.stats import multinomial 
from typing import List

from src.schemas import AnalysisOutput, Mention, OutletScore, SubjectScore, AnalysisInput


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
    loss: float,
    bic: float,
    n_restarts: int = 1,
) -> AnalysisOutput:
    """Assemble the API response object from estimated parameters.

    Args:
        outlets: Ordered list of outlet names (length *m*).
        subjects: Ordered list of subject names (length *k*).
        z: Estimated bias scores, shape ``(m, D)``.
        a: Estimated discrimination parameters, shape ``(k, D)``.
        b: Estimated baseline sentiment parameters, shape ``(k,)`` — scalar per subject.
        loss: Final value of the minimised objective (``solution.fun`` from
            L-BFGS-B).  Lower values indicate a better fit.
        bic: Bayesian Information Criteria. It penalizes the dimensionality of the model. Lower values indicate a better fit.
        n_restarts: Number of independent optimisation restarts that were run
            to produce this result.

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
    return AnalysisOutput(
        outlets=outlet_scores,
        subjects=subject_scores,
        loss=float(loss),
        bic=float(bic),
        n_restarts=n_restarts,
    )


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


def log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
    n_dims: int = 1,
) -> float:
    """Compute the penalised log-likelihood of the bias model.

    The linear predictor for outlet *i* and subject *j* is::

        q_ij = dot(z[i], a[j]) + b[j]

    where ``b[j]`` is a scalar baseline shared across all dimensions.

    A Gaussian (L2) penalty is applied to every scalar parameter.

    Args:
        x: Flat parameter vector of length ``(m + k) * n_dims + k``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.
        n_dims: Number of latent dimensions D (default 1).

    Returns:
        The penalised log-likelihood value (higher is better).
    """
    n_rows: int = mentions_matrix.shape[0]
    n_cols: int = mentions_matrix.shape[1]

    z: NDArray[np.float64] = x[: n_rows * n_dims].reshape(n_rows, n_dims)
    a: NDArray[np.float64] = x[n_rows * n_dims : (n_rows + n_cols) * n_dims].reshape(n_cols, n_dims)
    b: NDArray[np.float64] = x[(n_rows + n_cols) * n_dims :]  # shape (k,)

    logl: float = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            q_ij: float = float(np.dot(z[i], a[j])) + float(b[j])
            N_ij: int = int(np.sum(mentions_matrix[i, j]))
            logl += (
                (mentions_matrix[i, j, 2] - mentions_matrix[i, j, 0]) * q_ij
                - N_ij * np.log(np.exp(q_ij) + 1 + np.exp(-q_ij))
            )

    logl -= 0.5 * float(np.sum(z ** 2))
    logl -= 0.5 * float(np.sum(a ** 2))
    logl -= 0.5 * float(np.sum(b ** 2))

    return logl


def negative_log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
    n_dims: int = 1,
) -> float:
    """Return the negated log-likelihood for use with minimisation solvers.

    Args:
        x: Flat parameter vector of length ``(m + k) * n_dims + k``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.
        n_dims: Number of latent dimensions D (default 1).

    Returns:
        ``-log_likelihood(x, mentions_matrix, n_dims)``.
    """
    return -log_likelihood(x, mentions_matrix, n_dims)

def aproximate_bayesian_information_criteria(neglogl: float, number_of_parameters: int, n_data: int) -> float:

    """ Return the aproximate Bayesian Information Criteria.
    It is an aproximation because we are assuming that priors have little effects, so we say the log-posterior is very similar to log-likelihood. 

    Args:
        neglogl: Negative Log Likelihood.
        number_of_parameters of the model: m * D + k * (D + 1) .
        n_data: Amount of data, which in this case are the n independent mentions.
    Returns:
        ``BIC: Bayesian Information Criteria.
    """
    bic = 2 * neglogl + number_of_parameters * np.log(n_data)
    return bic

def grad_negative_log_likelihood(
    x: NDArray[np.float64],
    mentions_matrix: NDArray[np.int_],
    n_dims: int = 1,
) -> NDArray[np.float64]:
    """Compute the gradient of the negative penalised log-likelihood.

    For each dimension d, the partial derivatives are:

    - ``dL/dz[i,d] = sum_j W_ij * a[j,d] - z[i,d]``
    - ``dL/da[j,d] = sum_i W_ij * z[i,d] - a[j,d]``
    - ``dL/db[j]   = sum_i W_ij           - b[j]``   (scalar)

    where ``W_ij = (pos_ij - neg_ij) - N_ij * (exp(q_ij) - exp(-q_ij))
    / (exp(q_ij) + 1 + exp(-q_ij))`` and ``q_ij = dot(z[i], a[j]) + b[j]``.

    Args:
        x: Flat parameter vector of length ``(m + k) * n_dims + k``.
        mentions_matrix: Mention counts, shape ``(m, k, 3)``.
        n_dims: Number of latent dimensions D (default 1).

    Returns:
        Gradient vector of length ``(m + k) * n_dims + k``.
    """
    n_rows: int = mentions_matrix.shape[0]
    n_cols: int = mentions_matrix.shape[1]

    z: NDArray[np.float64] = x[: n_rows * n_dims].reshape(n_rows, n_dims)
    a: NDArray[np.float64] = x[n_rows * n_dims : (n_rows + n_cols) * n_dims].reshape(n_cols, n_dims)
    b: NDArray[np.float64] = x[(n_rows + n_cols) * n_dims :]  # shape (k,)

    grad_z: NDArray[np.float64] = np.zeros_like(z)
    grad_a: NDArray[np.float64] = np.zeros_like(a)
    grad_b: NDArray[np.float64] = np.zeros_like(b)

    for i in range(n_rows):
        for j in range(n_cols):
            q_ij: float = float(np.dot(z[i], a[j])) + float(b[j])
            N_ij: int = int(np.sum(mentions_matrix[i, j]))
            W_ij: float = (
                float(mentions_matrix[i, j, 2] - mentions_matrix[i, j, 0])
                - N_ij * (np.exp(q_ij) - np.exp(-q_ij)) / (np.exp(q_ij) + 1 + np.exp(-q_ij))
            )
            grad_z[i] += W_ij * a[j]
            grad_a[j] += W_ij * z[i]
            grad_b[j] += W_ij  # scalar accumulation

    grad_z -= z
    grad_a -= a
    grad_b -= b

    return -np.concatenate([grad_z.ravel(), grad_a.ravel(), grad_b])


def run_analysis(
    data: List[Mention],
    n_dims: int = 1,
    n_restarts: int = 1,
    sign_reference_outlet: str | None = None,
    sign_reference_positive: bool = True,
) -> AnalysisOutput:
    """Estimate latent bias parameters from a list of mention records.

    Builds the mention tensor, sets symmetric box constraints
    ``[-5, 5]`` on all parameters, and runs L-BFGS-B to minimise the
    negative penalised log-likelihood using an analytical gradient.

    Because the objective is non-convex, the solver can settle into
    different local minima depending on its random initialisation. When
    ``n_restarts > 1``, the optimisation is run that many times from
    independent random starting points and the restart with the lowest
    loss (``solution.fun``) is kept.

    When ``n_dims == 1``, the fitted ``z``/``a``/``b`` are canonicalized
    (see :func:`canonicalize_1d`) since the 1-D model is only identified
    up to an affine reparametrization of ``z``.

    Args:
        data: List of :class:`~src.schemas.Mention` objects covering one
            or more outlet–subject–sentiment combinations.
        n_dims: Number of latent dimensions D to fit (default 1).
        n_restarts: Number of independent optimisation restarts to run
            (default 1). The restart achieving the lowest loss is returned.
        sign_reference_outlet: When ``n_dims == 1``, the name of the
            outlet whose canonicalized ``z`` sign is controlled by
            ``sign_reference_positive``. Defaults to the first outlet
            (alphabetically) when None.
        sign_reference_positive: When ``n_dims == 1``, whether the
            reference outlet's canonicalized ``z`` should be positive
            (default) or negative.

    Returns:
        An :class:`~src.schemas.AnalysisOutput` with estimated *z*
        vectors (length D) and *a* vectors (length D) per outlet/subject,
        and a scalar *b* per subject, taken from the best restart.
    """
    mentions_matrix, outlets, subjects = build_tensor(data)

    m: int = len(outlets)
    k: int = len(subjects)
    n_params: int = (m + k) * n_dims + k
    n_mentions: int = mentions_matrix.sum()

    bounds: Bounds = Bounds([-5] * n_params, [5] * n_params)

    best_solution: OptimizeResult | None = None
    for _ in range(n_restarts):
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

        if best_solution is None or solution.fun < best_solution.fun:
            best_solution = solution

    z = best_solution.x[: m * n_dims].reshape(m, n_dims)
    a = best_solution.x[m * n_dims : (m + k) * n_dims].reshape(k, n_dims)
    b = best_solution.x[(m + k) * n_dims :]

    if n_dims == 1:
        idx = outlets.index(sign_reference_outlet) if sign_reference_outlet is not None else 0
        z, a, b = canonicalize_1d(z, a, b, sign_reference_idx=idx, sign_reference_positive=sign_reference_positive)

    bic = aproximate_bayesian_information_criteria(best_solution.fun, n_params, n_mentions)

    return build_output(outlets, subjects, z, a, b, loss=best_solution.fun, bic=bic, n_restarts=n_restarts)


def generate_mentions(q: float, n: int) -> NDArray[np.int_]:
    """Draw multinomial mention counts for a single (outlet, subject) pair.

    The probability of each sentiment category is derived from the
    log-odds ``q`` via the three-category model::

        P(negative) = exp(-q) / Z
        P(neutral)  = 1       / Z
        P(positive) = exp( q) / Z
        Z           = exp(-q) + 1 + exp(q)

    Args:
        q: Log-odds value for this (outlet, subject) pair,
            ``q_ij = dot(z_i, a_j) + b_j``.
        n: Total number of mentions to draw.

    Returns:
        Array of shape ``(3,)`` with integer counts for
        (negative, neutral, positive) mentions summing to *n*.
    """
    Z = np.exp(-q) + 1 + np.exp(q)
    probabilities = np.array([np.exp(-q), 1, np.exp(q)]) / Z
    return multinomial.rvs(n=n, p=probabilities)


def generate_data(
        outlets: List[OutletScore],
        subjects: List[SubjectScore],
        amount_of_mentions: int = 100,
    ) -> AnalysisInput:
    """Generate synthetic mention data from known ideological parameters.

    Simulates the generative model: for each (outlet, subject) pair,
    computes ``q_ij = dot(z_i, a_j) + b_j`` and draws *amount_of_mentions*
    mentions from the resulting multinomial distribution.

    Args:
        outlets: Latent bias scores for each outlet, each carrying a name
            and a bias vector *z* of length D.
        subjects: Discrimination and baseline parameters for each subject,
            each carrying a name, a discrimination vector *a* of length D,
            and a scalar baseline *b*.
        amount_of_mentions: Total number of mentions to draw for each
            (outlet, subject) pair.  Defaults to 100.

    Returns:
        An :class:`~src.schemas.AnalysisInput` containing one
        :class:`~src.schemas.Mention` row per (outlet, subject, sentiment)
        combination and ``n_dimensions`` set to the dimensionality of the
        input bias vectors.
    """
    idx_to_type: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
    mentions: list[dict] = []

    for outlet in outlets:
        for subject in subjects:
            qij = np.array(outlet.z).dot(np.array(subject.a)) + subject.b
            counts = generate_mentions(qij, amount_of_mentions)
            for k in range(3):
                mentions.append(
                    {
                        "outlet": outlet.outlet,
                        "subject": subject.subject,
                        "mention_type": idx_to_type[k],
                        "amount_of_mentions": int(counts[k]),
                    }
                )

    return AnalysisInput(data=mentions, n_dimensions=len(outlets[0].z))

if __name__ == "__main__":
    input_data = json.load(open("data/input.json", "r"))
    print(run_analysis(input_data["data"]))
