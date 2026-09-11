"""Maximum penalised likelihood estimation of the bias model."""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import Bounds, OptimizeResult, minimize

from sentiment_bias import model
from sentiment_bias.data import to_tensor
from sentiment_bias.results import BiasFit, dimension_columns

__all__ = ["fit"]


def fit(
    data: Any,
    n_dimensions: int = 1,
    *,
    n_restarts: int = 1,
    sign_reference_outlet: str | None = None,
    sign_reference_positive: bool = True,
    bounds: tuple[float, float] = (-5.0, 5.0),
    tol: float = 1e-6,
    seed: int | np.random.Generator | None = None,
) -> BiasFit:
    """Estimate latent bias parameters from mention counts.

    Maximises the L2-penalised log-likelihood with L-BFGS-B and an analytical
    gradient, under symmetric box constraints on every parameter.

    Because the objective is non-convex the solver can settle into different
    local minima depending on its random initialisation. When ``n_restarts >
    1`` the optimisation runs that many times from independent starting
    points and the restart with the lowest loss is kept.

    When ``n_dimensions == 1`` the fitted parameters are canonicalized (see
    :func:`sentiment_bias.model.canonicalize_1d`), since the 1-D model is
    only identified up to an affine reparametrization of ``z``.

    Args:
        data: Mention counts in any form accepted by
            :func:`sentiment_bias.to_tensor` — a ``MentionTensor``, a wide or
            long DataFrame, an ``(m, k, 3)`` array, or mention records.
        n_dimensions: Number of latent dimensions D to fit.
        n_restarts: Number of independent optimisation restarts.
        sign_reference_outlet: When ``n_dimensions == 1``, the outlet whose
            canonicalized ``z`` sign is pinned by ``sign_reference_positive``.
            Defaults to the first outlet alphabetically.
        sign_reference_positive: Whether the reference outlet's ``z`` should
            come out positive (default) or negative.
        bounds: Lower and upper box constraint applied to every parameter.
        tol: Convergence tolerance handed to the solver.
        seed: Seed or generator for the restart initialisations, making the
            fit reproducible.

    Returns:
        A :class:`~sentiment_bias.results.BiasFit` with the best restart's
        parameters and diagnostics.
    """
    if n_dimensions < 1:
        raise ValueError(f"n_dimensions must be >= 1, got {n_dimensions}")
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

    tensor = to_tensor(data)
    counts = tensor.counts
    m, k = tensor.n_outlets, tensor.n_subjects
    n_parameters = (m + k) * n_dimensions + k

    rng = np.random.default_rng(seed)
    box = Bounds([bounds[0]] * n_parameters, [bounds[1]] * n_parameters)

    best: OptimizeResult | None = None
    losses: list[float] = []
    for _ in range(n_restarts):
        x0: NDArray[np.float64] = rng.normal(size=n_parameters)
        solution: OptimizeResult = minimize(
            fun=model.negative_log_likelihood,
            x0=x0,
            args=(counts, n_dimensions),
            method="L-BFGS-B",
            jac=model.grad_negative_log_likelihood,
            bounds=box,
            tol=tol,
        )
        losses.append(float(solution.fun))
        if best is None or solution.fun < best.fun:
            best = solution

    z, a, b = model.unpack(best.x, m, k, n_dimensions)

    if n_dimensions == 1:
        index = (
            tensor.outlets.index(sign_reference_outlet)
            if sign_reference_outlet is not None
            else 0
        )
        z, a, b = model.canonicalize_1d(
            z, a, b,
            sign_reference_idx=index,
            sign_reference_positive=sign_reference_positive,
        )

    loss = float(best.fun)
    columns = dimension_columns(n_dimensions)
    return BiasFit(
        z=pd.DataFrame(z, index=pd.Index(tensor.outlets, name="outlet"), columns=columns),
        a=pd.DataFrame(a, index=pd.Index(tensor.subjects, name="subject"), columns=columns),
        b=pd.Series(b, index=pd.Index(tensor.subjects, name="subject"), name="b"),
        loss=loss,
        bic=float(model.bic(loss, n_parameters, tensor.n_mentions)),
        n_dimensions=n_dimensions,
        n_restarts=n_restarts,
        converged=bool(best.success),
        losses=losses,
        data=tensor,
        x=model.pack(z, a, b),
    )
