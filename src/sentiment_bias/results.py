"""The fitted-model container returned by :func:`sentiment_bias.fit`."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sentiment_bias import model
from sentiment_bias.data import MentionTensor

__all__ = ["BiasFit", "dimension_columns"]


def dimension_columns(n_dimensions: int) -> list[str]:
    """Return the column labels used for latent dimensions."""
    return [f"dim_{d + 1}" for d in range(n_dimensions)]


@dataclass(frozen=True)
class BiasFit:
    """Estimated parameters and diagnostics from one model fit.

    Attributes:
        z: Outlet bias scores, indexed by outlet name, one column per
            latent dimension.
        a: Subject discrimination parameters, indexed by subject name,
            same columns as ``z``.
        b: Subject baseline sentiment, indexed by subject name. A Series,
            not a frame: there is genuinely one scalar per subject.
        loss: Minimised negative penalised log-likelihood (lower is better).
        bic: Approximate Bayesian Information Criterion (lower is better).
        n_dimensions: Number of latent dimensions D that were fitted.
        n_restarts: Number of independent optimisation restarts that were run.
        converged: Whether the best restart reported success.
        losses: Final loss of every restart, in the order they were run —
            answers "did the restarts agree, or is the objective multimodal?".
        data: The mention counts that were fitted.
        x: Flat parameter vector of the best restart, after canonicalization.
    """

    z: pd.DataFrame
    a: pd.DataFrame
    b: pd.Series
    loss: float
    bic: float
    n_dimensions: int
    n_restarts: int
    converged: bool
    losses: list[float]
    data: MentionTensor
    x: NDArray[np.float64]

    @property
    def outlets(self) -> list[str]:
        """Outlet names, in the order used by ``z``."""
        return list(self.z.index)

    @property
    def subjects(self) -> pd.DataFrame:
        """``a`` and ``b`` joined, one row per subject."""
        return self.a.join(self.b)

    @property
    def n_parameters(self) -> int:
        """Number of free parameters, ``m * D + k * (D + 1)``."""
        m, k = len(self.z), len(self.a)
        return (m + k) * self.n_dimensions + k

    def log_odds(self) -> pd.DataFrame:
        """Fitted log-odds ``q_ij`` for every (outlet, subject) pair.

        Returns:
            Frame of shape ``(m, k)``, indexed by outlet with subjects as
            columns.
        """
        q = model.log_odds(self.z.to_numpy(), self.a.to_numpy(), self.b.to_numpy())
        return pd.DataFrame(q, index=self.z.index, columns=self.a.index)

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the fit."""
        head = (
            f"BiasFit: D={self.n_dimensions}, {len(self.z)} outlets, {len(self.a)} subjects\n"
            f"  loss={self.loss:.4f}  bic={self.bic:.2f}  "
            f"restarts={self.n_restarts}  {'converged' if self.converged else 'NOT CONVERGED'}\n"
            f"  {self.data.n_mentions} mentions, {self.n_parameters} parameters\n"
        )
        first = self.z.columns[0]
        ranked = self.z[first].sort_values()
        extremes = pd.concat([ranked.head(5), ranked.tail(5)]).drop_duplicates()
        return (
            f"{head}\nOutlet bias ({first}), most negative to most positive:\n"
            f"{extremes.to_string()}\n\nSubjects:\n{self.subjects.to_string()}"
        )

    def to_dict(self) -> dict:
        """Return a plain JSON-serialisable view of the estimated parameters."""
        return {
            "outlets": [
                {"outlet": name, "z": row.tolist()}
                for name, row in zip(self.z.index, self.z.to_numpy())
            ],
            "subjects": [
                {"subject": name, "a": row.tolist(), "b": float(self.b.loc[name])}
                for name, row in zip(self.a.index, self.a.to_numpy())
            ],
            "loss": self.loss,
            "bic": self.bic,
            "n_restarts": self.n_restarts,
        }

    def __repr__(self) -> str:
        status = "converged" if self.converged else "not-converged"
        return (
            f"<BiasFit D={self.n_dimensions} outlets={len(self.z)} subjects={len(self.a)} "
            f"loss={self.loss:.2f} bic={self.bic:.1f} restarts={self.n_restarts} {status}>"
        )
