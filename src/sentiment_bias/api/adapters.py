"""
Translation between the Pydantic wire format and the core's numpy/pandas types.

Every conversion between the two worlds lives here, so the core never imports
Pydantic and the routes stay three lines long.
"""

from typing import List

import numpy as np
from numpy.typing import NDArray

from sentiment_bias.api.schemas import (
    AnalysisInput,
    AnalysisOutput,
    Mention,
    OutletScore,
    SubjectScore,
)
from sentiment_bias.data import MentionTensor, from_records
from sentiment_bias.results import BiasFit

__all__ = [
    "tensor_from_mentions",
    "fit_to_output",
    "tensor_to_input",
    "scores_to_arrays",
]


def tensor_from_mentions(mentions: List[Mention]) -> MentionTensor:
    """Build a mention tensor from the wire format's mention list.

    Args:
        mentions: Mention records from an :class:`AnalysisInput`.

    Returns:
        The corresponding :class:`~sentiment_bias.data.MentionTensor`.
    """
    return from_records(mentions)


def fit_to_output(fit: BiasFit) -> AnalysisOutput:
    """Render a fitted model as the /analyze response body.

    Args:
        fit: The result of :func:`sentiment_bias.fit`.

    Returns:
        The corresponding :class:`AnalysisOutput`.
    """
    return AnalysisOutput(
        outlets=[
            OutletScore(outlet=name, z=row.tolist())
            for name, row in zip(fit.z.index, fit.z.to_numpy())
        ],
        subjects=[
            SubjectScore(subject=name, a=row.tolist(), b=float(fit.b.loc[name]))
            for name, row in zip(fit.a.index, fit.a.to_numpy())
        ],
        loss=fit.loss,
        bic=fit.bic,
        n_restarts=fit.n_restarts,
    )


def tensor_to_input(tensor: MentionTensor, n_dimensions: int) -> AnalysisInput:
    """Render simulated counts as the /generate response body.

    Args:
        tensor: The simulated mention counts.
        n_dimensions: Dimensionality of the parameters that generated them.

    Returns:
        An :class:`AnalysisInput` with one mention record per
        (outlet, subject, sentiment) combination.
    """
    frame = tensor.to_frame("long")
    return AnalysisInput(
        data=frame.to_dict(orient="records"),
        n_dimensions=n_dimensions,
    )


def scores_to_arrays(
    outlets: List[OutletScore],
    subjects: List[SubjectScore],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], List[str], List[str]]:
    """Unpack wire-format parameter scores into arrays for :func:`simulate`.

    Args:
        outlets: Latent bias scores, one per outlet.
        subjects: Discrimination and baseline parameters, one per subject.

    Returns:
        ``(z, a, b, outlet_names, subject_names)`` with ``z`` shaped
        ``(m, D)``, ``a`` shaped ``(k, D)`` and ``b`` shaped ``(k,)``.
    """
    z = np.array([outlet.z for outlet in outlets], dtype=np.float64)
    a = np.array([subject.a for subject in subjects], dtype=np.float64)
    b = np.array([subject.b for subject in subjects], dtype=np.float64)
    return z, a, b, [o.outlet for o in outlets], [s.subject for s in subjects]
