"""
Latent sentiment bias in media coverage.

Given mention counts broken down by (outlet, subject, sentiment polarity),
this package fits a latent-trait model relating outlet bias to subject
sentiment by maximum penalised likelihood::

    q_ij = dot(z[i], a[j]) + b[j]

where ``q_ij`` is the log-odds of a positive vs. negative mention of subject
*j* by outlet *i*, ``z[i]`` is the outlet's latent bias vector, ``a[j]`` the
subject's discrimination vector and ``b[j]`` its scalar baseline sentiment.

Fitting::

    import sentiment_bias as sb

    tensor = sb.read_csv("mentions.csv")
    fit = sb.fit(tensor, n_dimensions=1, n_restarts=10, seed=0)
    fit.z          # outlet bias, one row per outlet
    fit.subjects   # a and b, one row per subject

Simulating::

    tensor = sb.simulate(z, a, b, n_mentions="poisson", mean_mentions=100, seed=0)
    reduced = sb.thin(tensor, fraction=0.5, seed=0)

The optional HTTP layer lives in ``sentiment_bias.api`` and requires the
``api`` extra.
"""

from sentiment_bias import model
from sentiment_bias.data import (
    SENTIMENTS,
    MentionTensor,
    from_array,
    from_long,
    from_records,
    from_wide,
    read_csv,
    read_json,
    to_tensor,
)
from sentiment_bias.estimate import fit
from sentiment_bias.results import BiasFit
from sentiment_bias.simulate import simulate, thin

__version__ = "0.1.0"

__all__ = [
    "BiasFit",
    "MentionTensor",
    "SENTIMENTS",
    "__version__",
    "fit",
    "from_array",
    "from_long",
    "from_records",
    "from_wide",
    "model",
    "read_csv",
    "read_json",
    "simulate",
    "thin",
    "to_tensor",
]
