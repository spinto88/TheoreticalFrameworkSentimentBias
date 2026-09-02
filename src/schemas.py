"""
Pydantic schemas for the /analyze and /generate endpoints.

/analyze
    Request  : AnalysisInput  — wraps a list of Mention records plus the number
               of latent dimensions (1 or 2) to fit, and the number of
               independent optimisation restarts to run.
    Response : AnalysisOutput — carries per-outlet and per-subject scores,
               each expressed as a list of D floats (one per latent dimension),
               taken from whichever restart achieved the lowest loss.

/generate
    Request  : GenerateInput  — known latent parameters (z, a, b) plus the
               desired number of mentions to draw per (outlet, subject) pair.
    Response : AnalysisInput  — synthetic Mention records sampled from the
               generative model.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Mention(BaseModel):
    """A single mention observation from a media outlet.

    Attributes:
        outlet: Name or identifier of the media outlet.
        subject: Name or identifier of the subject being mentioned.
        mention_type: Sentiment polarity of the mention.
        amount_of_mentions: Raw count of mentions in this category.
            Must be a non-negative integer.
    """

    outlet: str
    subject: str
    mention_type: Literal["negative", "neutral", "positive"]
    amount_of_mentions: int = Field(..., ge=0)


class AnalysisInput(BaseModel):
    """Request body for the /analyze endpoint.

    Attributes:
        data: One or more mention observations.  Multiple rows may share
            the same (outlet, subject, mention_type) triple — their
            counts will be summed when building the mention tensor.
        n_dimensions: Number of latent dimensions to fit (1 or 2).
            Defaults to 1 (equivalent to the original scalar model).
        n_restarts: Number of independent optimisation restarts to run,
            each from a different random initialisation. The restart with
            the lowest loss is returned. Defaults to 1 (a single run); any
            positive value is accepted.
        ignore_neutral: If True, neutral mention counts are treated as 0
            (excluded from the fit entirely) before the model is run.
            Defaults to False.
        fixed_a: If provided, the discrimination vector ``a`` is held at
            this fixed value (length D) for every subject instead of being
            estimated. Must have exactly ``n_dimensions`` entries. Defaults
            to None (``a`` is estimated normally).
    """

    data: List[Mention]
    n_dimensions: int = Field(1, ge=1, le=2)
    n_restarts: int = Field(1, ge=1)
    ignore_neutral: bool = False
    fixed_a: Optional[List[float]] = None

    @model_validator(mode="after")
    def _check_fixed_a_length(self) -> "AnalysisInput":
        if self.fixed_a is not None and len(self.fixed_a) != self.n_dimensions:
            raise ValueError(
                f"fixed_a must have exactly {self.n_dimensions} entries "
                f"(one per latent dimension), got {len(self.fixed_a)}."
            )
        return self


class OutletScore(BaseModel):
    """Estimated bias score(s) for a single media outlet.

    Attributes:
        outlet: Outlet name as provided in the input.
        z: Latent bias vector of length D (one entry per dimension).
            The sign of each component is only meaningful in combination
            with the matching dimension of the subject's discrimination
            parameter ``a``.
    """

    outlet: str
    z: List[float]


class SubjectScore(BaseModel):
    """Estimated parameters for a single subject.

    Attributes:
        subject: Subject name as provided in the input.
        a: Discrimination vector of length D.  Higher absolute values
            in dimension d indicate that outlet bias along that axis
            has a stronger effect when this subject is covered.
        b: Scalar baseline sentiment.  Reflects the overall media sentiment
            toward this subject, independent of outlet bias and shared
            across all latent dimensions.
    """

    subject: str
    a: List[float]
    b: float


class AnalysisOutput(BaseModel):
    """Response body returned by the /analyze endpoint.

    Attributes:
        outlets: One :class:`OutletScore` per unique outlet in the input.
        subjects: One :class:`SubjectScore` per unique subject in the input.
        loss: Final value of the minimised objective (negative penalised
            log-likelihood), from the best of the requested restarts.
            Lower values indicate a better fit.
        bic: Bayesian Information Criteria. Final value of the minimised objective but penalizing the dimensionality of the model.  Lower values indicate a better fit.
        n_restarts: Number of independent optimisation restarts that were run
            to produce this result.
    """

    outlets: List[OutletScore]
    subjects: List[SubjectScore]
    loss: float
    bic: float
    n_restarts: int = 1


class GenerateInput(BaseModel):
    """Request body for the /generate endpoint.

    Attributes:
        outlets: Latent bias scores for each outlet to simulate.
        subjects: Discrimination and baseline parameters for each subject
            to simulate.
        amount_of_mentions: Total number of mentions to draw for each
            (outlet, subject) pair.  Defaults to 100.
    """

    outlets: List[OutletScore]
    subjects: List[SubjectScore]
    amount_of_mentions: int = Field(100, ge=0)
