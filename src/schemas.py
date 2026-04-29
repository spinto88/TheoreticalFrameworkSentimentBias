"""
Pydantic schemas for the /analyze and /generate endpoints.

/analyze
    Request  : AnalysisInput  — wraps a list of Mention records plus the number
               of latent dimensions (1 or 2) to fit.
    Response : AnalysisOutput — carries per-outlet and per-subject scores,
               each expressed as a list of D floats (one per latent dimension).

/generate
    Request  : GenerateInput  — known latent parameters (z, a, b) plus the
               desired number of mentions to draw per (outlet, subject) pair.
    Response : AnalysisInput  — synthetic Mention records sampled from the
               generative model.
"""

from typing import List, Literal

from pydantic import BaseModel, Field


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
    """

    data: List[Mention]
    n_dimensions: int = Field(1, ge=1, le=2)


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
    """

    outlets: List[OutletScore]
    subjects: List[SubjectScore]


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