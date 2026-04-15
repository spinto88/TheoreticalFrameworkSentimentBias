"""
Pydantic schemas for the /analyze endpoint.

Request  : AnalysisInput  — wraps a list of Mention records.
Response : AnalysisOutput — carries per-outlet and per-subject scores.
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
        n_dimensions: Dimensionality *D* of the latent space.  Each outlet
            bias vector ``z_i`` and each subject discrimination vector
            ``a_j`` will have this many components.  Defaults to 1,
            which recovers the original scalar model.
    """

    data: List[Mention]
    n_dimensions: int = Field(1, ge=1)


class OutletScore(BaseModel):
    """Estimated bias vector for a single media outlet.

    Attributes:
        outlet: Outlet name as provided in the input.
        z: Latent bias vector of length *D*.  In the 1-D case a positive
            value indicates a tendency to cover subjects positively and a
            negative value indicates the opposite.
    """

    outlet: str
    z: List[float]


class SubjectScore(BaseModel):
    """Estimated parameters for a single subject.

    Attributes:
        subject: Subject name as provided in the input.
        a: Discrimination vector of length *D*.  The log-odds contribution
            of outlet bias for this subject is the dot product
            ``z_i · a_j``.
        b: Baseline sentiment parameter.  Reflects the overall media
            sentiment toward this subject, independent of outlet bias.
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
