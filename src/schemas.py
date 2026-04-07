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
    """

    data: List[Mention]


class OutletScore(BaseModel):
    """Estimated bias score for a single media outlet.

    Attributes:
        outlet: Outlet name as provided in the input.
        z: Latent bias score.  Positive values indicate a tendency to
            cover subjects positively; negative values indicate a
            negative tendency.
    """

    outlet: str
    z: float


class SubjectScore(BaseModel):
    """Estimated parameters for a single subject.

    Attributes:
        subject: Subject name as provided in the input.
        a: Discrimination parameter.  Higher absolute values indicate
            that outlet bias has a stronger effect when this subject is
            being covered.
        b: Baseline sentiment parameter.  Reflects the overall media
            sentiment toward this subject, independent of outlet bias.
    """

    subject: str
    a: float
    b: float


class AnalysisOutput(BaseModel):
    """Response body returned by the /analyze endpoint.

    Attributes:
        outlets: One :class:`OutletScore` per unique outlet in the input.
        subjects: One :class:`SubjectScore` per unique subject in the input.
    """

    outlets: List[OutletScore]
    subjects: List[SubjectScore]
