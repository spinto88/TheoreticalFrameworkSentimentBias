from pydantic import BaseModel
from typing import List, Literal


class Mention(BaseModel):
    outlet: str
    subject: str
    mention_type: Literal["negative", "neutral", "positive"]
    amount_of_mentions: int

class AnalysisInput(BaseModel):
    data: List[Mention]

class OutletScore(BaseModel):
    outlet: str
    z: float

class SubjectScore(BaseModel):
    subject: str
    a: float
    b: float

class AnalysisOutput(BaseModel):
    outlets: List[OutletScore]
    subjects: List[SubjectScore]
