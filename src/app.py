"""
FastAPI application entry point.

Routes
------
POST /analyze
    Accepts a JSON body matching :class:`~src.schemas.AnalysisInput` and
    returns an :class:`~src.schemas.AnalysisOutput` with estimated bias
    parameters.

POST /generate
    Accepts a JSON body matching :class:`~src.schemas.GenerateInput` and
    returns an :class:`~src.schemas.AnalysisInput` with synthetic mention
    data sampled from the generative model.

GET /
    Serves the static frontend (``src/static/index.html``).
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.schemas import AnalysisInput, AnalysisOutput, GenerateInput
from src.service import run_analysis, generate_data

app = FastAPI(
    title="Sentiment Bias Analyzer",
    description="Estimates latent bias scores for media outlets and subjects from mention data.",
    version="0.1.0",
)


@app.post("/analyze", response_model=AnalysisOutput)
def analyze(input_data: AnalysisInput) -> AnalysisOutput:
    """Run the bias estimation model on a list of mention records.

    Args:
        input_data: Request body containing one or more
            :class:`~src.schemas.Mention` observations.

    Returns:
        Estimated *z* scores per outlet and (*a*, *b*) parameters per
        subject.
    """
    return run_analysis(input_data.data, n_dims=input_data.n_dimensions)


@app.post("/generate", response_model=AnalysisInput)
def generate(input_data: GenerateInput) -> AnalysisInput:
    """Generate synthetic data from a list of ideological scores of outlets and subjects

    Args:
        input_data: Request body containing expected
          *z* scores per outlet and (*a*, *b*) parameters per subject.
    Returns:
        :class:`~src.schemas.Mention` observations.
        
    """
    return generate_data(outlets = input_data.outlets, subjects = input_data.subjects, amount_of_mentions = input_data.amount_of_mentions)

@app.get("/")
def read_index() -> FileResponse:
    """Serve the frontend SPA."""
    return FileResponse("src/static/index.html")


app.mount("/static", StaticFiles(directory="src/static"), name="static")
