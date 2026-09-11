"""
FastAPI application: a thin HTTP layer over the sentiment_bias package.

Routes
------
POST /analyze
    Accepts a JSON body matching :class:`~sentiment_bias.api.schemas.AnalysisInput`
    and returns an :class:`~sentiment_bias.api.schemas.AnalysisOutput` with
    estimated bias parameters.

POST /generate
    Accepts a JSON body matching :class:`~sentiment_bias.api.schemas.GenerateInput`
    and returns an :class:`~sentiment_bias.api.schemas.AnalysisInput` with
    synthetic mention data sampled from the generative model.

GET /
    Serves the static frontend bundled with the package.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sentiment_bias import fit, simulate
from sentiment_bias.api import adapters
from sentiment_bias.api.schemas import AnalysisInput, AnalysisOutput, GenerateInput

STATIC_DIR = Path(__file__).resolve().parent / "static"

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
            :class:`~sentiment_bias.api.schemas.Mention` observations.

    Returns:
        Estimated *z* scores per outlet and (*a*, *b*) parameters per
        subject.
    """
    tensor = adapters.tensor_from_mentions(input_data.data)
    result = fit(
        tensor,
        n_dimensions=input_data.n_dimensions,
        n_restarts=input_data.n_restarts,
        sign_reference_outlet=input_data.sign_reference_outlet,
        sign_reference_positive=input_data.sign_reference_positive,
    )
    return adapters.fit_to_output(result)


@app.post("/generate", response_model=AnalysisInput)
def generate(input_data: GenerateInput) -> AnalysisInput:
    """Generate synthetic mention data from known ideological scores.

    Args:
        input_data: Request body containing expected *z* scores per outlet
            and (*a*, *b*) parameters per subject.

    Returns:
        :class:`~sentiment_bias.api.schemas.Mention` observations sampled
        from the generative model.
    """
    z, a, b, outlets, subjects = adapters.scores_to_arrays(
        input_data.outlets, input_data.subjects
    )
    tensor = simulate(
        z, a, b,
        n_mentions="poisson" if input_data.poisson else input_data.amount_of_mentions,
        mean_mentions=input_data.amount_of_mentions if input_data.poisson else None,
        outlets=outlets,
        subjects=subjects,
    )
    return adapters.tensor_to_input(tensor, n_dimensions=z.shape[1])


@app.get("/")
def read_index() -> FileResponse:
    """Serve the frontend SPA."""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
