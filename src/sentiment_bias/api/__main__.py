"""Console-script entry point: ``sentiment-bias-api``."""

import os

import uvicorn

from sentiment_bias.api.app import app


def main() -> None:
    """Serve the API, honouring SENTIMENT_BIAS_HOST / SENTIMENT_BIAS_PORT."""
    uvicorn.run(
        app,
        host=os.environ.get("SENTIMENT_BIAS_HOST", "0.0.0.0"),
        port=int(os.environ.get("SENTIMENT_BIAS_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
