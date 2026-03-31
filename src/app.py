from fastapi import FastAPI
from src.schemas import AnalysisInput, AnalysisOutput
from src.service import run_analysis

app = FastAPI()

@app.post("/analyze", response_model=AnalysisOutput)
def analyze(input_data: AnalysisInput):
    return run_analysis(input_data.data)
