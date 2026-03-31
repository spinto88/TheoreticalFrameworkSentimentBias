from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.schemas import AnalysisInput, AnalysisOutput
from src.service import run_analysis

app = FastAPI()

@app.post("/analyze", response_model=AnalysisOutput)
def analyze(input_data: AnalysisInput):
    return run_analysis(input_data.data)

@app.get("/")
def read_index():
    return FileResponse("src/static/index.html")

app.mount("/static", StaticFiles(directory="src/static"), name="static")
