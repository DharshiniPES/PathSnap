"""
FastAPI Backend & Evaluator Dashboard for Laptop 1 and Laptop 2.
PES University UE24CS352A: Machine Learning
PathSnap: Distributed Map-Matching & Route Reconstruction
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.api.inference_engine import PathSnapEngine

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Backend inference service hosting ML classifier and HMM/Viterbi route decoder.",
    version="1.0.0"
)

# Enable CORS for Laptop 2 (User Interface) communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODELS_DIR = os.path.join(BASE_DIR, "src/models/saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data/Student folder")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
UI_HTML_PATH = os.path.join(BASE_DIR, "src/ui/index.html")

# Mount visualizations as static files for UI plots
if os.path.exists(VIZ_DIR):
    app.mount("/static", StaticFiles(directory=VIZ_DIR), name="static")

engine = PathSnapEngine(models_dir=MODELS_DIR, data_dir=DATA_DIR)

@app.get("/health")
def health_check():
    """Health check returning loaded models and service status."""
    return {
        "status": "healthy",
        "service": "PathSnap Model & API Backend (Laptop 1)",
        "loaded_models": list(engine.loaded_models.keys()),
        "recommended_model": "xgboost_tuned",
        "hmm_decoder_available": True
    }

@app.get("/trips")
def list_trips(partition: str = Query("student_test", description="Partition name (student_test or train)")):
    """Returns the list of available trips for evaluation."""
    trips = engine.list_available_trips(partition=partition)
    return {
        "partition": partition,
        "count": len(trips),
        "trips": trips
    }

@app.get("/evaluate/{trip_id}")
@app.post("/evaluate/{trip_id}")
def evaluate_trip(
    trip_id: str,
    model_name: str = Query("xgboost_tuned", description="Model choice: xgboost_tuned, xgboost, random_forest, mlp, logistic_regression, knn"),
    use_hmm: bool = Query(True, description="Enable HMM + Viterbi topological decoder"),
    partition: str = Query("student_test", description="Partition: student_test or train")
):
    """
    Main evaluation endpoint:
    Processes the selected trip through the complete prediction pipeline,
    computes accuracy against the server-side ground-truth route, and returns
    metrics and route coordinates for display on Laptop 2.
    """
    if trip_id.isdigit():
        trip_id = f"{int(trip_id):08d}"

    try:
        result = engine.run_inference(
            trip_id=trip_id,
            partition=partition,
            model_name=model_name,
            use_hmm=use_hmm
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the evaluator dashboard directly from src/ui/index.html."""
    if os.path.exists(UI_HTML_PATH):
        with open(UI_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard file not found.</h1>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
