"""
api.py  —  FastAPI wrapper for the Dar es Salaam Business Advisory Model
============================================================================
Endpoints:
  GET  /health                  → quick sanity check (is the server alive?)
  GET  /districts               → list valid district names
  GET  /activities              → list all activities in the catalog
  POST /predict/path-a          → user HAS a business idea → evaluate it
  POST /predict/path-b          → user has NO idea → get recommendations

RUNNING LOCALLY:
  pip install fastapi uvicorn
  uvicorn api:app --reload --port 8000

Then test in browser: http://localhost:8000/docs  (auto-generated Swagger UI)
Or with curl:
  curl -X POST http://localhost:8000/predict/path-b \ 
       -H "Content-Type: application/json" \ 
       -d '{"district":"Ilala","ward":"Bonyokwa","capital_tzs":5000000,"age":28,"gender":"female"}' 
"""

import sys
import os

# ── Make sure Python can find predict.py and features.py ──────────────────────
# api.py lives in the same folder as predict.py; adjust this path if needed.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd                     

# ── Load the model ONCE at startup (not per request!) ─────────────────────────
# This is the key performance fix: joblib loads are slow (1-3 sec).
# Doing it here means the first request is instant.
print("Loading model artifacts...")
from predict import predict_path_a, predict_path_b, activity_catalog
from predict import location_lookup
print("Model loaded. Server ready.")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Dar es Salaam Business Advisory API",
    description="Predicts business success and recommends activities for entrepreneurs in Dar es Salaam.",
    version="1.0.0",
)

# CORS — allows your React frontend (on a different port) to call this API.
# Tighten origins in production; "*" is fine for local dev / demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Valid values (used for validation + docs) ──────────────────────────────────
VALID_DISTRICTS = sorted(location_lookup["District_Name"].dropna().unique().tolist())


# ── Request/Response schemas ───────────────────────────────────────────────────

class PathARequest(BaseModel):
    """
    Path A: the user already has a business idea.
    Provide the 4-digit ISIC code of the activity.
    Get the /activities endpoint to see all valid ISIC codes.
    """
    isic_detailed: int = Field(..., example=4711, description="4-digit ISIC activity code")
    district: str = Field(..., example="Ilala", description="One of: Ilala, Temeke, Kinondoni, Ubungo, Kigamboni")
    ward: Optional[str] = Field(None, example="Bonyokwa", description="Ward name (optional, improves accuracy)")
    village: Optional[str] = Field(None, example=None, description="Village name (stored for reference only)")
    capital_tzs: Optional[float] = Field(None, example=5_000_000, description="Startup capital in TZS (optional)")
    age: Optional[int] = Field(None, example=28, ge=16, le=100)
    gender: Optional[str] = Field(None, example="female", description="'male' or 'female' (optional)")
    workers: int = Field(1, example=1, ge=1, description="Number of workers including owner")

class PathBRequest(BaseModel):
    """
    Path B: the user has no idea yet — get ranked recommendations.
    """
    district: str = Field(..., example="Ilala")
    ward: Optional[str] = Field(None, example="Bonyokwa")
    village: Optional[str] = Field(None)
    capital_tzs: Optional[float] = Field(None, example=5_000_000)
    age: Optional[int] = Field(None, example=28, ge=16, le=100)
    gender: Optional[str] = Field(None, example="female")
    workers: int = Field(1, ge=1)
    top_n: int = Field(5, ge=1, le=10, description="How many recommendations to return (max 10)")


# ── Helper ─────────────────────────────────────────────────────────────────────

def _validate_district(district: str):
    if district not in VALID_DISTRICTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown district '{district}'. Valid options: {VALID_DISTRICTS}"
        )

def _validate_gender(gender: Optional[str]):
    if gender is not None and gender.lower() not in ("male", "female"):
        raise HTTPException(status_code=422, detail="gender must be 'male' or 'female'")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utilities"])
def health():
    """Quick ping — returns OK if the server is running and model is loaded."""
    return {"status": "ok", "model": "loaded"}


@app.get("/districts", tags=["Utilities"])
def get_districts():
    """Returns the list of valid district names for this model."""
    return {"districts": VALID_DISTRICTS}


@app.get("/activities", tags=["Utilities"])
def get_activities(district: Optional[str] = None, sector: Optional[str] = None):
    """
    Returns all activities in the catalog.
    Optionally filter by sector letter (e.g. sector=G for retail trade).
    """
    df = activity_catalog.copy()
    if sector:
        df = df[df["ISIC_Section"] == sector.upper()]
    return {
        "count": len(df),
        "activities": df[["ISIC_Detailed", "MainActivityDescription", "ISIC_Section",
                           "Sector_Name", "Typical_Capital_TZS"]].to_dict(orient="records")
    }


@app.post("/predict/path-a", tags=["Predictions"])
def predict_a(body: PathARequest):
    """
    Path A — evaluate a specific business idea.

    Returns: success_chance, expected profit, ROI, break-even months,
    existing competition in the area, and any age/gender warnings.
    """
    _validate_district(body.district)
    _validate_gender(body.gender)

    try:
        result = predict_path_a(
            isic_detailed=body.isic_detailed,
            district=body.district,
            ward=body.ward,
            village=body.village,
            capital_tzs=body.capital_tzs,
            age=body.age,
            gender=body.gender,
            workers=body.workers,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return result


@app.post("/predict/path-b", tags=["Predictions"])
def predict_b(body: PathBRequest):
    """
    Path B — recommend business ideas for someone with no specific idea yet.

    Returns a ranked list of activities, each with profit estimates and competition info.
    """
    _validate_district(body.district)
    _validate_gender(body.gender)

    try:
        result = predict_path_b(
            district=body.district,
            ward=body.ward,
            village=body.village,
            capital_tzs=body.capital_tzs,
            age=body.age,
            gender=body.gender,
            workers=body.workers,
            top_n=body.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return result