import os
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.schemas import (
    HeartDiseaseFeatures,
    PredictResponse,
    InfoResponse,
    HealthResponse,
)

MODEL_DIR = os.getenv("MODEL_DIR", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.joblib")

model = None
model_metadata = {
    "model_type": "Unknown",
    "feature_list": [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ],
    "accuracy": None,
}


def load_model():
    global model, model_metadata
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        model_metadata["model_type"] = loaded.get("model_type", type(model).__name__)
        model_metadata["feature_list"] = loaded.get(
            "features", model_metadata["feature_list"]
        )
        model_metadata["accuracy"] = loaded.get("accuracy")
    else:
        model = loaded
        model_metadata["model_type"] = type(model).__name__

    print(
        f"Loaded model successfully: {model_metadata['model_type']} with {len(model_metadata['feature_list'])} features."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    try:
        load_model()
    except Exception as e:
        print(f"Warning on startup loading model: {e}")
    yield
    # Shutdown logic if needed


app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI service serving ML predictions for heart disease detection based on clinical patient features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint to verify service readiness."""
    return HealthResponse(status="ok")


@app.get("/info", response_model=InfoResponse, tags=["Model Info"])
def get_info():
    """Returns details about the trained classifier and required input features."""
    return InfoResponse(
        model_type=model_metadata["model_type"],
        feature_list=model_metadata["feature_list"],
        accuracy=model_metadata["accuracy"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(features: HeartDiseaseFeatures):
    """Predicts presence (True) or absence (False) of heart disease given patient features."""
    global model
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"Model could not be loaded: {str(e)}"
            )

    feature_dict = features.model_dump()
    # Format input row following feature_list ordering
    input_data = pd.DataFrame(
        [[feature_dict[feat] for feat in model_metadata["feature_list"]]],
        columns=model_metadata["feature_list"],
    )

    try:
        pred = int(model.predict(input_data)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            # Probability of positive class (heart disease = 1)
            prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

        return PredictResponse(
            heart_disease=bool(pred == 1),
            prediction=pred,
            probability=round(prob, 4) if prob is not None else None,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error occurred: {str(e)}"
        )
