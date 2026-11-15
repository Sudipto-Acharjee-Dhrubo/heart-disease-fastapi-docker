from typing import List, Optional
from pydantic import BaseModel, Field


class HeartDiseaseFeatures(BaseModel):
    age: int = Field(..., description="Age of the patient in years", ge=1, le=120)
    sex: int = Field(..., description="Sex (1 = male, 0 = female)", ge=0, le=1)
    cp: int = Field(..., description="Chest pain type (0, 1, 2, 3)", ge=0, le=3)
    trestbps: float = Field(..., description="Resting blood pressure (in mm Hg)", ge=50, le=300)
    chol: float = Field(..., description="Serum cholesterol in mg/dl", ge=50, le=700)
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", ge=0, le=1)
    restecg: int = Field(..., description="Resting electrocardiographic results (0, 1, 2)", ge=0, le=2)
    thalach: float = Field(..., description="Maximum heart rate achieved", ge=50, le=250)
    exang: int = Field(..., description="Exercise induced angina (1 = yes, 0 = no)", ge=0, le=1)
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest", ge=0.0, le=10.0)
    slope: int = Field(..., description="Slope of the peak exercise ST segment (0, 1, 2)", ge=0, le=2)
    ca: int = Field(..., description="Number of major vessels (0-4) colored by fluoroscopy", ge=0, le=4)
    thal: int = Field(..., description="Thalassemia (0, 1, 2, 3)", ge=0, le=3)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145.0,
                "chol": 233.0,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150.0,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }
    }


class PredictResponse(BaseModel):
    heart_disease: bool = Field(..., description="Binary prediction: True if heart disease detected, False otherwise")
    prediction: int = Field(..., description="Raw model prediction (0 or 1)")
    probability: Optional[float] = Field(None, description="Confidence score / probability of heart disease")


class InfoResponse(BaseModel):
    model_type: str = Field(..., description="Type/architecture of the ML classifier")
    feature_list: List[str] = Field(..., description="List of expected input features in order")
    accuracy: Optional[float] = Field(None, description="Reported test accuracy score of the trained model")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Application health status")
