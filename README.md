# Heart Disease Prediction API (FastAPI + Docker + Render)

A machine learning microservice built with **FastAPI**, containerized using **Docker** and **Docker Compose**, and deployed to **Render**. The service serves predictions from a trained classifier predicting the presence or absence of heart disease based on clinical diagnostic features.

[![Render Deployment](https://img.shields.io/badge/Render-Live%20Deployment-success?style=for-the-badge&logo=render)](https://heart-disease-fastapi-docker-idbe.onrender.com/docs)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

### 🌐 Live Deployment Links
- **Interactive Swagger UI (API Docs)**: [https://heart-disease-fastapi-docker-idbe.onrender.com/docs](https://heart-disease-fastapi-docker-idbe.onrender.com/docs)
- **Base Service URL**: [https://heart-disease-fastapi-docker-idbe.onrender.com](https://heart-disease-fastapi-docker-idbe.onrender.com)
- **Health Check Endpoint**: [https://heart-disease-fastapi-docker-idbe.onrender.com/health](https://heart-disease-fastapi-docker-idbe.onrender.com/health)
- **Model Info Endpoint**: [https://heart-disease-fastapi-docker-idbe.onrender.com/info](https://heart-disease-fastapi-docker-idbe.onrender.com/info)

---

## 📋 Table of Contents
- [Live Deployment Links](#-live-deployment-links)
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset & Machine Learning Model](#dataset--machine-learning-model)
- [API Endpoints & Documentation](#api-endpoints--documentation)
- [Local Development & Testing](#local-development--testing)
  - [Option 1: Using Python Environment](#option-1-using-python-environment)
  - [Option 2: Using Docker Compose](#option-2-using-docker-compose)
- [Deployment on Render (Step-by-Step)](#deployment-on-render-step-by-step)
- [Sample curl Requests](#sample-curl-requests)

---

## 🚀 Project Overview

The objective of this project is to build and deploy an end-to-end machine learning web service:
1. **Model Training**: Train a binary classifier using Scikit-Learn on the Heart Disease dataset and persist it with Joblib.
2. **FastAPI Web Service**: Build robust REST API endpoints (`/health`, `/info`, `/predict`) with strict Pydantic input validation and interactive OpenAPI/Swagger documentation.
3. **Dockerization**: Containerize the app using a lightweight Python base image (`python:3.11-slim`) and provide multi-container orchestration via `docker-compose.yml`.
4. **Cloud Deployment**: Deploy the Docker container as a live web service on Render.

---

## 📂 Repository Structure

```text
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoint handlers
│   └── schemas.py           # Pydantic schemas for input features and response models
├── data/
│   └── heart.csv            # Heart Disease dataset
├── model/
│   └── heart_model.joblib   # Trained Scikit-Learn model artifact
├── Dockerfile               # Multi-platform production Dockerfile
├── docker-compose.yml       # Docker Compose service definition
├── requirements.txt         # Production Python dependencies
├── train.py                 # Script to train and save the model
└── README.md                # Comprehensive project documentation
```

---

## 🧠 Dataset & Machine Learning Model

The dataset is the standard **Heart Disease Dataset** (UCI Cleveland / Kaggle):
- **Target**: `target` (Binary: `1` = heart disease present, `0` = heart disease absent).
- **Features (13 clinical attributes)**:
  1. `age`: Age in years
  2. `sex`: Sex (`1` = male, `0` = female)
  3. `cp`: Chest pain type (`0`: typical angina, `1`: atypical angina, `2`: non-anginal pain, `3`: asymptomatic)
  4. `trestbps`: Resting blood pressure (in mm Hg on admission to the hospital)
  5. `chol`: Serum cholesterol in mg/dl
  6. `fbs`: Fasting blood sugar > 120 mg/dl (`1` = true, `0` = false)
  7. `restecg`: Resting electrocardiographic results (`0`, `1`, `2`)
  8. `thalach`: Maximum heart rate achieved
  9. `exang`: Exercise-induced angina (`1` = yes, `0` = no)
  10. `oldpeak`: ST depression induced by exercise relative to rest
  11. `slope`: Slope of the peak exercise ST segment (`0`, `1`, `2`)
  12. `ca`: Number of major vessels (0–4) colored by fluoroscopy
  13. `thal`: Thalassemia (`0` = normal, `1` = fixed defect, `2` = reversible defect, `3` = other)

### Model Training
The model is a `RandomForestClassifier` trained with a stratified train/test split.
To retrain the model at any time:
```bash
python train.py
```

---

## 🔌 API Endpoints & Documentation

Interactive API documentation (Swagger UI) is available at:
- **Live on Render**: [https://heart-disease-fastapi-docker-idbe.onrender.com/docs](https://heart-disease-fastapi-docker-idbe.onrender.com/docs)
- **Local**: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Redirects automatically to `/docs` |
| `GET` | `/health` | Health check endpoint returning `{"status": "ok"}` |
| `GET` | `/info` | Returns model architecture, accuracy, and required feature list |
| `POST` | `/predict` | Computes prediction: `{"heart_disease": bool, "prediction": 0/1, "probability": float}` |

### Sample Request (`POST /predict`)
```json
{
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
```

### Sample Response
```json
{
  "heart_disease": true,
  "prediction": 1,
  "probability": 0.62
}
```

---

## 💻 Local Development & Testing

### Option 1: Using Python Environment

1. **Clone the repository**:
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd <REPO_NAME>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI app with Uvicorn**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Visit Swagger UI**:
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

---

### Option 2: Using Docker & Docker Compose

1. **Build the Docker container**:
   ```bash
   docker-compose build
   ```

2. **Run the container**:
   ```bash
   docker-compose up
   ```
   *(Add `-d` flag to run in detached background mode)*

3. **Verify the container is running**:
   ```bash
   docker ps
   ```

4. **Access Swagger UI**:
   Open [http://localhost:8000/docs](http://localhost:8000/docs).

5. **Stop the container**:
   ```bash
   docker-compose down
   ```

---

## ☁️ Deployment on Render (Step-by-Step)

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Heart disease prediction FastAPI service with Docker"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
   git push -u origin main
   ```

2. **Log in to Render**:
   - Go to [https://render.com](https://render.com) and sign in (using GitHub).

3. **Create a New Web Service**:
   - Click **New +** > **Web Service**.
   - Select **Build and deploy from a Git repository**.
   - Connect your GitHub repository.

4. **Configure the Service**:
   - **Name**: `heart-disease-prediction-api` (or your preferred name)
   - **Region**: Choose the closest region (e.g., Oregon, Frankfurt, Singapore)
   - **Language / Environment**: Select **Docker**
   - **Branch**: `main`
   - **Root Directory**: Leave blank (root of repository)
   - **Instance Type**: **Free**

5. **Deploy**:
   - Click **Deploy Web Service**.
   - Render will automatically build the image using the `Dockerfile` and start the Uvicorn server binding to `0.0.0.0:$PORT`.
   - Monitor the **Logs** tab until you see `Uvicorn running on http://0.0.0.0:...` and the status becomes **Live**.

6. **Verify the Live Service**:
   - Open `<YOUR_RENDER_URL>/docs` to view Swagger UI.
   - Test endpoints with `curl` or Postman.

---

## 🧪 Sample curl Requests (Live API)

### 1. Health Check
```bash
curl -X GET "https://heart-disease-fastapi-docker-idbe.onrender.com/health"
```
**Expected Response:**
```json
{"status":"ok"}
```

### 2. Model Info
```bash
curl -X GET "https://heart-disease-fastapi-docker-idbe.onrender.com/info"
```

### 3. Predict (Positive / Disease Present)
```bash
curl -X POST "https://heart-disease-fastapi-docker-idbe.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### 4. Predict (Negative / Disease Absent)
```bash
curl -X POST "https://heart-disease-fastapi-docker-idbe.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 57,
    "sex": 1,
    "cp": 0,
    "trestbps": 140.0,
    "chol": 240.0,
    "fbs": 0,
    "restecg": 1,
    "thalach": 123.0,
    "exang": 1,
    "oldpeak": 0.2,
    "slope": 1,
    "ca": 0,
    "thal": 3
  }'
```
