from fastapi import FastAPI, HTTPException
from src.models.neural_network import DigitClassifier
from src.inference.predict import DigitPredictor
from src.config import MODEL_PATH
from src.api.schemas import PredictionRequest, PredictionResponse
import torch

app = FastAPI(
    title="Handwritten Digit Classifier",
    description="API for prediction handwritten digits using a PyTorch model.",
    version="1.0"
)

try:
    model = DigitClassifier()
    predictor = DigitPredictor(model=model,model_path=MODEL_PATH)
    predictor.load_model()
except Exception as e:
    predictor = None
    print("Model Loading Error:{}".format(e))

@app.get("/")
def home():
    return{
        "Message":"Welcome to Handwritten Digit Classifier API"}

@app.get("/health")
def health():
    return {
        "status":"healthy",
        "model_loaded": predictor is not None}

@app.post("/predict",response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=500,
                            detail="Model is not Loaded.")
    try:
        image = torch.tensor(request.pixels,dtype=torch.float32).unsqueeze(0)
        prediction,confidence = predictor.predict(image)
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Prediction Failed:{str(e)}")

