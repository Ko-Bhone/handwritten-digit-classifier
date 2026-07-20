from fastapi import FastAPI
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

model = DigitClassifier()
predictor = DigitPredictor(model=model,model_path=MODEL_PATH)
predictor.load_model()

@app.get("/")
def home():
    return{
        "Message":"Welcome to Handwritten Digit Classifier API"
    }

@app.post("/predict",response_model=PredictionResponse)
def predict(request: PredictionRequest):
    image = torch.tensor(request.pixels,dtype=torch.float32).unsqueeze(0)
    prediction, confidence = predictor.predict(image)
    return PredictionResponse(prediction=prediction,confidence=confidence)