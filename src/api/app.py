from fastapi import FastAPI
from src.models.neural_network import DigitClassifier
from src.inference.predict import DigitPredictor
from src.config import MODEL_PATH

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