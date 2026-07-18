import random
from src.data.loader import DigitDataLoader
from src.models.neural_network import DigitClassifier
from src.inference.predict import DigitPredictor
from src.config import (DATA_PATH, MODEL_PATH)

def main():
    loader = DigitDataLoader(DATA_PATH)
    x,y = loader.load()
    model = DigitClassifier()
    predictor = DigitPredictor(model=model,model_path=MODEL_PATH)
    predictor.load_model()

    index = random.randint(0,len(x)-1)
    sample = x[index].unsqueeze(0)
    actual = y[index].item()
    prediction, confidence = predictor.predict(sample)

    print("=" * 40)
    print(f"Sample Index: {index}")
    print(f"Actual Digit: {actual}")
    print(f"Predicted Digit: {prediction}")
    print(f"Confidence Score: {confidence:.4f}")
    print("=" * 40)


