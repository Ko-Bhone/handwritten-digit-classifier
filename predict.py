import random
from src.data.loader import DigitDataLoader
from src.models.neural_network import DigitClassifier
from src.inference.predict import DigitPredictor
from src.config import (DATA_PATH, MODEL_PATH)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



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
    print(f"Confidence Score: {confidence:.2%}")
    print("=" * 40)

    image = sample.squeeze().numpy().reshape(20,20).T
    plt.figure(figsize=(5,5))
    plt.imshow(image,cmap="gray")
    plt.title(f"Actual Digit: {actual}\n"
              f"Predicted Digit: {prediction}\n"
              f"Confidence Score: {confidence:.2%}",fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    save_path = Path("figures/prediction_result.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path,dpi=300)
    print(f"Prediction Figure Saved -> {save_path}")
    plt.show()

if __name__ == "__main__":
    main()


