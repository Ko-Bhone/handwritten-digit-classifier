import random
from src.data.loader import DigitDataloader
from src.models.neural_network import DigitClassifier
from src.inference.predict import DigitPredictor

Data_Path = "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat"
Model_Path = "models/digit_classifier.pth"

loader = DigitDataloader(Data_Path)
x,y = loader.load()
model = DigitClassifier()
predictor = DigitPredictor(model=model,model_path=Model_Path)
predictor.load_model()

index = random.randint(0,len(x) -1)
sample = x[index].unsqueeze(0)
actual = y[index].item()

prediction, confidence = predictor.predict(sample)

print("="*40)
print(f"Sample Index: {index}")
print(f"Actual Digit: {actual}")
print(f"Predicted Digit: {prediction}")
print(f"Confidence Score: {confidence:.4f}")
print("="*40)