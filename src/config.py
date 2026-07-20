from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat")
MODEL_DIR = (BASE_DIR/ "models")
MODEL_PATH = ("./models/digit_classifier.pth")
LEARNING_RATE = 0.1
EPOCHS = 3000