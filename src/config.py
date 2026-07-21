from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

DATA_PATH = ("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat")
MODEL_PATH = ("./models/digit_classifier.pth")
LEARNING_RATE = 0.1
EPOCHS = 3000
RANDOM_STATE = 42
