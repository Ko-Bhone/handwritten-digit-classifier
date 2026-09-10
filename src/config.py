from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

DATA_PATH = ("/Users/macbookpro/Downloads/Telegram Desktop/data/ex3data1.mat")
MODEL_PATH = ("./models/digit_classifier.pth")
LEARNING_RATE = 0.1
EPOCHS = 3000
RANDOM_STATE = 42
