from pathlib import Path
from src.data.loader import DigitDataLoader
from src.data.preprocess import DataPreprocessor
from src.models.neural_network import DigitClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import ModelEvaluator
from src.config import (DATA_PATH,MODEL_PATH,LEARNING_RATE,EPOCHS)
from src.data.eda import DataAnalyzer



def main() -> None:
    loader = DigitDataLoader(DATA_PATH)
    x,y = loader.load()
    analyzer = DataAnalyzer(x, y)
    analyzer.dataset_info()
    analyzer.missing_values()
    analyzer.label_distribution()
    print("X Shape:", x.shape)
    print("Y Shape:", y.shape)
    x_train, x_val, x_test, y_train, y_val, y_test = DataPreprocessor.split_data(x, y)

    model = DigitClassifier()
    print("Train:", x_train.shape)
    print("Test:", x_test.shape)
    print("\nDataset Split")
    print(f"Train      : {len(x_train)}")
    print(f"Validation : {len(x_val)}")
    print(f"Test       : {len(x_test)}")

    trainer = Trainer(model=model, lr=LEARNING_RATE, epochs=EPOCHS)
    trainer.train(x_train, y_train, x_val, y_val)
    trainer.save_model(MODEL_PATH)

    evaluator = ModelEvaluator()
    accuracy = evaluator.accuracy(model, x_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}%")

    prediction = trainer.predict(x_test[:5])
    print("Sample Prediction")
    for pred, actual in zip(prediction, y_test[:5].tolist()):
        print(f"Prediction:{pred} | Actual:{actual}")

    metrics = evaluator.calculate_metrics(model, x_test, y_test)
    print("\n===== Evaluation Metrics =====")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")

    evaluator.plot_confusion_matrix(model, x_test, y_test)
    evaluator.plot_loss_curve(trainer.loss_history)

if __name__ == "__main__":
    main()











