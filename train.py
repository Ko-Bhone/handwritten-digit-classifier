from pathlib import Path
from src.data.loader import DigitDataLoader
from src.data.preprocess import DataPreprocessor
from src.models.neural_network import DigitClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import ModelEvaluator
from src.config import (DATA_PATH,MODEL_PATH,LEARNING_RATE,EPOCHS)
from src.data.eda import DataAnalyzer
from predict import main as predict
from src.utils.logger import logger
import mlflow
import mlflow.pytorch



def main() -> None:
    mlflow.set_experiment("Handwritten Digit Classification")
    with mlflow.start_run():
        logger.info("Loading dataset...")
        loader = DigitDataLoader(DATA_PATH)
        x, y = loader.load()
        analyzer = DataAnalyzer(x, y)
        analyzer.dataset_info()
        analyzer.missing_values()
        analyzer.label_distribution()
        analyzer.show_random_samples()
        analyzer.pixel_statics()
        analyzer.average_digit_image()
        print("X Shape:", x.shape)
        print("Y Shape:", y.shape)

        logger.info("Splitting dataset...")
        x_train, x_val, x_test, y_train, y_val, y_test = DataPreprocessor.split_data(x, y)
        print("Train:", x_train.shape)
        print("Test:", x_test.shape)
        print("\nDataset Split")
        print(f"Train      : {len(x_train)}")
        print(f"Validation : {len(x_val)}")
        print(f"Test       : {len(x_test)}")
        model = DigitClassifier()

        mlflow.log_param("Learning Rate",LEARNING_RATE)
        mlflow.log_param("Epochs",EPOCHS)
        mlflow.log_param("optimizer","SGD")
        mlflow.log_param("loss_function","CrossEntropyLoss")

        logger.info("Training model...")
        trainer = Trainer(model=model, lr=LEARNING_RATE, epochs=EPOCHS)
        trainer.train(x_train, y_train, x_val, y_val)
        trainer.save_model(MODEL_PATH)

        logger.info("Evaluating model...")
        evaluator = ModelEvaluator()
        accuracy = evaluator.accuracy(model, x_test, y_test)
        print(f"Test Accuracy: {accuracy:.2f}%")

        prediction = trainer.predict(x_test[:5])
        print("Sample Prediction")
        for pred, actual in zip(prediction, y_test[:5].tolist()):
            print(f"Prediction:{pred} | Actual:{actual}")

        metrics = evaluator.calculate_metrics(model, x_test, y_test)
        mlflow.log_metrics(metrics)
        print("\n===== Evaluation Metrics =====")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1 Score  : {metrics['f1_score']:.4f}")
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        logger.info("Saving figures...")
        evaluator.plot_confusion_matrix(model, x_test, y_test)
        evaluator.plot_loss_curve(trainer.loss_history)
        logger.info("Training completed successfully.")

        predict()
        mlflow.log_artifact("figures/confusion_matrix.png")
        mlflow.log_artifact("figures/loss_curve.png")
        mlflow.log_artifact("figures/prediction_result.png")


if __name__ == "__main__":
    main()











