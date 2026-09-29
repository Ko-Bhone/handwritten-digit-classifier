import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

class ModelEvaluator:

    @staticmethod
    def get_predictions(model:torch.nn.Module, x_test: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            output = model(x_test)
            predictions = torch.argmax(output,dim=1)
        return predictions

    @staticmethod
    def accuracy(model:torch.nn.Module, x_test: torch.Tensor, y_test: torch.Tensor) -> float:
        predictions = ModelEvaluator.get_predictions(model,x_test)
        correct = (predictions == y_test).sum().item()
        total = y_test.size(0)
        if total == 0:
            raise ValueError("Test Dataset cannot be empty")
        return (correct / total) * 100

    @staticmethod
    def calculate_metrics(model: torch.nn.Module, x_test:torch.Tensor, y_test:torch.Tensor) -> dict[str, float]:
        predictions = ModelEvaluator.get_predictions(model,x_test)
        y_true = y_test.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        metric = {"accuracy":accuracy_score(y_true, y_pred),
                  "precision":precision_score(y_true, y_pred, average="macro", zero_division=0),
                  "recall":recall_score(y_true, y_pred, average="macro", zero_division=0),
                  "f1_score":f1_score(y_true, y_pred, average="macro", zero_division=0)}

        return metric

    @staticmethod
    def plot_confusion_matrix(model:torch.nn.Module, x_test:torch.Tensor,
                              y_test:torch.Tensor, save_path:str | Path="figures/confusion_matrix.png"):
        predictions = ModelEvaluator.get_predictions(model,x_test)
        y_true = y_test.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)

        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
        fig, ax = plt.subplots(figsize=(8,8))
        display.plot(cmap="Blues",ax=ax,colorbar=True)
        ax.set_title("Confusion Matrix")

        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Confusion matrix Saved -> {save_path}")

    @staticmethod
    def plot_loss_curve(loss_history:list[float],
                        val_loss_history: list[float] | None = None,
                        save_path: str | Path ="figures/loss_curve.png") -> None:

        if not loss_history:
            raise ValueError("loss_history is empty")
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(loss_history, linewidth=2, label="Training Loss")

        if val_loss_history is not None:
            if len(val_loss_history) != len(loss_history):
                raise ValueError("val_loss_history and loss_history must have same length")
            ax.plot(val_loss_history, linewidth=2, label="Validation Loss")

        ax.set_title("Training & Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Loss Curve Saved -> {save_path}")
