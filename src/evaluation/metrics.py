import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import seaborn as sns

class ModelEvaluator:

    def get_predictions(self,model,x_test):
        model.eval()
        with torch.no_grad():
            output = model(x_test)
            predictions = torch.argmax(output,dim=1)
        return predictions

    def accuracy(self,model,x_test,y_test):
        predictions = self.get_predictions(model,x_test)
        correct = (predictions == y_test).sum().item()
        total = y_test.size(0)
        accuracy = (correct / total) * 100
        return accuracy

    def calculate_metrics(self,model,x_test,y_test):
        predictions = self.get_predictions(model,x_test)
        y_true = y_test.numpy()
        y_pred = predictions.numpy()
        metric = {"accuracy":accuracy_score(y_true,y_pred),
                  "precision":precision_score(y_true,y_pred,average="macro"),
                  "recall":recall_score(y_true,y_pred,average="macro"),
                  "f1_score":f1_score(y_true,y_pred,average="macro")}

        return metric

    def plot_confusion_matrix(self,model,x_test,y_test,save_path="figures/confusion_matrix.png"):
        predictions = self.get_predictions(model,x_test)
        cm = confusion_matrix(y_test.numpy(),predictions.numpy())
        display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=range(10))
        fig, ax = plt.subplots(figsize=(8,8))
        display.plot(cmap="Blues",ax=ax,colorbar=True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        plt.savefig(save_path,dpi=300)
        plt.show()
        print(f"Confusion matrix Saved -> {save_path}")

    def plot_loss_curve(self,loss_history,save_path="figures/loss_curve.png"):
        plt.figure(figsize=(8,5))
        plt.plot(loss_history,linewidth=2)
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        plt.savefig(save_path,dpi=300)
        plt.show()
        print(f"Loss Curve Saved -> {save_path}")