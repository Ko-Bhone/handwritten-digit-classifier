import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

    def plot_confusion_matrix(self,model,x_test,y_test):
        predictions = self.get_predictions(model,x_test)
        cm = confusion_matrix(y_test.numpy(),predictions.numpy())
        plt.figure(figsize=(8,6))
        plt.imshow(cm,interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.show()
