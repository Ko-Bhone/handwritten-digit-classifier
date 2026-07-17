import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

class ModelEvaluator:


    def accuracy(self,model,x_test,y_test):
        model.eval()
        with torch.no_grad():
            output = model(x_test)
            prediction = torch.argmax(output,dim=1)
            correct = (prediction == y_test).sum().item()
            total = y_test.size(0)
            accuracy = (correct / total) * 100
            return accuracy

    def calculate_metrics(self,model,x_test,y_test):
        model.eval()
        with torch.no_grad():
            output = model(x_test)
            predictions = torch.argmax(output,dim=1)
        y_true = y_test.numpy()
        y_pred = predictions.numpy()
        metric = {"accuracy":accuracy_score(y_true,y_pred),
                  "precision":precision_score(y_true,y_pred,average="macro"),
                  "recall":recall_score(y_true,y_pred,average="macro"),
                  "f1_score":f1_score(y_true,y_pred,average="macro")}

        return metric
