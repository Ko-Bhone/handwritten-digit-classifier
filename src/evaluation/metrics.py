import torch

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
