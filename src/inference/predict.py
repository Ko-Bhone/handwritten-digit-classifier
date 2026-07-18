import torch

class DigitPredictor:
    def __init__(self,model,model_path):
        self.model = model
        self.model_path = model_path

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def predict(self,x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        return (prediction.item(),confidence.item())