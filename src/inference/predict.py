import torch
from pathlib import Path
import torch.nn.functional as F


class DigitPredictor:
    def __init__(self,model,model_path:str | Path):
        self.model = model
        self.model_path = Path(model_path)

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}")
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def predict(self,x: torch.Tensor) -> tuple[int,float]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        return (prediction.item(),confidence.item())