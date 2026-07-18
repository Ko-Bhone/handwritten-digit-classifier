import torch
from scipy.io import loadmat
from pathlib import Path


class DigitDataLoader:
    def __init__(self,data_path:str):
        self.data_path = Path(data_path)


    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}")
        data = loadmat(self.data_path)
        x = torch.tensor(data["X"],dtype=torch.float32)
        y = data["y"].reshape(-1)
        y[y==10] = 0
        x = torch.tensor(x,dtype=torch.float32)
        y = torch.tensor(y,dtype=torch.long)
        return x,y
