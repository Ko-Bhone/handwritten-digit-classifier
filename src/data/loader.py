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
        x = data["X"]
        y = data["y"].reshape(-1)
        y[y==10] = 0
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        return x,y
