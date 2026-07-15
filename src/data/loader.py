import torch
from scipy.io import loadmat


class DigitDataloader:
    def __init__(self,data_path):
        self.data_path = data_path


    def load(self):
        data = loadmat(self.data_path)
        x = data["X"]
        y = data["y"].reshape(-1)
        y[y==10] = 0
        x = torch.tensor(x,dtype=torch.float)
        y = torch.tensor(y,dtype=torch.log)
        return x,y
