import torch.nn as nn

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(400,25),nn.Sigmoid(),nn.Linear(25,10))

    def forward(self,x):
        return self.network(x)