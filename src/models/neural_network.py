import torch
import torch.nn as nn

INPUT_SIZE = 400
HIDDEN_SIZE = 25
OUTPUT_SIZE = 10

class DigitClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(nn.Linear(INPUT_SIZE,HIDDEN_SIZE),
                                    nn.Sigmoid(),
                                     nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.network(x)