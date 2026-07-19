import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

class Trainer:
    def __init__(self, model:nn.Module, lr:float=0.1, epochs:int=300):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train(self, x_train:torch.Tensor,
              y_train: torch.Tensor,
              x_val: torch.Tensor,
              y_val: torch.Tensor):
        for epoch in range(self.epochs):
            self.model.train()
            outputs = self.model(x_train)
            loss = self.criterion(outputs, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
            val_loss = self.validate(x_val, y_val)
            self.val_loss_history.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    key: value.clone()
                    for key, value in self.model.state_dict().items()
                }

            if epoch % 50 == 0:
                print(f"""
                        Epoch : {epoch}
                        Train Loss : {loss.item():.4f}
                        Val Loss   : {val_loss:.4f}""")
        print("Training loop Finished")


    def validate(self, x_val: torch.Tensor,
                 y_val: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_val)
            loss = self.criterion(outputs, y_val)
        return loss.item()

    def save_model(self, path:str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.best_model_state is None:
            raise RuntimeError(
                "No trained model found. Run train() first."
            )
        torch.save(self.best_model_state, path)

    def predict(self, x: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            predictions = torch.argmax(outputs, dim=1)
        return predictions