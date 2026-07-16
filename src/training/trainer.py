import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self,model,lr=0.1,epochs=300):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def train(self,x_train,y_train):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(),lr=self.lr)
        self.model.train()

        for epoch in range(self.epochs):
            outputs = self.model(x_train)
            loss = criterion(outputs,y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        print("Finished Training")

    def save_model(self,save_path):
        torch.save(self.model.state_dict(),save_path)
        print(f"Model saved -> {save_path}")

    def predict(self,x):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            prediction = torch.argmax(output,dim=1)
        return prediction