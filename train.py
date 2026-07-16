from src.data.loader import DigitDataloader
from src.data.preprocess import Datapreprocess
from src.models.neural_network import DigitClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import ModelEvaluator

Data_Path = "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat"

loader = DigitDataloader(Data_Path)
x,y = loader.load()
print("X Shape:",x.shape)
print("Y Shape:",y.shape)
processor = Datapreprocess()
x_train,x_test,y_train,y_test = processor.split(x,y)

print("Train:",x_train.shape)
print("Test:",x_test.shape)

model = DigitClassifier()
trainer = Trainer(model=model,lr=0.1,epochs=300)
trainer.train(x_train,y_train)
trainer.save_model("./models/digit_classifier")

evaluator = ModelEvaluator()
accuracy = evaluator.accuracy(model,x_test,y_test)
print(f"Test Accuracy: {accuracy:.2f}%")

prediction = trainer.predict(x_test[:5])
print("Prediction")
print("Actual")
print(y_test[:5])