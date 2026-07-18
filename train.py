from src.data.loader import DigitDataloader
from src.data.preprocess import DataPreprocessor
from src.models.neural_network import DigitClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import ModelEvaluator


Data_Path = "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat"
loader = DigitDataloader(Data_Path)
x,y = loader.load()
print("X Shape:",x.shape)
print("Y Shape:",y.shape)
processor = DataPreprocessor()
x_train, x_val, x_test, y_train, y_val, y_test = processor.split_data(x,y)
model = DigitClassifier()

print("Train:",x_train.shape)
print("Test:",x_test.shape)
print("\nDataset Split")
print(f"Train      : {len(x_train)}")
print(f"Validation : {len(x_val)}")
print(f"Test       : {len(x_test)}")

trainer = Trainer(model=model,lr=0.1,epochs=300)
trainer.train(x_train,y_train,x_val,y_val)
trainer.save_model("./models/digit_classifier.pth")

evaluator = ModelEvaluator()
accuracy = evaluator.accuracy(model,x_test,y_test)
print(f"Test Accuracy: {accuracy:.2f}%")

prediction = trainer.predict(x_test[:5])
print("Prediction",prediction)
print("Actual")
print(y_test[:5])

metrics = evaluator.calculate_metrics(model, x_test, y_test)
print("\n===== Evaluation Metrics =====")
print(f"Accuracy  : {metrics['accuracy']:.4f}")
print(f"Precision : {metrics['precision']:.4f}")
print(f"Recall    : {metrics['recall']:.4f}")
print(f"F1 Score  : {metrics['f1_score']:.4f}")

evaluator.plot_confusion_matrix(model,x_test,y_test)
evaluator.plot_loss_curve(trainer.loss_history)