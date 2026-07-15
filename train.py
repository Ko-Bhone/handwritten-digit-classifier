from src.data.loader import DigitDataloader
from src.data.preprocess import Datapreprocess

Data_Path = "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex3data1.mat"

loader = DigitDataloader(Data_Path)
x,y = loader.load()
print("X Shape:",x.shape)
print("Y Shape:",y.shape)
processor = Datapreprocess()
x_train,x_test,y_train,y_test = processor.split(x,y)

print("Train:",x_train.shape)
print("Test:",x_test.shape)