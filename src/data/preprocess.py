from sklearn.model_selection import train_test_split

class Datapreprocess:


    def split(self,x,y,test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=42,stratify=y)
        return x_train, x_test, y_train, y_test