from sklearn.model_selection import train_test_split


class DataPreprocessor:
    @staticmethod
    def split_data(self, x, y, train_size=0.70, validation_size=0.15, test_size=0.15, random_state=42):
        x_train, x_temp, y_train, y_temp = train_test_split(x, y,
                                                test_size=(1 - train_size),
                                                random_state=random_state,
                                                stratify=y)

        validation_ratio = validation_size / (validation_size + test_size)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=validation_ratio, random_state=random_state, stratify=y_temp)
        return (x_train, x_val, x_test, y_train, y_val, y_test)