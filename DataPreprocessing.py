import numpy as np

class DataPreprocessing():
    def __init__(self, X, Y, test_size=0.2):
        self.X = X
        self.Y = Y
        self.test_size = test_size

    # Please refer to the following link (first answer) to learn more about the function 
    # https://stackoverflow.com/questions/66079043/split-dataset-without-using-scikit-learn-train-test-split
    def train_test_split(self):
    
        i = int((1 - self.test_size) * self.X.shape[0]) 
        o = np.random.permutation(self.X.shape[0])
        
        X_train, X_test = np.split(np.take(self.X,o,axis=0), [i])
        Y_train, Y_test = np.split(np.take(self.Y,o), [i])

        return X_train, X_test, Y_train, Y_test