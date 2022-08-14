import numpy as np
# from math import pow, sqrt
import math
# from Test_Linear_Regression import Y_pred, Y_train

# from Logr import standardize

# from Logistic_Regression_Test import X


class MathOps():

    def mean(self, vector):
        mean = 0.0
        for i in range(len(vector)):
            mean += vector[i]
        return mean/len(vector)

    def stdv(self, vector):
        vector_mean_diff_sum = 0.0
        for i in range(len(vector)):
            vector_mean_diff_sum += ((vector[i] - self.mean(vector))**2.0)
        return (vector_mean_diff_sum/len(vector))**(0.5)

    def log(self, a, b=2):
        if b == 2:
            b = self.exp(1)
        if a < b:
            return 0
        return 1 + self.log(a/b, b)

    def exp(self, num):
        return 2.7182818284590452353602874713527**(num)


################################################################   Linear Regression  ################################################################


class LinearRegressor():

    def __init__(self, X_train, Y_train, num_of_iterations=500, learning_rate=0.001) -> None:
        self.num_of_iterations = num_of_iterations
        self.learning_rate = learning_rate

        self.m, self.n = X_train.shape
        self.b = 0
        self.W = np.zeros(self.n)
        self.X = X_train
        self.Y_train = Y_train
        self.mth = MathOps()

    def train(self):
        for i in range(self.num_of_iterations):
            Y_pred = self.predict(self.X)

            # calculate gradients
            dW = - (2 * (self.X.T).dot(self.Y_train - Y_pred)) / self.m
            db = - 2 * np.sum(self.Y_train - Y_pred) / self.m

            # update weights
            self.W = self.W - self.learning_rate * dW
            self.b = self.b - self.learning_rate * db

        return self

    def predict(self, X):
        return (X.dot(self.W)+self.b)


################################################################   Logistic Regression   ################################################################


class LogisticRegression:

    def __init__(self, X_train, X_test, Y_train, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train

        self.weights = np.zeros(X_train.shape[1])
        self.bias = 0

    # @staticmethod
    def standardize(self, X):
        for i in range(np.shape(X)[1]):
            X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.std(X[:, i])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, h, y):
        return ((-y * np.log(h) - (1 - y) * np.log(1 - h))/len(y)).mean()

    def train(self):
        # Perform gradient descent
        iteration = 0
        loss = []

        # Standardize all features for training in vector X
        self.standardize(self.X_train)

        while iteration < self.n_iterations:
            linear_pred = np.dot(self.X_train, self.weights) + self.bias
            probability = self._sigmoid(linear_pred)

            # Calculate derivatives
            dW = (1 / self.X_train.shape[0]) * (2 *
                                                np.dot(self.X_train.T, (probability - self.Y_train)))
            db = (1 / self.X_train.shape[0]) * \
                (2 * np.sum(probability - self.Y_train))

            # Update the coefficients
            self.weights = self.weights - self.learning_rate * dW
            self.bias = self.bias - self.learning_rate * db

            loss.append(self.loss(probability, self.Y_train))
            print("Iteration ", iteration, "Loss ------> ", loss[iteration])
            iteration += 1
        return loss

    def predict(self, threshold=0.5):
        # Standardize all features for training in vector X
        self.standardize(self.X_test)

        y_pred = np.dot(self.X_test, self.weights) + self.bias
        probabs = self._sigmoid(y_pred)
        results = []
        for probabs in y_pred:
            if probabs > threshold:
                results.append(1)
            else:
                results.append(0)
        return results

    def score(self, y1, y2):  # y1 is the correct answers
        # y2 is calculated by the model
        y1 = y1.reshape(y1.shape[0], 1)
        y2 = y2.reshape(y2.shape[0], 1)
        y1_not = (1 - y1).reshape(y1.shape[0], 1)
        y2_not = (1 - y2).reshape(y1.shape[0], 1)
        a = np.multiply(y1_not, y2_not) + np.multiply(y1, y2)
        # 1 means  correct prediction, 0 means wrong prediction

        ones_ = np.count_nonzero(a == 1)  # count ones to get the percentage
        return (ones_ / y1.shape[0]) * 100


################################################################   K-Nearest Neighbors  ################################################################

class KNN():
    def __init__(self, neighbors=3) -> None:
        self.K = neighbors

    def train(self, X_train):
        self.X = X_train

    def euc_dist(self, X_test_i, X_Train_i):
        # Condition 1: Training Dataset holds one feature, Condition 2: Training Dataset holds 2 features
        return (((X_Train_i - X_test_i)**2)**0.5)

    def predict(self, X_test):
        neighbors_list = []
        for x_test in X_test:
            dist = []
            count_test_loop+=1
            for x in self.X:
                dist.append(math.sqrt(sum([math.pow((x_i - x_test_i), 2) for x_i, x_test_i in zip(x, x_test)])))
            sorted_index = np.argsort(dist)
            neighbors_list.append(sorted_index[0:self.K])
        return neighbors_list
