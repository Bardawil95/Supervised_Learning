import pandas as pd, matplotlib.pyplot as plt, numpy as np
from DataPreprocessing import DataPreprocessing
from SupervisedLearning import LinearRegressor

# Importing dataset
df = pd.read_csv("Salary_Data.csv")

X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

# Splitting dataset into train and test set

dp = DataPreprocessing(X, Y, test_size=0.3)

X_train, X_test, Y_train, Y_test = dp.train_test_split()

model = LinearRegressor(
    X_train, Y_train, num_of_iterations=1000, learning_rate=0.01)

model.train()

# Prediction on test set

Y_pred = model.predict(X_test)

print("Predicted values ", np.round(Y_pred[:3], 2))
print("Real values	 ", Y_test[:3])

print("Trained W	 ", round(model.W[0], 2))
print("Trained b	 ", round(model.b, 2))

# Visualization on test set

plt.scatter(X_test, Y_test, color='blue')

plt.plot(X_test, Y_pred, color='orange')

plt.title('Salary vs Experience')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
