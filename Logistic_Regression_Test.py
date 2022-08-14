import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from SupervisedLearning import LogisticRegression as SLLR
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X = data.data
y = data.target

# X, y = make_classification(n_features=4, n_classes=2)

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SLLR(X_train, X_test, Y_train)

loss = model.train()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Function vs Number of Iterations")
plt.plot(loss)
plt.show()
preds = model.predict(threshold=0.5)

print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))

lr_model = LogisticRegression()
lr_model.fit(X_train, Y_train)
lr_preds = lr_model.predict(X_test)

print(accuracy_score(y_test, lr_preds))
print(confusion_matrix(y_test, lr_preds))