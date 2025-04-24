import numpy as np
from model import SimpleMLP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

i_dim = 30
h_dim = 32
o_dim = 1
loss = "binary"

model = SimpleMLP(i_dim=i_dim, h_dim=h_dim, o_dim=o_dim, n_iters=n_iters, loss=loss)
model.train(X_train, y_train)

y_pred = model.forward(X_test)

for inp, pred, real in zip(X_test, y_pred, y_test):
    print(f"Model predict: {pred}, Real class: {real}")

print("\nReport:")
print(f"Accuracy: {accuracy_score(y_pred, y_test)}")
print(f"Precision: {precision_score(y_pred, y_test)}")
print(f"Recall: {recall_score(y_pred, y_test)}")
print(f"F1-Score: {f1_score(y_pred, y_test)}")