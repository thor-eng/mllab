import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---------------- SWITCH ----------------
use_csv = False   # True → use data.csv | False → use arrays

# ---------------- DATA ----------------
if use_csv:
    df = pd.read_csv("data.csv")
    X = df[["Hours"]]
    y = df["Pass"]
else:
    X = np.array([1,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8]).reshape(-1,1)
    y = np.array([0,0,0,0,0,0,1,1,1,1,1,1,1,1])

# ---------------- MODEL ----------------
model = LogisticRegression()
model.fit(X, y)

# ---------------- PREDICTION ----------------
if use_csv:
    new_value = pd.DataFrame([[5]], columns=["Hours"])
else:
    new_value = np.array([[5]])

predicted = model.predict(new_value)
print("Predicted Class:", predicted)

# ---------------- GRAPH ----------------
plt.scatter(X, y)

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_prob = model.predict_proba(X_range)[:,1]
# y_prob = model.predict_proba(X)[:,1]  or
# X_range = np.linspace(min(X), max(X), 100).reshape(-1,1)
# y_prob = model.predict_proba(X_range)[:,1]

plt.plot(X_range, y_prob)
plt.xlabel("Hours")
plt.ylabel("Probability of Pass")
plt.title("Logistic Regression Curve")
plt.show()

# ---------------- CONFUSION MATRIX ----------------
y_pred = model.predict(X)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

# ---------------- METRICS ----------------
print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))