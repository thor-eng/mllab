import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---------------- SWITCH ----------------
use_csv = False   # True → use data.csv | False → use arrays

# ---------------- DATA ----------------
if use_csv:
    df = pd.read_csv("data.csv")
    X = df[["CGPA", "IQ", "Internships"]]
    y = df["Placed"]
else:
    X = np.array([
        [6.5, 100, 0],
        [7.0, 105, 1],
        [7.5, 110, 1],
        [8.0, 115, 2],
        [8.5, 120, 2],
        [9.0, 125, 3],
        [6.0, 95, 0],
        [7.2, 108, 1],
        [8.3, 118, 2],
        [9.1, 130, 3]
    ])
    y = np.array([0,0,1,1,1,1,0,1,1,1])

    df = pd.DataFrame(X, columns=["CGPA","IQ","Internships"])
    df["Placed"] = y

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- MODEL ----------------
model = MLPClassifier(
    hidden_layer_sizes=(5,5),
    max_iter=1000,
    random_state=42
)

model.fit(X_scaled, y)

# ---------------- PREDICTION ----------------
if use_csv:
    new_student = pd.DataFrame([[8.5, 120, 2]], columns=["CGPA","IQ","Internships"])
else:
    new_student = np.array([[8.5, 120, 2]])

new_scaled = scaler.transform(new_student)
print("Placed:", model.predict(new_scaled))

# ---------------- GRAPH (2D projection) ----------------
cgpa = X[:, 0]
iq = X[:, 1]

for i in range(len(cgpa)):
    if y[i] == 0:
        plt.scatter(cgpa[i], iq[i], color='red')
    else:
        plt.scatter(cgpa[i], iq[i], color='green')

plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.title("MLP Data (CGPA vs IQ)")
plt.show()

# ---------------- CONFUSION MATRIX ----------------
y_pred = model.predict(X_scaled)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

# ---------------- METRICS ----------------
print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))