import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- SWITCH ----------------
use_csv = False   # True → use data.csv | False → use arrays

# ---------------- DATA ----------------
if use_csv:
    df = pd.read_csv("data.csv")
    X = df[["Experience"]]
    y = df["Salary"]
else:
    X = np.array([1,2,3,4,5]).reshape(-1,1)
    y = np.array([30000,35000,40000,45000,50000])
    df = pd.DataFrame({"Experience": X.flatten(), "Salary": y})

# ---------------- MODEL ----------------
model = LinearRegression()
model.fit(X, y)

# ---------------- PREDICTION ----------------
if use_csv:
    new_value = pd.DataFrame([[3]], columns=["Experience"])
else:
    new_value = np.array([[3]])

predicted = model.predict(new_value)
print("Predicted Salary:", predicted)

# ---------------- GRAPH ----------------
y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")

plt.show()