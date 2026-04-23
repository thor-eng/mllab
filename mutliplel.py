import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- SWITCH ----------------
use_csv = False   # True → use data.csv | False → use arrays

# ---------------- DATA ----------------
if use_csv:
    df = pd.read_csv("data.csv")
    X = df[["Area", "Bedrooms", "Age"]]
    y = df["Price"]
else:
    data = {
        "Area": [800, 1000, 1200, 1500, 1800],
        "Bedrooms": [2, 2, 3, 3, 4],
        "Age": [20, 15, 10, 8, 5],
        "Price": [40, 50, 65, 80, 100]
    }
    df = pd.DataFrame(data)
    X = df[["Area", "Bedrooms", "Age"]]
    y = df["Price"]

# ---------------- MODEL ----------------
model = LinearRegression()
model.fit(X, y)

# ---------------- PREDICTION ----------------
if use_csv:
    new_house = pd.DataFrame([[1400, 3, 7]], columns=["Area","Bedrooms","Age"])
else:
    new_house = np.array([[1400, 3, 7]])

predicted_price = model.predict(new_house)
print("Predicted House Price (in lakhs):", predicted_price)

# ---------------- GRAPH ----------------
y_pred = model.predict(X)

plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()])

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")

plt.show()