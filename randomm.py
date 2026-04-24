import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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

    # df = pd.DataFrame(X, columns=["CGPA","IQ","Internships"])
    # df["Placed"] = y

# ---------------- MODEL ----------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X, y)

# ---------------- PREDICTION ----------------
if use_csv:
    new_student = pd.DataFrame([[8.5, 120, 2]], columns=["CGPA","IQ","Internships"])
else:
    new_student = np.array([[8.5, 120, 2]])

print("Placed:", rf.predict(new_student))

# ---------------- GRAPH (same style) ----------------
cgpa = X[:, 0]
iq = X[:, 1]

for i in range(len(cgpa)):
    if y[i] == 0:
        plt.scatter(cgpa[i], iq[i], color='red')
    else:
        plt.scatter(cgpa[i], iq[i], color='green')

plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.title("Random Forest Data (CGPA vs IQ)")
plt.show()

# ---------------- CONFUSION MATRIX ----------------
y_pred = rf.predict(X)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

# ---------------- METRICS ----------------
print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))







# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# # ---------------- SWITCH ----------------
# use_csv = False   # True → use data.csv | False → use arrays

# # ---------------- DATA ----------------
# if use_csv:
#     df = pd.read_csv("data.csv")
#     X = df[["CGPA", "IQ", "Internships"]]
#     y = df["Salary"]   # continuous output
# else:
#     X = np.array([
#         [6.5, 100, 0],
#         [7.0, 105, 1],
#         [7.5, 110, 1],
#         [8.0, 115, 2],
#         [8.5, 120, 2],
#         [9.0, 125, 3],
#         [6.0, 95, 0],
#         [7.2, 108, 1],
#         [8.3, 118, 2],
#         [9.1, 130, 3]
#     ])

#     # continuous target (salary in lakhs)
#     y = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 2.8, 3.8, 4.8, 5.8])

#     df = pd.DataFrame(X, columns=["CGPA","IQ","Internships"])
#     df["Salary"] = y

# # ---------------- MODEL ----------------
# rf = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42
# )

# rf.fit(X, y)

# # ---------------- PREDICTION ----------------
# if use_csv:
#     new_student = pd.DataFrame([[8.5, 120, 2]], columns=["CGPA","IQ","Internships"])
# else:
#     new_student = np.array([[8.5, 120, 2]])

# print("Predicted Salary:", rf.predict(new_student))

# # ---------------- GRAPH (Actual vs Predicted) ----------------
# y_pred = rf.predict(X)

# plt.scatter(y, y_pred)
# plt.plot([y.min(), y.max()], [y.min(), y.max()])

# plt.xlabel("Actual Salary")
# plt.ylabel("Predicted Salary")
# plt.title("Random Forest Regression")

# plt.show()

# # ---------------- METRICS ----------------
# print("MSE:", mean_squared_error(y, y_pred))
# print("R2 Score:", r2_score(y, y_pred))