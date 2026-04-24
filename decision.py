import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---------------- SWITCH ----------------
use_csv = False

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

# ---------------- MODEL ----------------
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X, y)

# ---------------- PREDICTION ----------------
if use_csv:
    new_student = pd.DataFrame([[8.2, 115, 2]], columns=["CGPA","IQ","Internships"])
else:
    new_student = np.array([[8.2, 115, 2]])

print("Placed:", model.predict(new_student))

# ---------------- GRAPH ----------------
# cgpa = X[:, 0]
# iq = X[:, 1]

# for i in range(len(cgpa)):
#     if y[i] == 0:
#         plt.scatter(cgpa[i], iq[i], color='red')
#     else:
#         plt.scatter(cgpa[i], iq[i], color='green')

# plt.xlabel("CGPA")
# plt.ylabel("IQ")
# plt.title("Placement (CGPA vs IQ)")
# plt.show() 


cgpa = X[:, 0]
iq = X[:, 1]

x_min, x_max = cgpa.min()-1, cgpa.max()+1
y_min, y_max = iq.min()-5, iq.max()+5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

# fix internships = 1 (better than 0)
zz = np.ones_like(xx.ravel())

Z = model.predict(np.c_[xx.ravel(), yy.ravel(), zz])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)

for i in range(len(cgpa)):
    if y[i] == 0:
        plt.scatter(cgpa[i], iq[i], color='red')
    else:
        plt.scatter(cgpa[i], iq[i], color='green')

plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.title("Decision Tree Boundary (Internships fixed)")
plt.show()



# ---------------- CONFUSION MATRIX ----------------
y_pred = model.predict(X)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

# ---------------- METRICS ----------------
print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("F1 Score:", f1_score(y, y_pred))










# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# # ---------------- SWITCH ----------------
# use_csv = False

# # ---------------- DATA ----------------
# if use_csv:
#     df = pd.read_csv("data.csv")
#     X = df[["CGPA", "IQ", "Internships"]]
#     y = df["Salary"]   # continuous value
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

#     # continuous output (salary in lakhs)
#     y = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 2.8, 3.8, 4.8, 5.8])

#     df = pd.DataFrame(X, columns=["CGPA","IQ","Internships"])
#     df["Salary"] = y

# # ---------------- MODEL ----------------
# model = DecisionTreeRegressor(
#     max_depth=4,
#     min_samples_split=2,
#     random_state=42
# )

# model.fit(X, y)

# # ---------------- PREDICTION ----------------
# if use_csv:
#     new_student = pd.DataFrame([[8.2, 115, 2]], columns=["CGPA","IQ","Internships"])
# else:
#     new_student = np.array([[8.2, 115, 2]])

# print("Predicted Salary:", model.predict(new_student))

# # ---------------- GRAPH (Actual vs Predicted) ----------------
# # y_pred = model.predict(X)

# # plt.scatter(y, y_pred)
# # plt.plot([y.min(), y.max()], [y.min(), y.max()])

# # plt.xlabel("Actual Salary")
# # plt.ylabel("Predicted Salary")
# # plt.title("Decision Tree Regression")

# # plt.show()

# import matplotlib.pyplot as plt

# plt.hist(y, bins=8)

# plt.xlabel("Salary")   # or your target name
# plt.ylabel("Frequency")
# plt.title("Distribution of Target Variable")

# plt.show()

# # ---------------- METRICS ----------------
# print("MSE:", mean_squared_error(y, y_pred))
# print("R2 Score:", r2_score(y, y_pred))







# # graph


# # from mpl_toolkits.mplot3d import Axes3D
# # import matplotlib.pyplot as plt

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')

# # # color using y (Placed)
# # colors = ['red' if val == 0 else 'green' for val in y]

# # ax.scatter(X[:,0], X[:,1], X[:,2], c=colors)

# # ax.set_xlabel("CGPA")
# # ax.set_ylabel("IQ")
# # ax.set_zlabel("Internships")

# # plt.show()