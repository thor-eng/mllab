import pandas as pd

# load dataset
df = pd.read_csv("data.csv")

# separate features and target
X = df.drop("PlayTennis", axis=1).values
y = df["PlayTennis"].values

# initialize hypothesis
h = ['0'] * len(X[0])

# ---------------- FIND-S ----------------
for i in range(len(X)):
    if y[i] == "Yes":
        for j in range(len(h)):
            if h[j] == '0':
                h[j] = X[i][j]
            elif h[j] != X[i][j]:
                h[j] = '?'

print("Final Hypothesis:", h)

# ---------------- PREDICTION ----------------
new_sample = ["Sunny", "Warm", "Normal", "Strong"]

result = "Yes"
for j in range(len(h)):
    if h[j] != '?' and h[j] != new_sample[j]:
        result = "No"
        break

print("Prediction:", result)




# import pandas as pd

# # ---------------- SWITCH ----------------
# use_csv = False   # True → use data.csv | False → use arrays

# # ---------------- DATA ----------------
# if use_csv:
#     df = pd.read_csv("data.csv")
#     X = df.drop("PlayTennis", axis=1).values
#     y = df["PlayTennis"].values
# else:
#     X = [
#         ["Sunny", "Warm", "Normal", "Strong"],
#         ["Sunny", "Warm", "High", "Strong"],
#         ["Rainy", "Cold", "High", "Strong"],
#         ["Sunny", "Warm", "Normal", "Weak"],
#         ["Rainy", "Warm", "Normal", "Strong"]
#     ]
#     y = ["Yes", "Yes", "No", "Yes", "No"]

# # ---------------- FIND-S ----------------
# h = ['0'] * len(X[0])

# for i in range(len(X)):
#     if y[i] == "Yes":
#         for j in range(len(h)):
#             if h[j] == '0':
#                 h[j] = X[i][j]
#             elif h[j] != X[i][j]:
#                 h[j] = '?'

# print("Final Hypothesis:", h)

# # ---------------- PREDICTION ----------------
# new_sample = ["Sunny", "Warm", "Normal", "Strong"]

# result = "Yes"
# for j in range(len(h)):
#     if h[j] != '?' and h[j] != new_sample[j]:
#         result = "No"
#         break

# print("Prediction:", result)