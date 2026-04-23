import pandas as pd

# ---------------- SWITCH ----------------
use_csv = False   # True → use data.csv | False → use arrays

# ---------------- DATA ----------------
if use_csv:
    df = pd.read_csv("data.csv")
    X = df.drop("PlayTennis", axis=1).values
    y = df["PlayTennis"].values
else:
    X = [
        ["Sunny", "Warm", "Normal", "Strong"],
        ["Sunny", "Warm", "High", "Strong"],
        ["Rainy", "Cold", "High", "Strong"],
        ["Sunny", "Warm", "Normal", "Weak"],
        ["Rainy", "Warm", "Normal", "Strong"]
    ]
    y = ["Yes", "Yes", "No", "Yes", "No"]

# ---------------- INITIALIZE ----------------
S = ['0'] * len(X[0])
G = [['?'] * len(X[0])]

# ---------------- ALGORITHM ----------------
for i in range(len(X)):

    if y[i] == "Yes":
        # update S
        for j in range(len(S)):
            if S[j] == '0':
                S[j] = X[i][j]
            elif S[j] != X[i][j]:
                S[j] = '?'

        # remove inconsistent from G
        G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(len(S)))]

    else:  # No
        new_G = []
        for g in G:
            for j in range(len(g)):
                if g[j] == '?':
                    if S[j] != X[i][j]:
                        new_h = g.copy()
                        new_h[j] = S[j]
                        new_G.append(new_h)
        G = new_G

# ---------------- OUTPUT ----------------
print("Specific Hypothesis S:", S)
print("General Hypothesis G:", G)

# ---------------- PREDICTION ----------------
new_sample = ["Sunny", "Warm", "Normal", "Strong"]

result = "No"
for g in G:
    match = True
    for j in range(len(g)):
        if g[j] != '?' and g[j] != new_sample[j]:
            match = False
            break
    if match:
        result = "Yes"
        break

print("Prediction:", result)