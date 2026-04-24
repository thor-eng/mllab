import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ---------------- SWITCHES ----------------
use_csv = False        # True → use data.csv
use_split = False    # True → use train/test split

# ---------------- DATA ----------------
if use_csv:
    df = pd.read_csv("data.csv")
    X = df[["AnnualIncome", "SpendingScore"]]
else:
    X = np.array([
        [20000,15],[22000,20],[25000,25],[30000,30],
        [40000,40],[45000,42],[50000,50],[55000,55],
        [60000,60],[65000,65],[70000,70],[75000,75]
    ])
    df = pd.DataFrame(X, columns=["AnnualIncome", "SpendingScore"])

# ---------------- OPTIONAL SPLIT ----------------
if use_split:
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
else:
    X_train = X   # fallback → use full data

# ---------------- MODEL ----------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

# ---------------- CLUSTERS ----------------
df["Cluster"] = kmeans.predict(X)

# ---------------- SINGLE PREDICTION ----------------
if use_csv:
    new_customer = pd.DataFrame([[60000, 65]], columns=["AnnualIncome", "SpendingScore"])
else:
    new_customer = np.array([[60000, 65]])

print("Cluster:", kmeans.predict(new_customer))

# ---------------- CENTROIDS ----------------
print("Cluster Centers:\n", kmeans.cluster_centers_)

# ---------------- GRAPH ----------------
# plt.scatter(df["AnnualIncome"], df["SpendingScore"], c=df["Cluster"])

# plt.scatter(
#     kmeans.cluster_centers_[:,0],
#     kmeans.cluster_centers_[:,1],
#     marker='X',
#     s=200
# )

labels = kmeans.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker='X',
    s=200
)

plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("K-Means Clustering")
plt.show()