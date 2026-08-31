"""
train_model.py
---------------
Purpose : ML Training Script
Role    : Loads the dataset, selects the features, scales the data using
          StandardScaler, trains the K-Means clustering model, identifies
          the listener groups, and saves the trained model and scaler.

Follows the MODEL FLOW from the project architecture document:
1. Read music_listeners.csv
2. Select listener behaviour features
3. Scale the features using StandardScaler
4. Create K-Means model with 3 clusters
5. Train the model using fit()
6. Find and understand the three listener groups
7. Save model.pkl and scaler.pkl
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

FEATURES = [
    "listening_hours_per_week",
    "songs_per_day",
    "skip_rate",
    "playlist_count",
]

# ---------------------------------------------------------------
# STEP 1: Read music_listeners.csv
# ---------------------------------------------------------------
df = pd.read_csv("music_listeners.csv")
print(f"Loaded dataset with {len(df)} rows.")

# ---------------------------------------------------------------
# STEP 2: Select listener behaviour features
# (No target column exists -- this is Unsupervised Learning)
# ---------------------------------------------------------------
X = df[FEATURES]

# ---------------------------------------------------------------
# STEP 3: Scale the features using StandardScaler
# IMPORTANT RULE: use fit_transform() only during training
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------
# STEP 4: Create K-Means model with 3 clusters
# ---------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# ---------------------------------------------------------------
# STEP 5: Train the model using fit()
# ---------------------------------------------------------------
kmeans.fit(X_scaled)

# ---------------------------------------------------------------
# STEP 6: Find and understand the three listener groups
# K-Means itself only produces cluster numbers (0, 1, 2).
# We interpret the cluster centres (in original units) to assign
# meaningful names, ranked by overall listening activity.
# ---------------------------------------------------------------
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers_original, columns=FEATURES)
centers_df["cluster"] = centers_df.index

# Rank clusters by listening_hours_per_week (a strong activity signal)
centers_df = centers_df.sort_values("listening_hours_per_week").reset_index(drop=True)
label_order = ["Casual Listener", "Music Explorer", "Heavy Listener"]

cluster_to_label = {
    int(row["cluster"]): label_order[i] for i, row in centers_df.iterrows()
}

print("\nCluster centres (original feature units):")
print(centers_df)
print("\nCluster number -> Listener segment mapping:")
for cluster_num, label in cluster_to_label.items():
    print(f"  Cluster {cluster_num} -> {label}")

# ---------------------------------------------------------------
# STEP 7: Save model.pkl and scaler.pkl
# We also save the cluster-to-label mapping and feature order so
# app.py can display meaningful segment names and use the exact
# same feature order at prediction time.
# ---------------------------------------------------------------
joblib.dump(kmeans, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(cluster_to_label, "cluster_labels.pkl")
joblib.dump(FEATURES, "feature_order.pkl")

print("\nSaved model.pkl, scaler.pkl, cluster_labels.pkl, feature_order.pkl")
print("Training complete.")
