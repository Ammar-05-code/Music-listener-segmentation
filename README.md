# Music Listener Segmentation (Unsupervised Machine Learning)

A K-Means clustering project that groups music listeners into three
behavioural segments — **Casual Listener**, **Music Explorer**, and
**Heavy Listener** — based on their listening habits. The prediction
app is built with **Streamlit**.

## Project Files

| File | Purpose |
|---|---|
| `requirements.txt` | Dependency manifest (pandas, scikit-learn, joblib, streamlit) |
| `music_listeners.csv` | Dataset: listening_hours_per_week, songs_per_day, skip_rate, playlist_count |
| `generate_dataset.py` | (Optional) script used to generate the sample dataset |
| `train_model.py` | Loads data, scales features, trains K-Means, saves the model |
| `model.pkl` | Saved trained K-Means model |
| `scaler.pkl` | Saved StandardScaler (must be reused, never refit, at prediction time) |
| `cluster_labels.pkl` | Maps raw cluster numbers (0/1/2) to meaningful segment names |
| `feature_order.pkl` | Locks in the exact feature order used during training |
| `app.py` | Streamlit application — predicts a segment for a new listener |

## Model Flow

1. Read `music_listeners.csv`
2. Select the four listener behaviour features
3. Scale the features using `StandardScaler`
4. Create a K-Means model with 3 clusters
5. Train the model using `fit()`
6. Interpret the cluster centres to name the three listener groups
7. Save `model.pkl` and `scaler.pkl`
8. In the app: receive new listener input
9. Scale the new input using the **saved** scaler (`transform()` only)
10. Predict the cluster using `model.predict()`
11. Display the meaningful listener segment

## Setup & Run

### Step 1 — Create the project folder and open it in VS Code
Open this folder in VS Code (`File > Open Folder...`).

### Step 2 — Install Python
Make sure Python 3.9+ is installed. Open a terminal in VS Code
(`Terminal > New Terminal`).

### Step 3 — Install dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 4 — (Optional) Regenerate the dataset
A ready-made `music_listeners.csv` is already included. If you want a
fresh random sample instead, run:
```bash
python generate_dataset.py
```

### Step 5 — Train the model
```bash
python train_model.py
```
This prints the cluster centres and the cluster-number → segment-name
mapping, then saves `model.pkl`, `scaler.pkl`, `cluster_labels.pkl`,
and `feature_order.pkl`.

### Step 6 — Run the Streamlit app
```bash
streamlit run app.py
```
This opens the app in your browser (usually at `http://localhost:8501`).

- **Predict a Listener tab:** enter listening hours/week, songs/day,
  skip rate, and playlist count, then click **Predict Segment** to see
  the listener's cluster and segment name.
- **Explore the Model tab:** view the learned cluster centres, a
  preview of the training data, and a short explanation of why this
  project is unsupervised learning.

## Important Rules (from the project spec)

1. During training, use `fit_transform()` on the training features.
2. During prediction, use `transform()` only — never refit the scaler.
3. Use the same feature order during training and prediction
   (`feature_order.pkl` enforces this).
4. Cluster numbers (0, 1, 2) are not automatically meaningful — they
   are mapped to segment names by interpreting the cluster centres
   (see `cluster_labels.pkl`, built in `train_model.py`).
5. `model.pkl` is the trained model; `scaler.pkl` is the preprocessing
   object.
6. Train the model (`python train_model.py`) before running the app
   for the first time.

## Why This Is Unsupervised Learning

| Concept | Used? |
|---|---|
| Features (X) | ✅ Yes |
| Target (y) | ❌ No — no known answer column |
| Preprocessing (StandardScaler) | ✅ Yes |
| Model creation | ✅ Yes (K-Means) |
| `fit()` | ✅ Yes |
| `predict()` | ✅ Yes |
| Train/Test Split | ❌ No |
| Classification / Regression | ❌ No |
| Accuracy / Precision / Recall | ❌ No — no ground truth to score against |
| Cross-Validation | ❌ No |
| Hyperparameter Tuning | ❌ No — fixed K-Means settings |
| Joblib | ✅ Yes — saves/loads model and scaler |
