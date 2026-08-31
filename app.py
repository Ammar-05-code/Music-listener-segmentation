"""
app.py
------
Purpose : Backend / Application (built with Streamlit)
Role    : Loads the saved model and scaler, receives listener details,
          preprocesses the input, predicts the cluster and returns the
          listener segment.

Run with:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Music Listener Segmentation",
    page_icon="🎧",
    layout="centered",
)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
LABELS_PATH = "cluster_labels.pkl"
FEATURES_PATH = "feature_order.pkl"

SEGMENT_INFO = {
    "Casual Listener": {
        "emoji": "🌙",
        "desc": "Listens in short bursts, skips often, and keeps a small playlist collection.",
    },
    "Music Explorer": {
        "emoji": "🧭",
        "desc": "A steady, curious listener who explores a moderate variety of tracks and playlists.",
    },
    "Heavy Listener": {
        "emoji": "🔥",
        "desc": "Deeply engaged — long listening hours, high song counts, rarely skips.",
    },
}


@st.cache_resource
def load_artifacts():
    """Load the trained model, scaler, and label mapping saved by train_model.py."""
    missing = [p for p in [MODEL_PATH, SCALER_PATH] if not os.path.exists(p)]
    if missing:
        return None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    cluster_to_label = joblib.load(LABELS_PATH) if os.path.exists(LABELS_PATH) else None
    feature_order = joblib.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else [
        "listening_hours_per_week", "songs_per_day", "skip_rate", "playlist_count"
    ]
    return model, scaler, cluster_to_label, feature_order


st.title("🎧 Music Listener Segmentation")
st.caption("Unsupervised Learning (K-Means Clustering) — Streamlit App")

artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "Model files not found. Please run `python train_model.py` first "
        "to generate `model.pkl` and `scaler.pkl`."
    )
    st.stop()

model, scaler, cluster_to_label, feature_order = artifacts

tab_predict, tab_explore = st.tabs(["🔍 Predict a Listener", "📊 Explore the Model"])

# ---------------------------------------------------------------------
# TAB 1: Predict a new listener's segment
# ---------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter listener details")

    col1, col2 = st.columns(2)
    with col1:
        listening_hours = st.number_input(
            "Listening hours per week", min_value=0.0, max_value=100.0, value=10.0, step=0.5
        )
        songs_per_day = st.number_input(
            "Songs per day", min_value=0, max_value=200, value=20, step=1
        )
    with col2:
        skip_rate = st.slider(
            "Skip rate", min_value=0.0, max_value=1.0, value=0.30, step=0.01
        )
        playlist_count = st.number_input(
            "Playlist count", min_value=0, max_value=100, value=5, step=1
        )

    if st.button("Predict Segment", type="primary"):
        # STEP 8: Receive new listener input
        input_dict = {
            "listening_hours_per_week": listening_hours,
            "songs_per_day": songs_per_day,
            "skip_rate": skip_rate,
            "playlist_count": playlist_count,
        }
        # Use the exact same feature order as training
        input_df = pd.DataFrame([[input_dict[f] for f in feature_order]], columns=feature_order)

        # STEP 9: Scale the new input using the saved scaler
        # IMPORTANT RULE: use transform() only, never fit again
        input_scaled = scaler.transform(input_df)

        # STEP 10: Predict the cluster using model.predict()
        cluster_num = int(model.predict(input_scaled)[0])

        # STEP 11: Display the meaningful listener segment
        label = cluster_to_label.get(cluster_num, f"Cluster {cluster_num}") if cluster_to_label else f"Cluster {cluster_num}"
        info = SEGMENT_INFO.get(label, {"emoji": "🎵", "desc": ""})

        st.success(f"{info['emoji']} **Predicted Segment: {label}**  (raw cluster #{cluster_num})")
        st.write(info["desc"])

# ---------------------------------------------------------------------
# TAB 2: Explore the trained model / cluster centres
# ---------------------------------------------------------------------
with tab_explore:
    st.subheader("Cluster centres (learned by K-Means)")

    if hasattr(model, "cluster_centers_"):
        centers_original = scaler.inverse_transform(model.cluster_centers_)
        centers_df = pd.DataFrame(centers_original, columns=feature_order)
        centers_df.insert(0, "Cluster #", range(len(centers_df)))
        if cluster_to_label:
            centers_df.insert(1, "Segment", [cluster_to_label.get(i, "") for i in range(len(centers_df))])
        st.dataframe(centers_df.round(2), use_container_width=True, hide_index=True)

        st.caption(
            "K-Means only outputs cluster numbers (0, 1, 2...). "
            "The segment names above were assigned by interpreting each "
            "cluster centre's behaviour — this mapping is not automatic."
        )

    if os.path.exists("music_listeners.csv"):
        st.subheader("Training dataset preview")
        df = pd.read_csv("music_listeners.csv")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        st.caption(f"Full dataset: {len(df)} rows, no target/label column (unsupervised).")

    st.subheader("Why this is Unsupervised Learning")
    st.markdown(
        """
- **Features (X):** ✅ the four listener behaviour columns are used as input.
- **Target (y):** ❌ there is no known answer column to learn from.
- **fit() / predict():** ✅ used, but to group data — not to classify against known labels.
- **Accuracy / Precision / Recall:** ❌ not applicable — there's no ground truth to compare against.
- **Train/Test split, Cross-validation, Hyperparameter tuning:** ❌ not used in this simple project.
        """
    )
