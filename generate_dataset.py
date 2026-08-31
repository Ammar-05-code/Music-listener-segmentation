"""
generate_dataset.py
--------------------
Utility script (not part of the required project files) used only to
create a realistic music_listeners.csv for this project.

Generates listener behaviour data with three loosely-separated natural
groupings (casual / explorer / heavy), WITHOUT adding any target/label
column -- this keeps the dataset genuinely unsupervised, exactly as the
architecture document specifies.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_per_group = 70

# Casual listeners: low hours, low songs/day, higher skip rate, few playlists
casual = pd.DataFrame({
    "listening_hours_per_week": np.random.normal(4, 1.5, n_per_group).clip(0.5, 10),
    "songs_per_day": np.random.normal(8, 3, n_per_group).clip(1, 20),
    "skip_rate": np.random.normal(0.45, 0.1, n_per_group).clip(0.1, 0.9),
    "playlist_count": np.random.normal(2, 1, n_per_group).clip(0, 6),
})

# Music explorers: medium hours, medium-high songs/day, moderate skip rate, many playlists
explorer = pd.DataFrame({
    "listening_hours_per_week": np.random.normal(14, 3, n_per_group).clip(6, 24),
    "songs_per_day": np.random.normal(28, 6, n_per_group).clip(15, 45),
    "skip_rate": np.random.normal(0.35, 0.08, n_per_group).clip(0.1, 0.6),
    "playlist_count": np.random.normal(9, 2.5, n_per_group).clip(3, 16),
})

# Heavy listeners: very high hours, very high songs/day, low skip rate, many playlists
heavy = pd.DataFrame({
    "listening_hours_per_week": np.random.normal(28, 4, n_per_group).clip(18, 45),
    "songs_per_day": np.random.normal(55, 8, n_per_group).clip(35, 80),
    "skip_rate": np.random.normal(0.15, 0.06, n_per_group).clip(0.02, 0.35),
    "playlist_count": np.random.normal(14, 3, n_per_group).clip(6, 25),
})

df = pd.concat([casual, explorer, heavy], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows

# Round for readability
df["listening_hours_per_week"] = df["listening_hours_per_week"].round(1)
df["songs_per_day"] = df["songs_per_day"].round(0).astype(int)
df["skip_rate"] = df["skip_rate"].round(2)
df["playlist_count"] = df["playlist_count"].round(0).astype(int)

df.to_csv("music_listeners.csv", index=False)
print(f"Saved music_listeners.csv with {len(df)} rows.")
