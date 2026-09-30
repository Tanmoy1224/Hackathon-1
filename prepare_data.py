import numpy as np
import pandas as pd
import os

CSV_FILE = "driver_fatigue_dataset.csv"  # Check the path if it's inside dataset/
WINDOW_SIZE = 30  # 30 frames (~1.0 sec at 30 FPS)
STRIDE = 5        # Advance 5 frames between windows

def build_temporal_data():
    if not os.path.exists(CSV_FILE):
        # Look inside dataset/ folder if not found in root
        alt_path = os.path.join("dataset", CSV_FILE)
        if os.path.exists(alt_path):
            file_to_load = alt_path
        else:
            raise FileNotFoundError(f"Cannot find '{CSV_FILE}' or '{alt_path}'. Make sure your CSV exists!")
    else:
        file_to_load = CSV_FILE

    print(f"Loading data from: {file_to_load}")
    df = pd.read_csv(file_to_load)

    # 1. Extract feature columns and label column
    # If using EAR/MAR features:
    feature_cols = [c for c in df.columns if c != "label"]
    
    features = df[feature_cols].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)

    total_frames = len(df)
    print(f"Total recorded frames: {total_frames}")
    print(f"Features detected per frame: {features.shape[1]} ({feature_cols})")

    if total_frames < WINDOW_SIZE:
        raise ValueError(f"You have only {total_frames} frames. Need at least {WINDOW_SIZE} to create 1 sequence.")

    # 2. Sliding Window Assembly
    X_list = []
    y_list = []

    for start_idx in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
        end_idx = start_idx + WINDOW_SIZE
        
        # Window of 30 consecutive frames
        win_feats = features[start_idx:end_idx]
        win_labels = labels[start_idx:end_idx]

        # Majority vote for label across the window
        unique, counts = np.unique(win_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        X_list.append(win_feats)
        y_list.append(majority_label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"\n--> Successfully assembled!")
    print(f"X shape: {X.shape}  (Samples, Window Size, Features)")
    print(f"y shape: {y.shape}  (Samples,)")

    # 3. Save to disk
    np.save("X_temporal.npy", X)
    np.save("y_temporal.npy", y)
    print("Saved 'X_temporal.npy' and 'y_temporal.npy' to current directory.")

if __name__ == "__main__":
    build_temporal_data()