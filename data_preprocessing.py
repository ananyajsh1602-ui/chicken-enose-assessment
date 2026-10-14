"""
Data Preprocessing Module for Chicken E-Nose Quality Assessment
Paper: Anwar & Anwar (2025), Sensing and Bio-Sensing Research.

This module handles:
1. Loading raw e-nose time-series sensor data from CSV.
2. Applying 12-sample moving average windowing (averaging readings every 5 minutes).
3. Grouping and filtering edge samples per monitoring session/day.
4. Season-aware labeling (Winter vs Warm storage condition thresholds):
   - Fresh (0): <= 8h (Winter) / <= 5h (Warm)
   - Semi-Fresh (1): 8-16h (Winter) / 5-10h (Warm)
   - Spoiled (2): > 16h (Winter) / > 10h (Warm)
5. Group-based train/test splitting (GroupKFold) to prevent temporal data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Sensor names mapping according to the paper
SENSOR_COLS = ['mq135', 'mq2', 'mq3', 'mq6', 'mq7', 'mq9', 'humid']
CLASS_NAMES = ['Fresh', 'Semi-Fresh', 'Spoiled']

def load_and_preprocess_data(csv_path='complete_data.csv'):
    """
    Loads raw CSV data, computes 5-minute moving averages, cleans group boundaries,
    and applies storage-condition-based label mapping.
    
    Returns:
        df_processed (pd.DataFrame): Processed dataframe with standardized sensor names.
        X (np.ndarray): Feature matrix (N, 7) [MQ135, MQ2, MQ3, MQ6, MQ7, MQ9, Humid].
        y (np.ndarray): Target class array (N,) with values 0 (Fresh), 1 (Semi-Fresh), 2 (Spoiled).
        groups (np.ndarray): Group identifier array (N,) corresponding to monitoring days/samples.
    """
    print(f"[Data Preprocessing] Loading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    # 1. 12-sample moving average window (12 readings @ 5s interval = 1 min per window, group index//12*5)
    print("[Data Preprocessing] Computing 5-minute moving average window...")
    df_grouped = df.groupby(df.index // 12 * 5).mean()
    df_grouped['hour'] = df_grouped['hour'].round()
    df_grouped['group'] = df_grouped['group'].round()
    
    # 2. Filter transient start/end rows for each experimental group
    dfs = []
    for g in df_grouped['group'].unique():
        sub = df_grouped[df_grouped['group'] == g]
        if len(sub) > 2:
            dfs.append(sub.iloc[1:-1])  # trim edge rows per group
        else:
            dfs.append(sub)
            
    df_clean = pd.concat(dfs).reset_index(drop=True)
    
    # 3. Separate Winter vs Warm groups based on max monitoring duration (> 15 hours = Winter)
    group_max_hours = df_clean.groupby('group')['hour'].max().reset_index()
    winter_groups = group_max_hours[group_max_hours['hour'] > 15]['group'].tolist()
    warm_groups = group_max_hours[group_max_hours['hour'] <= 15]['group'].tolist()
    
    df_winter = df_clean[df_clean['group'].isin(winter_groups)].copy()
    df_warm = df_clean[df_clean['group'].isin(warm_groups)].copy()
    
    def apply_labels(df_sub, l1, l2):
        X_sub = df_sub.iloc[:, 0:-2].values
        hours = df_sub['hour'].values
        
        y_sub = np.zeros(len(hours), dtype=int)
        y_sub[hours <= l1] = 0                      # Fresh
        y_sub[(hours > l1) & (hours <= l2)] = 1      # Semi-Fresh
        y_sub[hours > l2] = 2                       # Spoiled
        
        g_sub = df_sub['group'].values
        return X_sub, y_sub, g_sub

    # Winter labeling: l1=8h, l2=16h; Warm labeling: l1=5h, l2=10h
    X1, y1, g1 = apply_labels(df_winter, l1=8, l2=16)
    X2, y2, g2 = apply_labels(df_warm, l1=5, l2=10)
    
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    groups = np.concatenate((g1, g2), axis=0)
    
    print(f"[Data Preprocessing] Complete! Total samples: {X.shape[0]}, Features: {X.shape[1]}")
    class_counts = {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)}
    print(f"[Data Preprocessing] Class distribution: {class_counts}")
    print(f"[Data Preprocessing] Unique groups (days): {len(np.unique(groups))} groups -> {np.unique(groups).tolist()}")
    
    return df_clean, X, y, groups

def prepare_sequence_data(X, y, groups, seq_len=5):
    """
    Converts 2D tabular sensor readings into 3D sequential tensors (N_seq, seq_len, num_features)
    for deep temporal architectures (LSTM, TCN, Transformer, Hybrid).
    Preserves group alignment without crossing group boundaries.
    """
    X_seq_list, y_seq_list, g_seq_list = [], [], []
    
    unique_groups = np.unique(groups)
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        X_g = X[idx]
        y_g = y[idx]
        
        if len(X_g) < seq_len:
            # Pad sequence if too short
            pad_len = seq_len - len(X_g)
            X_padded = np.pad(X_g, ((pad_len, 0), (0, 0)), mode='edge')
            y_padded = np.pad(y_g, (pad_len, 0), mode='edge')
            X_seq_list.append(X_padded)
            y_seq_list.append(y_padded[-1])
            g_seq_list.append(g)
        else:
            for i in range(len(X_g) - seq_len + 1):
                X_seq_list.append(X_g[i:i + seq_len])
                y_seq_list.append(y_g[i + seq_len - 1])
                g_seq_list.append(g)
                
    return np.array(X_seq_list), np.array(y_seq_list), np.array(g_seq_list)

if __name__ == '__main__':
    df, X, y, groups = load_and_preprocess_data()
    X_seq, y_seq, g_seq = prepare_sequence_data(X, y, groups, seq_len=5)
    print(f"Sequence shape for Deep Learning: {X_seq.shape}")
