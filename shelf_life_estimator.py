"""
Remaining Shelf-Life (RUL) & Freshness Duration Estimation Module
Translates E-Nose multi-gas sensor readings into remaining fresh hours and refrigerated days.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def compute_shelf_life_targets(df_clean):
    """
    Computes Remaining Fresh Hours (RFH) and Remaining Safe Hours (RSH)
    based on monitoring duration and storage condition thresholds.
    """
    hours = df_clean['hour'].values
    groups = df_clean['group'].values
    
    # Identify storage condition (Winter: >15h duration; Warm: <=15h duration)
    group_max_hours = df_clean.groupby('group')['hour'].max()
    is_winter = df_clean['group'].map(lambda g: group_max_hours[g] > 15)
    
    # Define spoilage and fresh limit hours
    t_fresh = np.where(is_winter, 8.0, 5.0)
    t_spoil = np.where(is_winter, 16.0, 10.0)
    
    # Calculate remaining hours
    remaining_fresh_hrs = np.maximum(0.0, t_fresh - hours)
    remaining_safe_hrs = np.maximum(0.0, t_spoil - hours)
    
    # Convert to equivalent refrigerated days (4°C) using Q10 Arrhenius scaling (factor ~4.84)
    refrigerated_days = (remaining_safe_hrs * 4.84) / 24.0
    
    return remaining_fresh_hrs, remaining_safe_hrs, refrigerated_days

def train_and_evaluate_shelf_life_regressor(X, remaining_fresh_hrs, remaining_safe_hrs, groups):
    """
    Evaluates Remaining Shelf-Life prediction using GroupKFold to prevent leakage.
    """
    print("\n" + "="*65)
    print("  REMAINING SHELF-LIFE (RUL) ESTIMATION BENCHMARK (GroupKFold)")
    print("="*65)
    
    gkf = GroupKFold(n_splits=5)
    y_targets = np.column_stack([remaining_fresh_hrs, remaining_safe_hrs])
    
    maes_fresh, maes_safe, r2_safe = [], [], []
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_targets, groups=groups)):
        X_tr, y_tr = X[tr_idx], y_targets[tr_idx]
        X_va, y_va = X[va_idx], y_targets[va_idx]
        
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_va)
        
        mae_fresh = mean_absolute_error(y_va[:, 0], preds[:, 0])
        mae_safe = mean_absolute_error(y_va[:, 1], preds[:, 1])
        r2 = r2_score(y_va[:, 1], preds[:, 1])
        
        maes_fresh.append(mae_fresh)
        maes_safe.append(mae_safe)
        r2_safe.append(r2)
        
        print(f"Fold {fold+1}/5 | Safe Hours MAE: {mae_safe:.2f} hrs | R²: {r2:.4f}")
        
    summary_metrics = {
        'safe_hours_mae': np.mean(maes_safe),
        'safe_hours_r2': np.mean(r2_safe),
        'fresh_hours_mae': np.mean(maes_fresh)
    }
    print(f"\n>>> Mean Prediction Error: ±{summary_metrics['safe_hours_mae']:.2f} hours (R² = {summary_metrics['safe_hours_r2']:.4f})")
    
    # Train final deployment model on complete data for real-time inference
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X, y_targets)
    
    return final_model, summary_metrics

def predict_sample_shelf_life(model, sensor_reading):
    """
    Provides a real-time shelf life prediction for an incoming sensor reading.
    """
    pred = model.predict(np.array(sensor_reading).reshape(1, -1))[0]
    fresh_hrs = max(0.0, float(pred[0]))
    safe_hrs = max(0.0, float(pred[1]))
    chilled_days = (safe_hrs * 4.84) / 24.0
    
    status = "FRESH" if fresh_hrs > 0.5 else ("SEMI-FRESH" if safe_hrs > 0.5 else "SPOILED")
    
    return {
        "Quality Status": status,
        "Remaining Fresh Time": f"{fresh_hrs:.1f} hours (under ambient storage)",
        "Remaining Safe Window": f"{safe_hrs:.1f} hours (before spoilage limit)",
        "Estimated Chilled Shelf-Life": f"{chilled_days:.1f} days (at 4°C refrigeration)"
    }

if __name__ == '__main__':
    from data_preprocessing import load_and_preprocess_data
    df_clean, X, y, groups = load_and_preprocess_data('complete_data.csv')
    rfh, rsh, r_days = compute_shelf_life_targets(df_clean)
    model, metrics = train_and_evaluate_shelf_life_regressor(X, rfh, rsh, groups)
    
    print("\n--- Live Test Predictions Across Freshness Categories ---")
    for val, name in enumerate(['Fresh', 'Semi-Fresh', 'Spoiled']):
        sample_idx = np.where(y == val)[0][10]
        res = predict_sample_shelf_life(model, X[sample_idx])
        print(f"\n[{name} Sample - Elapsed Hour {df_clean.iloc[sample_idx]['hour']:.1f}h]:")
        for k, v in res.items():
            print(f"  {k}: {v}")
