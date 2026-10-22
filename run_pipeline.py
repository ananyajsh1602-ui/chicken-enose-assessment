"""
Master Pipeline Script for Chicken E-Nose Assessment
Runs:
- Preprocessing & GroupKFold preparation
- Task 1: Deep Learning Temporal Models (LSTM, TCN, Transformer, Hybrid Conv1D-LSTM)
- Task 2: Statistical ANOVA/MANOVA & Bio-Chemical Quality Validation (TBC, TVB-N, pH, GC)
- Plots and Result Summary Table Export
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import load_and_preprocess_data, prepare_sequence_data, CLASS_NAMES
from task1_models import evaluate_all_temporal_models, train_and_save_best_model
from task2_validation import perform_statistical_validation, simulate_and_validate_microbial_chemical_markers
# from shelf_life_estimator import compute_shelf_life_targets, train_and_evaluate_shelf_life_regressor, predict_sample_shelf_life

def run_master_pipeline():
    print("\n" + "="*70)
    print("  ELECTRONIC NOSE CHICKEN QUALITY ASSESSMENT - MASTER ML PIPELINE  ")
    print("="*70)
    
    # 1. Preprocessing
    df_clean, X, y, groups = load_and_preprocess_data('complete_data.csv')
    X_seq, y_seq, g_seq = prepare_sequence_data(X, y, groups, seq_len=5)
    
    # 2. Task 1: Deep Learning Temporal Models (GroupKFold)
    print("\n" + "="*70)
    print("  TASK 1: TEMPORAL DEEP LEARNING MODEL EVALUATION (GroupKFold)  ")
    print("="*70)
    task1_results = evaluate_all_temporal_models(X_seq, y_seq, g_seq, n_splits=5, epochs=3)
    
    # Format Task 1 Results DataFrame
    df_task1 = pd.DataFrame(task1_results).T
    df_task1_display = pd.DataFrame({
        'Accuracy (%)': df_task1['accuracy_mean'] * 100,
        'Precision (%)': df_task1['precision_mean'] * 100,
        'Recall (%)': df_task1['recall_mean'] * 100,
        'F1-Score (%)': df_task1['f1_mean'] * 100
    }).round(2)
    
    print("\n=== TASK 1 FINAL PERFORMANCE COMPARISON TABLE ===")
    print(df_task1_display)
    df_task1_display.to_csv('task1_model_comparison.csv')
    print("Saved 'task1_model_comparison.csv'")
    
    # 3. Task 2: Freshness Label Validation
    print("\n" + "="*70)
    print("  TASK 2: FRESHNESS LABEL VALIDATION (ANOVA, MANOVA, TBC, TVB-N, pH)  ")
    print("="*70)
    anova_res, manova_res = perform_statistical_validation(X, y)
    val_df, r2_scores = simulate_and_validate_microbial_chemical_markers(df_clean, X, y)
    val_df.to_csv('task2_freshness_validation_markers.csv', index=False)
    print("Saved 'task2_freshness_validation_markers.csv'")

    # 4. Task 3: Remaining Shelf-Life & Freshness Duration Estimation (RUL)
    # print("\n" + "="*70)
    # print("  TASK 3: REMAINING SHELF-LIFE & FRESHNESS DURATION ESTIMATION (RUL)  ")
    # print("="*70)
    # rfh_true, rsh_true, rdays_true = compute_shelf_life_targets(df_clean)
    # shelf_life_model, shelf_metrics = train_and_evaluate_shelf_life_regressor(X, rfh_true, rsh_true, groups)
    # import joblib
    # joblib.dump(shelf_life_model, 'saved_model/shelf_life_model.pkl')
    # print("[Save] Shelf life regressor saved to 'saved_model/shelf_life_model.pkl'")

    
    # Generate live predictions table across representative classes
    print("\n=== REPRESENTATIVE REAL-TIME SHELF-LIFE PREDICTIONS ===")
    sample_rows = []
    for val, name in enumerate(['Fresh', 'Semi-Fresh', 'Spoiled']):
        s_idx = np.where(y == val)[0][10]
        p_res = predict_sample_shelf_life(shelf_life_model, X[s_idx])
        p_res['True Label'] = name
        p_res['Elapsed Hour'] = f"{df_clean.iloc[s_idx]['hour']:.1f}h"
        sample_rows.append(p_res)
    df_sample_display = pd.DataFrame(sample_rows)[['True Label', 'Elapsed Hour', 'Quality Status', 'Remaining Fresh Time', 'Remaining Safe Window', 'Estimated Chilled Shelf-Life']]
    print(df_sample_display.to_string(index=False))
    
    # Save full sample-level predictions
    preds_all = shelf_life_model.predict(X)
    df_shelf_export = pd.DataFrame({
        'hour': df_clean['hour'],
        'true_label': [CLASS_NAMES[i] for i in y] if 'CLASS_NAMES' in locals() else y,
        'true_safe_hours': rsh_true,
        'pred_safe_hours': np.round(preds_all[:, 1], 2),
        'true_fresh_hours': rfh_true,
        'pred_fresh_hours': np.round(preds_all[:, 0], 2),
        'pred_chilled_days_4C': np.round((np.maximum(0, preds_all[:, 1]) * 4.84) / 24.0, 2)
    })
    df_shelf_export.to_csv('task3_shelf_life_predictions.csv', index=False)
    print("Saved 'task3_shelf_life_predictions.csv'")

    # 5. Generate Visualizations
    print("\n[Visualization] Generating diagnostic plots...")
    os.makedirs('plots', exist_ok=True)
    
    # Figure 1: Task 1 Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    df_task1_display[['Accuracy (%)', 'F1-Score (%)']].plot(kind='bar', figsize=(10, 5), colormap='viridis')
    plt.title('Task 1: Deep Learning Architectures Performance (5-Fold GroupKFold)', fontsize=13)
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=15)
    plt.ylim(50, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/task1_model_comparison.png', dpi=300)
    plt.close()
    
    # Figure 2: Task 2 Bio-Chemical & Microbial Markers across Freshness Classes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    sns.boxplot(x='label_name', y='TBC_log_CFU_g', data=val_df, ax=axes[0], palette='Blues')
    axes[0].set_title('Microbial Count (Total Bacterial Count)', fontsize=11)
    axes[0].set_ylabel('TBC (log CFU/g)')
    axes[0].axhline(6.0, color='orange', linestyle='--', label='Fresh Limit (6.0)')
    axes[0].axhline(7.0, color='red', linestyle='--', label='Spoilage Limit (7.0)')
    axes[0].legend()
    
    sns.boxplot(x='label_name', y='TVBN_mg_100g', data=val_df, ax=axes[1], palette='Oranges')
    axes[1].set_title('TVB-N Concentration Dynamics', fontsize=11)
    axes[1].set_ylabel('TVB-N (mg/100g)')
    axes[1].axhline(15.0, color='orange', linestyle='--', label='Fresh Limit (15.0)')
    axes[1].axhline(25.0, color='red', linestyle='--', label='Spoiled Limit (25.0)')
    axes[1].legend()

    sns.boxplot(x='label_name', y='pH', data=val_df, ax=axes[2], palette='Purples')
    axes[2].set_title('pH Level Shift', fontsize=11)
    axes[2].set_ylabel('pH')
    axes[2].axhline(6.2, color='orange', linestyle='--', label='Fresh Upper Bound (6.2)')
    axes[2].legend()

    plt.suptitle('Task 2: Validation of Freshness Labels via Quality Markers', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('plots/task2_bio_chemical_validation.png', dpi=300)
    plt.close()

    # # Figure 3: Task 3 Shelf-Life & Freshness Duration Estimation Diagnostics
    # fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    
    # # Subplot 1: True vs Predicted Safe Hours Parity Plot
    # sample_sub = df_shelf_export.sample(n=min(2000, len(df_shelf_export)), random_state=42)
    # axes3[0].scatter(sample_sub['true_safe_hours'], sample_sub['pred_safe_hours'], alpha=0.3, color='#1f77b4', s=20)
    # max_h = max(sample_sub['true_safe_hours'].max(), sample_sub['pred_safe_hours'].max())
    # axes3[0].plot([0, max_h], [0, max_h], 'r--', label='Ideal 1:1 Agreement')
    # axes3[0].set_title(f'Remaining Safe Hours: Predicted vs True (MAE: ±{shelf_metrics["safe_hours_mae"]:.2f}h)', fontsize=11)
    # axes3[0].set_xlabel('True Remaining Safe Hours')
    # axes3[0].set_ylabel('Predicted Remaining Safe Hours')
    # axes3[0].grid(True, linestyle=':', alpha=0.6)
    # axes3[0].legend()
    
    # Subplot 2: Refrigerated Shelf-Life (Days at 4°C) Distribution by Quality Class
    # sns.boxplot(x='true_label', y='pred_chilled_days_4C', hue='true_label', data=sample_sub, ax=axes3[1], palette='YlGnBu', legend=False)
    # axes3[1].set_title('Predicted Refrigerated Shelf-Life at 4°C (Days) by Class', fontsize=11)
    # axes3[1].set_xlabel('Freshness Class')
    # axes3[1].set_ylabel('Estimated Chilled Days Remaining')
    # axes3[1].grid(axis='y', linestyle=':', alpha=0.6)
    
    # plt.suptitle('Task 3: Remaining Shelf-Life (RUL) & Refrigerated Duration Estimation', fontsize=13, y=1.02)
    # plt.tight_layout()
    # plt.savefig('plots/task3_shelf_life_estimation.png', dpi=300)
    # plt.close()
    
    # print("Plots successfully saved in 'plots/' directory (including 'task3_shelf_life_estimation.png')!")
    # print("\n" + "="*70)
    # # Save deployment model (Hybrid Conv1D-LSTM trained on full data)
    # print("\n[Deployment] Saving best model for real-time inference...")
    # train_and_save_best_model(X_seq, y_seq, save_dir='saved_model',
    #                            model_name='Hybrid (Conv1D-LSTM)', epochs=40)

    # print("  ALL PIPELINE EXECUTION STEPS COMPLETED SUCCESSFULLY!  ")
    # print("="*70)

if __name__ == '__main__':
    run_master_pipeline()
