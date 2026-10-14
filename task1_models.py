"""
Task 1: Deep Learning Temporal Architectures Module (TensorFlow / Keras Implementation)
Paper: Anwar & Anwar (2025), Sensing and Bio-Sensing Research.

Evaluates LSTM, Temporal CNN (TCN), Transformer, and Hybrid (Conv1D-LSTM) models
using 5-Fold Group-based Splitting (GroupKFold) to prevent temporal data leakage.
"""

import os
import math
import numpy as np

# Suppress verbose TensorFlow C++ logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Fix seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ==========================================
# 1. LSTM Temporal Model Architecture
# ==========================================
def build_lstm_model(seq_len=5, input_dim=7, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
    """
    Bidirectional Multi-Layer LSTM Architecture.
    Captures long-range sequential gas concentration trends across temporal windows.
    """
    inputs = keras.Input(shape=(seq_len, input_dim), name="input_sequence")
    x = inputs
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(hidden_dim, return_sequences=return_sequences, dropout=dropout)
        )(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name="output_probabilities")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="LSTM_Classifier")


# ==========================================
# 2. Temporal Convolutional Network (TCN)
# ==========================================
def temporal_block(x, n_outputs, kernel_size=3, dilation_rate=1, dropout=0.2):
    """
    Single TCN Residual Block with dilated causal convolutions to prevent look-ahead bias.
    """
    in_channels = x.shape[-1]
    
    # First dilated causal convolution
    y = keras.layers.Conv1D(n_outputs, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)
    y = keras.layers.Dropout(dropout)(y)
    
    # Second dilated causal convolution
    y = keras.layers.Conv1D(n_outputs, kernel_size, padding='causal', dilation_rate=dilation_rate)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)
    y = keras.layers.Dropout(dropout)(y)
    
    # Residual projection to match dimensions if needed
    res = x if in_channels == n_outputs else keras.layers.Conv1D(n_outputs, 1, padding='same')(x)
    return keras.layers.Add()([y, res])


def build_tcn_model(seq_len=5, input_dim=7, num_channels=[32, 64], kernel_size=3, num_classes=3, dropout=0.2):
    """
    Temporal Convolutional Network (TCN) Architecture.
    Extracts multi-scale temporal receptive fields via causal dilated convolutions.
    """
    inputs = keras.Input(shape=(seq_len, input_dim), name="input_sequence")
    x = inputs
    for i, channels in enumerate(num_channels):
        dilation_rate = 2 ** i
        x = temporal_block(x, n_outputs=channels, kernel_size=kernel_size, dilation_rate=dilation_rate, dropout=dropout)
        
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name="output_probabilities")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="TCN_Classifier")


# ==========================================
# 3. Temporal Transformer Architecture
# ==========================================
class SinusoidalPositionalEncoding(keras.layers.Layer):
    """Sinusoidal positional encoding layer for sequence time-steps."""
    def __init__(self, d_model, max_len=100, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe, dtype=tf.float32)[tf.newaxis, :, :]

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


def transformer_encoder_block(x, d_model=64, num_heads=4, ff_dim=128, dropout=0.2):
    """Standard Transformer Encoder block with Multi-Head Attention and MLP."""
    # Multi-Head Self-Attention
    attn_out = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)(x, x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    
    # Position-wise Feed-Forward Network
    ffn = keras.Sequential([
        keras.layers.Dense(ff_dim, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(d_model)
    ])
    ffn_out = ffn(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)
    return x


def build_transformer_model(seq_len=5, input_dim=7, d_model=64, nhead=4, num_layers=2, num_classes=3, dropout=0.2):
    """
    Temporal Transformer Architecture.
    Learns global attention weights across temporal sensor windows.
    """
    inputs = keras.Input(shape=(seq_len, input_dim), name="input_sequence")
    x = keras.layers.Dense(d_model)(inputs)
    x = SinusoidalPositionalEncoding(d_model)(x)
    
    for _ in range(num_layers):
        x = transformer_encoder_block(x, d_model=d_model, num_heads=nhead, ff_dim=128, dropout=dropout)
        
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name="output_probabilities")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="Transformer_Classifier")


# ==========================================
# 4. Hybrid Conv1D-LSTM Architecture
# ==========================================
def build_hybrid_conv_lstm_model(seq_len=5, input_dim=7, conv_channels=32, lstm_hidden=64, num_classes=3, dropout=0.2):
    """
    Hybrid Conv1D-LSTM Architecture.
    Combines local sensor-channel feature extraction with sequential temporal memory modeling.
    """
    inputs = keras.Input(shape=(seq_len, input_dim), name="input_sequence")
    x = keras.layers.Conv1D(filters=conv_channels, kernel_size=3, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(dropout)(x)
    
    x = keras.layers.Bidirectional(keras.layers.LSTM(lstm_hidden, return_sequences=False))(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name="output_probabilities")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_Conv1D_LSTM_Classifier")


# ==========================================
# 5. Training & GroupKFold Evaluation Loop
# ==========================================
def train_and_eval_single_fold(model, X_tr, y_tr, X_va, y_va, epochs=40, lr=0.002, batch_size=64):
    """
    Trains a TensorFlow/Keras model on the fold training data and evaluates on validation data.
    Uses AdamW optimizer and early stopping on validation loss to restore best weights.
    """
    optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
    ]
    
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    prob_preds = model.predict(X_va, batch_size=batch_size, verbose=0)
    preds = np.argmax(prob_preds, axis=1)
    
    acc = accuracy_score(y_va, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_va, preds, average='macro', zero_division=0)
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def evaluate_all_temporal_models(X_seq, y_seq, g_seq, n_splits=5, epochs=40):
    """
    Evaluates LSTM, TCN, Transformer, and Hybrid Conv1D-LSTM models
    across 5-Fold GroupKFold cross validation to strictly prevent temporal leakage.
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    models_factory = {
        "LSTM": lambda: build_lstm_model(seq_len=X_seq.shape[1], input_dim=X_seq.shape[2], hidden_dim=64, num_layers=2, num_classes=3),
        "Temporal CNN (TCN)": lambda: build_tcn_model(seq_len=X_seq.shape[1], input_dim=X_seq.shape[2], num_channels=[32, 64], kernel_size=3, num_classes=3),
        "Transformer": lambda: build_transformer_model(seq_len=X_seq.shape[1], input_dim=X_seq.shape[2], d_model=64, nhead=4, num_layers=2, num_classes=3),
        "Hybrid (Conv1D-LSTM)": lambda: build_hybrid_conv_lstm_model(seq_len=X_seq.shape[1], input_dim=X_seq.shape[2], conv_channels=32, lstm_hidden=64, num_classes=3)
    }

    results = {}

    for name, model_fn in models_factory.items():
        print(f"\n==========================================")
        print(f"  Training & Evaluating (TensorFlow): {name}")
        print(f"==========================================")
        
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_seq, y_seq, groups=g_seq)):
            X_tr, y_tr = X_seq[train_idx], y_seq[train_idx]
            X_va, y_va = X_seq[val_idx], y_seq[val_idx]

            # Fit scaler per fold using 2D reshape to avoid feature-level leakage
            N_tr, S, F = X_tr.shape
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr.reshape(-1, F)).reshape(N_tr, S, F)
            
            N_va = X_va.shape[0]
            X_va_scaled = scaler.transform(X_va.reshape(-1, F)).reshape(N_va, S, F)

            # Build fresh model instance for fold
            keras.backend.clear_session()
            model = model_fn()
            
            metrics = train_and_eval_single_fold(model, X_tr_scaled, y_tr, X_va_scaled, y_va, epochs=epochs, lr=0.002)

            for key in fold_metrics:
                fold_metrics[key].append(metrics[key])

            print(f"Fold {fold+1}/{n_splits} - Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

        # Summary statistics across folds
        summary = {
            'accuracy_mean': np.mean(fold_metrics['accuracy']),
            'accuracy_std': np.std(fold_metrics['accuracy']),
            'precision_mean': np.mean(fold_metrics['precision']),
            'precision_std': np.std(fold_metrics['precision']),
            'recall_mean': np.mean(fold_metrics['recall']),
            'recall_std': np.std(fold_metrics['recall']),
            'f1_mean': np.mean(fold_metrics['f1']),
            'f1_std': np.std(fold_metrics['f1']),
        }

        results[name] = summary
        print(f"\n>>> {name} Final GroupKFold Mean Metrics:")
        print(f"    Accuracy:  {summary['accuracy_mean']*100:.2f}% ± {summary['accuracy_std']*100:.2f}%")
        print(f"    Precision: {summary['precision_mean']*100:.2f}% ± {summary['precision_std']*100:.2f}%")
        print(f"    Recall:    {summary['recall_mean']*100:.2f}% ± {summary['recall_std']*100:.2f}%")
        print(f"    F1-Score:  {summary['f1_mean']*100:.2f}% ± {summary['f1_std']*100:.2f}%")

    return results


if __name__ == '__main__':
    from data_preprocessing import load_and_preprocess_data, prepare_sequence_data
    _, X, y, groups = load_and_preprocess_data()
    X_seq, y_seq, g_seq = prepare_sequence_data(X, y, groups, seq_len=5)
    results = evaluate_all_temporal_models(X_seq, y_seq, g_seq, n_splits=5, epochs=5)

def train_and_save_best_model(X_seq, y_seq, save_dir='saved_model', 
                               model_name='Hybrid (Conv1D-LSTM)', epochs=40):
    """
    Retrains the chosen model on the FULL dataset (no held-out fold)
    and saves it to disk along with the fitted scaler.
    Call this once after cross-validation to produce the deployment model.
    """
    import joblib, os
    os.makedirs(save_dir, exist_ok=True)

    N, S, F = X_seq.shape

    # Fit scaler on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seq.reshape(-1, F)).reshape(N, S, F)

    # Build and train model on full data
    model_builders = {
        "LSTM": lambda: build_lstm_model(seq_len=S, input_dim=F),
        "Temporal CNN (TCN)": lambda: build_tcn_model(seq_len=S, input_dim=F),
        "Transformer": lambda: build_transformer_model(seq_len=S, input_dim=F),
        "Hybrid (Conv1D-LSTM)": lambda: build_hybrid_conv_lstm_model(seq_len=S, input_dim=F),
    }

    keras.backend.clear_session()
    model = model_builders[model_name]()

    optimizer = keras.optimizers.AdamW(learning_rate=0.002, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    print(f"\n[Save] Training {model_name} on full dataset ({N} samples)...")
    model.fit(X_scaled, y_seq, epochs=epochs, batch_size=64, verbose=1)

    # Save model and scaler
    model.save(os.path.join(save_dir, 'freshness_model.keras'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

    print(f"[Save] Model saved to '{save_dir}/freshness_model.keras'")
    print(f"[Save] Scaler saved to '{save_dir}/scaler.pkl'")
    return model, scaler
