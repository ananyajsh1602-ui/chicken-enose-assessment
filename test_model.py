"""
Manual Model Tester — No Arduino needed.
Feed sample sensor values and see freshness prediction + shelf life.
"""
import numpy as np
import joblib
import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from shelf_life_estimator import predict_sample_shelf_life
from collections import deque

MODEL_PATH      = 'saved_model/freshness_model.keras'
SCALER_PATH     = 'saved_model/scaler.pkl'
SHELF_MODEL_PATH= 'saved_model/shelf_life_model.pkl'
CLASS_NAMES     = ['🟢 FRESH', '🟡 SEMI-FRESH', '🔴 SPOILED']

model       = keras.models.load_model(MODEL_PATH)
scaler      = joblib.load(SCALER_PATH)
shelf_model = joblib.load(SHELF_MODEL_PATH)

# ── Define your test samples here ─────────────────────────
#    Order: [mq135, mq2, mq3, mq6, mq7, mq9, humid]
#    Values represent delta changes relative to clean air baseline (Anwar & Anwar 2025).
test_samples = {
    "Clean Air Baseline (t=0, Zero Delta)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Fresh Chicken Sample":                  [40.8, 17.1, -10.1, 13.9, 5.7, 17.8, 0.36],
    "Semi-Fresh Chicken Sample":             [238.7, 164.3, 116.8, 110.0, 59.2, 283.8, 0.47],
    "Spoiled Chicken Sample":                [392.4, 366.0, 383.5, 310.0, 205.6, 605.9, 0.59],
    "Simulated Gas Spike (MQ9 Spoilage)":    [40.0, 17.0, -10.0, 14.0, 6.0, 600.0, 0.4]
}
# ──────────────────────────────────────────────────────────

SEQ_LEN = 5

for label, single_window in test_samples.items():
    # Simulate SEQ_LEN identical windows (same reading repeated 5 times)
    # This is valid for testing — real use has 5 different windows
    sequence = np.array([single_window] * SEQ_LEN)       # shape: (5, 7)
    sequence_scaled = scaler.transform(sequence)           # shape: (5, 7)
    model_input = sequence_scaled[np.newaxis, :, :]       # shape: (1, 5, 7)

    probs = model.predict(model_input, verbose=0)[0]
    pred_class = np.argmax(probs)
    confidence = probs[pred_class] * 100

    shelf_info = predict_sample_shelf_life(shelf_model, np.array(single_window))

    print(f"\n{'='*55}")
    print(f"  TEST: {label}")
    print(f"  Input: {single_window}")
    print(f"{'='*55}")
    print(f"  Prediction  : {CLASS_NAMES[pred_class]}  ({confidence:.1f}% confidence)")
    print(f"  Probabilities → Fresh:{probs[0]*100:.1f}%  Semi:{probs[1]*100:.1f}%  Spoiled:{probs[2]*100:.1f}%")
    if pred_class < 2:
        print(f"  Remaining Safe Time : {shelf_info['Remaining Safe Window']}")
        print(f"  Remaining Fresh Time: {shelf_info['Remaining Fresh Time']}")
        print(f"  Chilled Shelf Life  : {shelf_info['Estimated Chilled Shelf-Life']}")
    else:
        print(f"  ⚠️  SPOILED — DO NOT CONSUME")
