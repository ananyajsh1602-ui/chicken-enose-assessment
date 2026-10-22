"""
predict_only.py  —  Freshness Classification ONLY
---------------------------------------------------
Reads from Arduino, maintains a sliding window of 5 averaged windows,
and predicts: FRESH / SEMI-FRESH / SPOILED using the LSTM model.

No shelf-life estimation — use this to verify the classifier in isolation.

Usage:
    python predict_only.py
"""

import serial
import serial.tools.list_ports
import numpy as np
import joblib
import time
import os
import sys
from collections import deque
from datetime import datetime

# UTF-8 output for emojis on Windows consoles
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

# ── Configuration ─────────────────────────────────────────────────────────────
BAUD_RATE           = 9600
MODEL_PATH          = 'saved_model/freshness_model.keras'
SCALER_PATH         = 'saved_model/scaler.pkl'
SEQ_LEN             = 5        # must match training
READINGS_PER_WINDOW = 12       # 12 readings × 5s = 60s per window
NUM_FEATURES        = 7        # mq135, mq2, mq3, mq6, mq7, mq9, humid
CLASS_NAMES         = ['FRESH', 'SEMI-FRESH', 'SPOILED']
CLASS_ICONS         = ['🟢', '🟡', '🔴']
CALIBRATE_BASELINE  = True      # subtract clean-air baseline (ΔS)

# ── Sensor-based spoilage override thresholds (ΔS from baseline) ─────────────
# Derived from p10 of Spoiled class in training data (conservative lower bound).
# If enough sensors breach their spoiled-zone floor, override LSTM to SPOILED.
# Order: [MQ135, MQ2, MQ3, MQ6, MQ7, MQ9, Humid]
#  MQ9 & MQ6 are weighted most — strongest spoilage discriminators in this dataset.
SPOIL_THRESHOLDS = {
    'MQ9':   86,   # index 5 — hydrogen sulphide / spoilage gases
    'MQ6':   33,   # index 3 — benzene / toluene from protein breakdown
    'MQ7':   33,   # index 4 — carbon monoxide
    'MQ135': 106,  # index 0 — ammonia
    'MQ2':   114,  # index 1 — LPG / alcohol (less reliable for chicken)
}
# Override to SPOILED if at least this many sensors breach their threshold:
SPOIL_SENSOR_VOTE_MIN = 3   # at least 3 out of 5 sensors must say "spoiled"
# ──────────────────────────────────────────────────────────────────────────────


def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(k in (p.description or '') for k in ['Arduino', 'CH340', 'USB Serial']):
            return p.device
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")
    return ports[int(input("Select port number: "))].device


def parse_line(line):
    """Parse one CSV line from Arduino. Returns list of 7 floats or None."""
    try:
        parts = line.strip().split(',')
        if len(parts) != 8:
            return None
        return [float(p) for p in parts[:7]]   # drop temp column
    except ValueError:
        return None


def collect_one_window(ser):
    """Blocks until 12 valid readings are collected. Returns averaged array (7,)."""
    readings = []
    print("  Collecting window... ", end='', flush=True)
    while len(readings) < READINGS_PER_WINDOW:
        line = ser.readline().decode('utf-8', errors='ignore')
        if 'mq135' in line.lower():
            continue
        values = parse_line(line)
        if values is not None:
            readings.append(values)
            print(f"\r  Collecting window... {len(readings)}/{READINGS_PER_WINDOW}",
                  end='', flush=True)
    print()
    return np.mean(readings, axis=0)   # (7,)


def run():
    print("=" * 58)
    print("  CHICKEN FRESHNESS CLASSIFIER  (classifier only)")
    print("=" * 58)

    # ── Load model & scaler ──────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at '{MODEL_PATH}'")
        print("  Run 'python run_pipeline.py' first.")
        sys.exit(1)

    print("[Init] Loading LSTM model and scaler...")
    model  = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[Init] Loaded successfully.\n")

    # ── Connect to Arduino ───────────────────────────────────────────────────
    port = find_arduino_port()
    print(f"[Init] Connecting to {port}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=3)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open port '{port}': {e}")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()
    print("[Init] Connected!\n")

    # ── Clean-air baseline calibration ──────────────────────────────────────
    baseline = None
    if CALIBRATE_BASELINE:
        print("=" * 58)
        print("  STEP 1: CLEAN AIR BASELINE CALIBRATION")
        print("=" * 58)
        print("  Ensure chamber contains CLEAN AIR (no chicken).")
        print("  Collecting 1 calibration window (~60s)...\n")
        baseline = collect_one_window(ser)
        labels = ['MQ135', 'MQ2', 'MQ3', 'MQ6', 'MQ7', 'MQ9', 'Humid%']
        print(f"\n  ✓ Baseline S₀ captured:")
        print("    " + "  ".join(f"{l}:{v:.1f}" for l, v in zip(labels, baseline)))
        print("\n  Place the chicken sample in the chamber now.")
        print("=" * 58 + "\n")

    # ── Sliding window buffer ────────────────────────────────────────────────
    window_buffer = deque(maxlen=SEQ_LEN)
    window_count  = 0
    session_start = time.time()
    labels        = ['MQ135', 'MQ2', 'MQ3', 'MQ6', 'MQ7', 'MQ9', 'Humid%']

    print(f"Collecting first {SEQ_LEN} windows ({SEQ_LEN} min) before first prediction...\n")

    try:
        while True:
            window_count += 1
            elapsed_min = (time.time() - session_start) / 60
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Window #{window_count}  "
                  f"(Elapsed: {elapsed_min:.1f} min)")

            avg_window = collect_one_window(ser)

            # Delta transformation
            if baseline is not None:
                model_window = avg_window - baseline
                print("  Raw  → " + "  ".join(f"{l}:{v:.1f}"  for l, v in zip(labels, avg_window)))
                print("  ΔS   → " + "  ".join(f"{l}:{v:+.1f}" for l, v in zip(labels, model_window)))
            else:
                model_window = avg_window
                print("  Raw  → " + "  ".join(f"{l}:{v:.1f}" for l, v in zip(labels, avg_window)))

            window_buffer.append(model_window)

            # Wait until SEQ_LEN windows accumulated
            if len(window_buffer) < SEQ_LEN:
                remaining = SEQ_LEN - len(window_buffer)
                print(f"  ⏳ Need {remaining} more window(s) before first prediction...\n")
                continue

            # ── Build model input ────────────────────────────────────────────
            sequence        = np.array(list(window_buffer))       # (5, 7)
            sequence_scaled = scaler.transform(sequence)          # (5, 7)
            model_input     = sequence_scaled[np.newaxis, :, :]   # (1, 5, 7)

            # ── Predict (LSTM) ───────────────────────────────────────────────
            probs      = model.predict(model_input, verbose=0)[0]
            pred_class = int(np.argmax(probs))
            confidence = probs[pred_class] * 100

            # ── Sensor-threshold override ─────────────────────────────────────
            # The LSTM can miss spoilage when it hasn't seen a rising trend
            # (sensor saturation / already-spoiled samples). Use absolute ΔS
            # values from the last window to catch this case.
            last_delta = list(window_buffer)[-1]  # (7,) — already ΔS
            sensor_votes = [
                last_delta[5] >= SPOIL_THRESHOLDS['MQ9'],    # MQ9
                last_delta[3] >= SPOIL_THRESHOLDS['MQ6'],    # MQ6
                last_delta[4] >= SPOIL_THRESHOLDS['MQ7'],    # MQ7
                last_delta[0] >= SPOIL_THRESHOLDS['MQ135'],  # MQ135
                last_delta[1] >= SPOIL_THRESHOLDS['MQ2'],    # MQ2
            ]
            votes_for_spoiled = sum(sensor_votes)
            override_triggered = (votes_for_spoiled >= SPOIL_SENSOR_VOTE_MIN
                                  and pred_class < 2)  # only override if LSTM says Fresh/Semi

            if override_triggered:
                display_class = 2     # SPOILED
                display_icon  = CLASS_ICONS[2]
                display_name  = CLASS_NAMES[2]
                override_note = (f"  ║  ⚠ Sensor override: {votes_for_spoiled}/5 sensors in spoiled zone  ║")
            else:
                display_class = pred_class
                display_icon  = CLASS_ICONS[pred_class]
                display_name  = CLASS_NAMES[pred_class]
                override_note = None

            # Sensor vote details for transparency
            sensor_vote_names = ['MQ9', 'MQ6', 'MQ7', 'MQ135', 'MQ2']
            vote_str = '  '.join(
                f"{n}:{'✅' if v else '❌'}"
                for n, v in zip(sensor_vote_names, sensor_votes)
            )

            print(f"\n  ╔══════════════════════════════════════════════════════╗")
            print(f"  ║  RESULT     : {display_icon} {display_name:<40}║")
            # if override_triggered:
            #     print(f"  ║  LSTM said  : {CLASS_ICONS[pred_class]} {CLASS_NAMES[pred_class]} ({confidence:.1f}%) — overridden        ║")
            # else:
            #     print(f"  ║  Confidence : {confidence:.1f}%{' ' * (40 - len(f'{confidence:.1f}%'))}║")
            # print(f"  ╠══════════════════════════════════════════════════════╣")
            # print(f"  ║  LSTM probabilities:                                 ║")
            # print(f"  ║  🟢 Fresh: {probs[0]*100:5.1f}%  "
            #       f"🟡 Semi: {probs[1]*100:5.1f}%  "
            #       f"🔴 Spoiled: {probs[2]*100:5.1f}%      ║")
            print(f"  ║  Sensor spoilage votes ({votes_for_spoiled}/5 needed {SPOIL_SENSOR_VOTE_MIN}):              ║")
            print(f"  ║  {vote_str:<52}║")
            print(f"  ╚══════════════════════════════════════════════════════╝\n")

    except KeyboardInterrupt:
        print("\n[Stopped] Session ended by user.")
        ser.close()


if __name__ == '__main__':
    run()
