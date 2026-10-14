"""
Real-Time Chicken Freshness Detector (Option B — Sliding Window)
----------------------------------------------------------------
How it works:
  - Reads 12 Arduino readings every 60 seconds (one averaged window)
  - Maintains a sliding buffer of the last 5 windows (seq_len=5)
  - After the first 5 windows (5 minutes), predicts freshness every 60s
  - Prints: Fresh / Semi-Fresh / Spoiled with confidence %

Usage:
    python realtime_predict.py
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

# Ensure UTF-8 output on Windows consoles for emojis and box borders
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from shelf_life_estimator import predict_sample_shelf_life


# ── Configuration ──────────────────────────────────────────────────────────
BAUD_RATE       = 9600
MODEL_PATH      = 'saved_model/freshness_model.keras'
SCALER_PATH     = 'saved_model/scaler.pkl'
SEQ_LEN         = 5          # Must match seq_len used during training
READINGS_PER_WINDOW = 12     # 12 × 5s = 60 seconds per window
NUM_FEATURES    = 7          # mq135, mq2, mq3, mq6, mq7, mq9, humid
CLASS_NAMES         = ['🟢 FRESH', '🟡 SEMI-FRESH', '🔴 SPOILED']
CALIBRATE_BASELINE  = True       # Subtract clean air baseline (matches Anwar & Anwar 2025 ΔS dataset)
# ───────────────────────────────────────────────────────────────────────────


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
        # Order from Arduino: mq135, mq2, mq3, mq6, mq7, mq9, humid, temp
        # We use only the first 7 (drop temp — not in training features)
        values = [float(p) for p in parts[:7]]
        return values
    except ValueError:
        return None


def collect_one_window(ser):
    """
    Blocks until 12 valid readings are collected from Arduino.
    Returns a numpy array of shape (7,) — the averaged window.
    """
    readings = []
    print(f"  Collecting window... ", end='', flush=True)
    while len(readings) < READINGS_PER_WINDOW:
        line = ser.readline().decode('utf-8', errors='ignore')
        if 'mq135' in line.lower():   # skip header line
            continue
        values = parse_line(line)
        if values is not None:
            readings.append(values)
            print(f"\r  Collecting window... {len(readings)}/{READINGS_PER_WINDOW}", 
                  end='', flush=True)
    print()
    return np.mean(readings, axis=0)   # shape: (7,)


def run_realtime():
    # 1. Load model and scaler
    print("="*55)
    print("  CHICKEN FRESHNESS REAL-TIME DETECTOR (Option B)")
    print("="*55)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at '{MODEL_PATH}'")
        print("  Run 'python run_pipeline.py' first to train and save the model.")
        sys.exit(1)

    print("[Init] Loading model and scaler...")
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[Init] Model loaded successfully.")

    # ── NEW: Load Shelf-Life Model ─────────────────────────────────────────
    SHELF_MODEL_PATH = 'saved_model/shelf_life_model.pkl'
    if not os.path.exists(SHELF_MODEL_PATH):
        print("[ERROR] Shelf life model not found at "
            f"'{SHELF_MODEL_PATH}'\nRun 'run_pipeline.py' first.")
        sys.exit(1)

    print("[Init] Loading shelf-life regressor...")
    shelf_model = joblib.load(SHELF_MODEL_PATH)
    print("[Init] Shelf-life model loaded successfully.")
# ──────────────────────────────────────────────────────────────────────────
    # 2. Connect to Arduino
    port = find_arduino_port()
    print(f"[Init] Connecting to Arduino on {port}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=3)
    except serial.SerialException as e:
        print(f"\n[ERROR] Could not open port '{port}': {e}")
        if "PermissionError" in str(e) or "Access is denied" in str(e):
            print("  👉 The port is currently in use by another application.")
            print("  👉 Close the Serial Monitor / Serial Plotter inside Arduino IDE and try again.\n")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()
    print("[Init] Connected!\n")

    # 3. Clean Air Baseline Calibration (matches Anwar & Anwar 2025 Delta S dataset)
    baseline_window = None
    if CALIBRATE_BASELINE:
        print("="*55)
        print("  STEP 1: CLEAN AIR BASELINE CALIBRATION")
        print("="*55)
        print("  Ensure chamber contains CLEAN AIR (no chicken sample).")
        print("  Collecting 1 calibration window (60s) to establish S₀...\n")
        baseline_window = collect_one_window(ser)
        print(f"\n  ✓ Baseline S₀ established:")
        print(f"    MQ135:{baseline_window[0]:.1f} | MQ2:{baseline_window[1]:.1f} | "
              f"MQ3:{baseline_window[2]:.1f} | MQ6:{baseline_window[3]:.1f} | "
              f"MQ7:{baseline_window[4]:.1f} | MQ9:{baseline_window[5]:.1f} | "
              f"Humid:{baseline_window[6]:.1f}%\n")
        print("  All readings will now be evaluated as ΔS = S(t) - S₀.")
        print("  You may now place the chicken sample in the chamber.")
        print("="*55 + "\n")

    # 4. Sliding window buffer — holds last SEQ_LEN windows
    window_buffer = deque(maxlen=SEQ_LEN)   # each item: shape (7,)

    print(f"Collecting first {SEQ_LEN} windows ({SEQ_LEN} minutes) before first prediction...\n")

    window_count = 0
    session_start = time.time()

    try:
        while True:
            window_count += 1
            elapsed_min = (time.time() - session_start) / 60
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Window #{window_count} "
                  f"(Elapsed: {elapsed_min:.1f} min)")

            # Collect one averaged window (takes ~60 seconds)
            avg_window = collect_one_window(ser)

            # Compute delta relative to clean air baseline
            if baseline_window is not None:
                model_window = avg_window - baseline_window
                print(f"  Raw sensor    → MQ135:{avg_window[0]:.1f}  MQ2:{avg_window[1]:.1f}  "
                      f"MQ3:{avg_window[2]:.1f}  MQ6:{avg_window[3]:.1f}  "
                      f"MQ7:{avg_window[4]:.1f}  MQ9:{avg_window[5]:.1f}  "
                      f"Humid:{avg_window[6]:.1f}%")
                print(f"  Delta (ΔS)    → MQ135:{model_window[0]:+.1f}  MQ2:{model_window[1]:+.1f}  "
                      f"MQ3:{model_window[2]:+.1f}  MQ6:{model_window[3]:+.1f}  "
                      f"MQ7:{model_window[4]:+.1f}  MQ9:{model_window[5]:+.1f}  "
                      f"Humid:{model_window[6]:+.1f}%")
            else:
                model_window = avg_window
                print(f"  Raw averages  → MQ135:{avg_window[0]:.1f}  MQ2:{avg_window[1]:.1f}  "
                      f"MQ3:{avg_window[2]:.1f}  MQ6:{avg_window[3]:.1f}  "
                      f"MQ7:{avg_window[4]:.1f}  MQ9:{avg_window[5]:.1f}  "
                      f"Humid:{avg_window[6]:.1f}%")

            window_buffer.append(model_window)

            # Need SEQ_LEN windows before predicting
            if len(window_buffer) < SEQ_LEN:
                remaining = SEQ_LEN - len(window_buffer)
                print(f"  Waiting for {remaining} more window(s) before first prediction...\n")
                continue

            # 5. Prepare input: shape (1, SEQ_LEN, NUM_FEATURES)
            sequence = np.array(list(window_buffer))  # shape: (5, 7)

            # Scale using the saved scaler
            sequence_scaled = scaler.transform(sequence)            # shape: (5, 7)
            model_input = sequence_scaled[np.newaxis, :, :]        # shape: (1, 5, 7)
            # 5. Predict freshness class
            probs = model.predict(model_input, verbose=0)[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class] * 100

            # 6. Predict remaining shelf life using the last window (1D, shape (7,))
            last_window_1d = list(window_buffer)[-1]   # shape: (7,)
            shelf_info = predict_sample_shelf_life(shelf_model, last_window_1d)

            # 7. Display result
            print(f"\n  ╔══════════════════════════════════════════════════════╗")
            print(f"  ║  RESULT    : {CLASS_NAMES[pred_class]:<41}║")
            print(f"  ║  Confidence: {confidence:.1f}%                                   ║")
            print(f"  ╠══════════════════════════════════════════════════════╣")

            if pred_class < 2:   # Fresh or Semi-Fresh — show time remaining
                print(f"  ║  Remaining Safe Time : {shelf_info['Remaining Safe Window']:<30}║")
                print(f"  ║  Remaining Fresh Time: {shelf_info['Remaining Fresh Time']:<30}║")
                print(f"  ║  Chilled Shelf Life  : {shelf_info['Estimated Chilled Shelf-Life']:<30}║")
            else:                # Spoiled
                print(f"  ║  ⚠️  CHICKEN IS SPOILED — DO NOT CONSUME              ║")

            print(f"  ╚══════════════════════════════════════════════════════╝\n")


        
    except KeyboardInterrupt:
        print("\n[Stopped] Session ended by user.")
        ser.close()


if __name__ == '__main__':
    run_realtime()
