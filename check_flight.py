#!/usr/bin/env python3
# =============================================================================
# check_flight.py — UAV Post-Flight Sensor Health Analysis Tool
# =============================================================================
# Usage:
#   python check_flight.py --ulg path/to/flight.ulg
#   python check_flight.py --ulg path/to/flight.ulg --model path/to/model.keras
#   python check_flight.py --ulg path/to/flight.ulg --output report.png
#
# What it does:
#   1. Accepts a raw PX4 .ulg file from the pilot via CLI argument
#   2. Extracts the 9 required uORB topic CSVs using pyulog
#   3. Runs the exact same 31-feature preprocessing pipeline used during training
#   4. Loads the saved .keras model and .pkl scaler
#   5. Runs sliding-window inference over the full flight
#   6. Generates a visual Health Report showing fault probabilities over time
#   7. Prints a clear maintenance summary to the terminal
#
# IMPORTANT — The model receives ONLY the preprocessed, normalised (50, 31)
# windows. The .keras file has NO preprocessing logic inside it. The scaler
# (.pkl) and all preprocessing steps in this file are mandatory at inference.
#
# Requirements:
#   pip install pyulog tensorflow scikit-learn scipy pandas numpy
#               matplotlib seaborn joblib
# =============================================================================

import os
import sys
import glob
import shutil
import argparse
import tempfile
import warnings
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import joblib
import tensorflow as tf

warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1 — CONFIGURATION
# These must exactly match the values used during training.
# =============================================================================

# ── Default paths (can be overridden via CLI arguments) ──────────────────────
DEFAULT_MODEL_PATH  = r"D:\FINAL YEAR PROJECT\models\uav_fault_detector.keras"
DEFAULT_SCALER_PATH = r"D:\FINAL YEAR PROJECT\models\feature_scaler.pkl"

# ── Sliding window — MUST match training ─────────────────────────────────────
WINDOW_SIZE = 50           # 50 samples = 5 seconds at 10 Hz
STRIDE      = 25           # 50% overlap

# ── Resampling — MUST match training ─────────────────────────────────────────
RESAMPLE_PERIOD = '100ms'  # 10 Hz master clock
TARGET_HZ       = 10

# ── Class definitions — MUST match training ───────────────────────────────────
LABEL_NAMES = ['No Fault', 'Accelerometer', 'Gyroscope',
               'Magnetometer', 'Barometer', 'GPS']

# Colour for each class in the health report chart
CLASS_COLOURS = [
    '#2ecc71',   # 0 No Fault     — green
    '#e74c3c',   # 1 Accelerometer— red
    '#e67e22',   # 2 Gyroscope    — orange
    '#9b59b6',   # 3 Magnetometer — purple
    '#3498db',   # 4 Barometer    — blue
    '#f39c12',   # 5 GPS          — amber
]

# Fault severity threshold — above this confidence a fault is flagged
FAULT_THRESHOLD = 0.60   # 60%

# ── Feature column layout — MUST match training ───────────────────────────────
ACCEL_COLS = ['accel_x',  'accel_y',  'accel_z']
GYRO_COLS  = ['gyro_x',   'gyro_y',   'gyro_z']
MAG_COLS   = ['mag_x',    'mag_y',    'mag_z']
BARO_COLS  = ['baro_pressure', 'baro_temperature']
GPS_COLS   = ['gps_lat',  'gps_lon',  'gps_alt',
              'gps_vel',  'gps_vn',   'gps_ve',  'gps_vd']
MOTOR_COLS = ['motor_0',  'motor_1',  'motor_2', 'motor_3']
ATT_COLS   = ['att_roll', 'att_pitch','att_yaw']
POS_COLS   = ['pos_x',    'pos_y',    'pos_z',
              'vel_x',    'vel_y',    'vel_z']

ALL_FEATURES = (ACCEL_COLS + GYRO_COLS + MAG_COLS + BARO_COLS
                + GPS_COLS  + MOTOR_COLS + ATT_COLS + POS_COLS)
N_FEATURES   = len(ALL_FEATURES)   # must be 31


# =============================================================================
# SECTION 2 — ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog='check_flight',
        description=(
            'UAV Post-Flight Sensor Health Analysis Tool\n'
            'Accepts a PX4 .ulg flight log and produces a visual health report.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--ulg', required=True,
        help='Path to the PX4 .ulg flight log file (required)'
    )
    parser.add_argument(
        '--model', default=DEFAULT_MODEL_PATH,
        help=f'Path to the trained .keras model file (default: {DEFAULT_MODEL_PATH})'
    )
    parser.add_argument(
        '--scaler', default=DEFAULT_SCALER_PATH,
        help=f'Path to the fitted .pkl scaler file (default: {DEFAULT_SCALER_PATH})'
    )
    parser.add_argument(
        '--output', default=None,
        help='Path to save the health report image (optional, default: show on screen)'
    )
    parser.add_argument(
        '--threshold', type=float, default=FAULT_THRESHOLD,
        help=f'Fault detection confidence threshold 0-1 (default: {FAULT_THRESHOLD})'
    )
    return parser.parse_args()


# =============================================================================
# SECTION 3 — ULG EXTRACTION
# Uses the pyulog command-line tool to extract CSVs from the .ulg file.
# =============================================================================

def extract_ulg_to_csv(ulg_path: str, output_dir: str) -> str:
    """
    Run pyulog's ulog2csv command to extract all uORB topics as CSV files
    into output_dir.

    pyulog installs a command-line entry point 'ulog2csv' that accepts:
        ulog2csv <file.ulg> -o <output_directory>

    Returns output_dir so callers can immediately locate the CSVs.
    Raises RuntimeError if pyulog is not installed or extraction fails.
    """
    print(f"[1/5] Extracting topics from: {ulg_path}")

    # Verify the file exists
    if not os.path.isfile(ulg_path):
        raise FileNotFoundError(f"ULG file not found: {ulg_path}")

    # Try pyulog's ulog2csv command
    try:
        result = subprocess.run(
            ['ulog2csv', ulg_path, '-o', output_dir],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ulog2csv failed:\n{result.stderr}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "ulog2csv not found. Install pyulog with: pip install pyulog"
        )

    print(f"      Topics extracted to: {output_dir}")
    return output_dir


def find_csv(csv_dir: str, suffix: str):
    """
    Find the first CSV file in csv_dir whose name ends with suffix.
    Returns full path or None.
    """
    matches = glob.glob(os.path.join(csv_dir, f'*{suffix}'))
    return matches[0] if matches else None


# =============================================================================
# SECTION 4 — PREPROCESSING HELPERS
# These are IDENTICAL to the training pipeline in uav_fault_detection_train.py.
# Any deviation here will corrupt model predictions.
# =============================================================================

def butter_lowpass_filter(signal: np.ndarray,
                           cutoff: float = 4.0,
                           fs: float = TARGET_HZ,
                           order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter for IMU channels.
    Identical parameters to training: cutoff=4 Hz, order=4, fs=10 Hz.
    """
    nyq           = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a          = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def quaternion_to_euler(q0, q1, q2, q3):
    """
    Convert PX4 quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw].
    Uses ZYX convention. Identical to training pipeline.
    """
    roll  = np.arctan2(2.0*(q0*q1 + q2*q3),
                       1.0 - 2.0*(q1**2 + q2**2))
    sin_p = np.clip(2.0*(q0*q2 - q3*q1), -1.0, 1.0)
    pitch = np.arcsin(sin_p)
    yaw   = np.arctan2(2.0*(q0*q3 + q1*q2),
                       1.0 - 2.0*(q2**2 + q3**2))
    return roll, pitch, yaw


def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert PX4 microsecond timestamp to DatetimeIndex."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
    return df.set_index('datetime').sort_index()


def resample_topic(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Resample a topic DataFrame to 10 Hz.
    method='mean'        → for high-rate topics (motors, attitude)
    method='interpolate' → for low- and matched-rate topics
    """
    if method == 'mean':
        return df.resample(RESAMPLE_PERIOD).mean()
    return (df.resample(RESAMPLE_PERIOD)
              .mean()
              .interpolate(method='linear', limit_direction='both'))


# =============================================================================
# SECTION 5 — INDIVIDUAL TOPIC LOADERS
# Identical to the training pipeline.
# =============================================================================

def load_accel(csv_dir):
    f = find_csv(csv_dir, '_sensor_accel_0.csv')
    if f is None: return None
    df = pd.read_csv(f, usecols=['timestamp', 'x', 'y', 'z'])
    df = to_datetime_index(df)
    df = resample_topic(df[['x', 'y', 'z']], method='interpolate')
    df.columns = ACCEL_COLS
    return df

def load_gyro(csv_dir):
    f = find_csv(csv_dir, '_sensor_gyro_0.csv')
    if f is None: return None
    df = pd.read_csv(f, usecols=['timestamp', 'x', 'y', 'z'])
    df = to_datetime_index(df)
    df = resample_topic(df[['x', 'y', 'z']], method='interpolate')
    df.columns = GYRO_COLS
    return df

def load_mag(csv_dir):
    f = find_csv(csv_dir, '_sensor_mag_0.csv')
    if f is None: return None
    df = pd.read_csv(f, usecols=['timestamp', 'x', 'y', 'z'])
    df = to_datetime_index(df)
    df = resample_topic(df[['x', 'y', 'z']], method='interpolate')
    df.columns = MAG_COLS
    return df

def load_baro(csv_dir):
    f = find_csv(csv_dir, '_sensor_baro_0.csv')
    if f is None: return None
    df = pd.read_csv(f, usecols=['timestamp', 'pressure', 'temperature'])
    df = to_datetime_index(df)
    df = resample_topic(df[['pressure', 'temperature']], method='interpolate')
    df.columns = BARO_COLS
    return df

def load_gps(csv_dir):
    f = find_csv(csv_dir, '_sensor_gps_0.csv')
    if f is None: return None
    cols = ['timestamp', 'lat', 'lon', 'alt',
            'vel_m_s', 'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']
    df = pd.read_csv(f, usecols=cols)
    df['lat'] = df['lat'] / 1e7    # integer → degrees
    df['lon'] = df['lon'] / 1e7
    df = to_datetime_index(df)
    df = resample_topic(
        df[['lat', 'lon', 'alt',
            'vel_m_s', 'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']],
        method='interpolate'
    )
    df.columns = GPS_COLS
    return df

def load_actuator_outputs(csv_dir):
    f = find_csv(csv_dir, '_actuator_outputs_0.csv')
    if f is None: return None
    df = pd.read_csv(f, usecols=['timestamp',
                                  'output[0]', 'output[1]',
                                  'output[2]', 'output[3]'])
    df = to_datetime_index(df)
    df = resample_topic(
        df[['output[0]', 'output[1]', 'output[2]', 'output[3]']],
        method='mean'
    )
    df.columns = MOTOR_COLS
    return df

def load_attitude(csv_dir):
    f = find_csv(csv_dir, '_vehicle_attitude_0.csv')
    if f is None: return None
    df = pd.read_csv(f, usecols=['timestamp',
                                  'q[0]', 'q[1]', 'q[2]', 'q[3]'])
    df = to_datetime_index(df)
    df = resample_topic(df[['q[0]', 'q[1]', 'q[2]', 'q[3]']], method='mean')
    roll, pitch, yaw = quaternion_to_euler(
        df['q[0]'].values, df['q[1]'].values,
        df['q[2]'].values, df['q[3]'].values
    )
    return pd.DataFrame(
        {'att_roll': roll, 'att_pitch': pitch, 'att_yaw': yaw},
        index=df.index
    )

def load_local_position(csv_dir):
    f = find_csv(csv_dir, '_vehicle_local_position_0.csv')
    if f is None: return None
    df = pd.read_csv(f, usecols=['timestamp',
                                  'x', 'y', 'z', 'vx', 'vy', 'vz'])
    df = to_datetime_index(df)
    df = resample_topic(df[['x', 'y', 'z', 'vx', 'vy', 'vz']],
                        method='interpolate')
    df.columns = POS_COLS
    return df


# =============================================================================
# SECTION 6 — FULL PREPROCESSING PIPELINE
# =============================================================================

def preprocess_flight(csv_dir: str):
    """
    Run the full 31-feature preprocessing pipeline on extracted CSVs.

    Steps:
      1. Load all 9 required topic CSVs
      2. Apply Butterworth low-pass filter to IMU (accel + gyro)
      3. Inner-join all topics on common 10 Hz datetime index
      4. Drop NaN rows
      5. Return merged DataFrame and its time axis in seconds

    Returns
    -------
    features     : np.ndarray  shape (T, 31)   float32  — NOT yet normalised
    time_seconds : np.ndarray  shape (T,)       float64  — seconds from flight start
    None, None if loading fails.
    """
    print("[2/5] Preprocessing sensor data...")

    accel  = load_accel(csv_dir)
    gyro   = load_gyro(csv_dir)
    mag    = load_mag(csv_dir)
    baro   = load_baro(csv_dir)
    gps    = load_gps(csv_dir)
    motors = load_actuator_outputs(csv_dir)
    att    = load_attitude(csv_dir)
    pos    = load_local_position(csv_dir)

    missing = [name for name, t in zip(
        ['accel','gyro','mag','baro','gps','motors','att','pos'],
        [accel, gyro, mag, baro, gps, motors, att, pos]
    ) if t is None]

    if missing:
        print(f"[ERROR] Missing required topics: {missing}")
        return None, None

    # Apply Butterworth low-pass filter to IMU (same as training)
    for col in ACCEL_COLS:
        accel[col] = butter_lowpass_filter(accel[col].values)
    for col in GYRO_COLS:
        gyro[col]  = butter_lowpass_filter(gyro[col].values)

    # Inner join — keep only rows where all topics have data
    merged = (accel
              .join(gyro,   how='inner')
              .join(mag,    how='inner')
              .join(baro,   how='inner')
              .join(gps,    how='inner')
              .join(motors, how='inner')
              .join(att,    how='inner')
              .join(pos,    how='inner'))

    # Drop any residual NaN rows
    merged = merged.dropna()

    if len(merged) < WINDOW_SIZE:
        print(f"[ERROR] Only {len(merged)} samples after merge — "
              f"minimum required is {WINDOW_SIZE}.")
        return None, None

    # Build a time axis in seconds from flight start (for plotting)
    time_ns      = merged.index.astype(np.int64)
    time_seconds = (time_ns - time_ns[0]) / 1e9

    print(f"      {len(merged)} samples  "
          f"({time_seconds[-1]:.1f} s flight duration)  "
          f"| {N_FEATURES} features")

    return merged[ALL_FEATURES].values.astype(np.float32), time_seconds.values


# =============================================================================
# SECTION 7 — NORMALISATION & WINDOWING
# =============================================================================

def normalise(features: np.ndarray, scaler) -> np.ndarray:
    """
    Apply the training-set Z-score statistics to the inference data.

    The scaler was fitted on training windows only during training.
    Here we simply call transform() — NEVER fit() at inference time.
    Refitting on new data would apply different statistics and corrupt predictions.
    """
    n_samples, n_feat = features.shape
    return scaler.transform(features).astype(np.float32)


def sliding_window_inference(features: np.ndarray):
    """
    Segment the normalised feature array into overlapping windows.
    Identical window size and stride to training.

    Returns
    -------
    windows    : np.ndarray  shape (N, 50, 31)
    center_idx : np.ndarray  shape (N,)  — original sample index of each window center
    """
    windows, center_idx = [], []
    n = len(features)
    center = WINDOW_SIZE // 2

    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        windows.append(features[start:end])
        center_idx.append(start + center)

    return (np.array(windows, dtype=np.float32),
            np.array(center_idx, dtype=np.int32))


# =============================================================================
# SECTION 8 — INFERENCE
# =============================================================================

def run_inference(model, windows: np.ndarray) -> np.ndarray:
    """
    Run the 1D-CNN on the batch of windows.

    model.predict() receives shape (N, 50, 31) and returns shape (N, 6).
    Each row is the softmax probability distribution over 6 fault classes.
    The model itself does NO preprocessing — it operates purely on the
    normalised numerical array passed to it.
    """
    print("[4/5] Running model inference...")
    probs = model.predict(windows, batch_size=64, verbose=0)
    print(f"      Predicted {len(probs)} windows")
    return probs   # shape (N, 6)


# =============================================================================
# SECTION 9 — HEALTH REPORT GENERATION
# =============================================================================

def build_health_report(probs: np.ndarray,
                         center_idx: np.ndarray,
                         time_seconds: np.ndarray,
                         ulg_filename: str,
                         threshold: float,
                         output_path: str = None):
    """
    Generate the visual flight health report.

    Layout (3-row figure):
      Row 1 — Stacked area chart: probability of each class over flight time
      Row 2 — Predicted fault class timeline (colour-coded bar per window)
      Row 3 — Summary panel: overall verdict + per-sensor anomaly flags

    Parameters
    ----------
    probs        : (N, 6) softmax probabilities per window
    center_idx   : (N,)   original sample index of each window center
    time_seconds : (T,)   time axis of the full merged feature array
    ulg_filename : str    filename of the input .ulg (for report title)
    threshold    : float  confidence above which a fault is flagged
    output_path  : str    save path, or None to display interactively
    """
    print("[5/5] Generating health report...")

    # Time axis for windows (center sample time in seconds)
    win_time   = time_seconds[center_idx]
    pred_class = np.argmax(probs, axis=1)
    max_conf   = np.max(probs, axis=1)

    # ── Derive summary statistics ─────────────────────────────────────────
    # For each fault class (1-5), find the maximum confidence across all windows
    fault_max_conf = {}
    fault_flagged  = {}
    for c in range(1, 6):
        conf              = float(np.max(probs[:, c]))
        fault_max_conf[c] = conf
        fault_flagged[c]  = conf >= threshold

    any_fault     = any(fault_flagged.values())
    overall_label = 'FAULT DETECTED' if any_fault else 'NORMAL OPERATION'
    overall_colour= '#e74c3c'         if any_fault else '#2ecc71'

    # Dominant fault class (highest max confidence among fault classes)
    dominant_class = max(range(1, 6), key=lambda c: fault_max_conf[c])
    dominant_conf  = fault_max_conf[dominant_class]

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[4, 1.5, 2],
        hspace=0.35
    )

    ax_prob    = fig.add_subplot(gs[0])
    ax_class   = fig.add_subplot(gs[1])
    ax_summary = fig.add_subplot(gs[2])

    for ax in [ax_prob, ax_class, ax_summary]:
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#0f3460')

    title_str = (
        f"UAV Flight Health Report  |  {os.path.basename(ulg_filename)}\n"
        f"Flight duration: {time_seconds[-1]:.1f}s  |  "
        f"Windows analysed: {len(probs)}  |  "
        f"Fault threshold: {threshold*100:.0f}%"
    )
    fig.suptitle(title_str, fontsize=11,
                 color='white', y=0.98, fontweight='bold')

    # ── Row 1 — Stacked probability area chart ────────────────────────────
    ax_prob.set_title('Sensor Fault Probability Over Flight Time',
                      color='white', fontsize=10, pad=8)

    # Stack probabilities so no-fault fills the bottom — faults on top
    bottom = np.zeros(len(win_time))
    for c in range(6):
        ax_prob.fill_between(
            win_time, bottom, bottom + probs[:, c],
            color=CLASS_COLOURS[c], alpha=0.85,
            label=LABEL_NAMES[c]
        )
        bottom += probs[:, c]

    # Threshold line
    ax_prob.axhline(y=threshold, color='white', linestyle='--',
                    linewidth=0.8, alpha=0.6, label=f'Threshold ({threshold*100:.0f}%)')

    ax_prob.set_xlim(win_time[0], win_time[-1])
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel('Probability', color='white', fontsize=9)
    ax_prob.set_xlabel('Time (seconds)', color='white', fontsize=9)
    ax_prob.tick_params(colors='white', labelsize=8)
    ax_prob.legend(loc='upper right', fontsize=8,
                   facecolor='#16213e', labelcolor='white',
                   framealpha=0.8, ncol=4)

    # ── Row 2 — Predicted class timeline ─────────────────────────────────
    ax_class.set_title('Predicted Fault Class per Window',
                       color='white', fontsize=10, pad=8)

    # Draw a coloured rectangle for each window
    bar_w = (win_time[-1] - win_time[0]) / len(win_time) * 0.9 if len(win_time) > 1 else 0.5
    for i, (t, c) in enumerate(zip(win_time, pred_class)):
        ax_class.add_patch(
            plt.Rectangle((t - bar_w/2, 0), bar_w, 1,
                           color=CLASS_COLOURS[c], alpha=0.9)
        )

    ax_class.set_xlim(win_time[0], win_time[-1])
    ax_class.set_ylim(0, 1)
    ax_class.set_ylabel('Class', color='white', fontsize=9)
    ax_class.set_xlabel('Time (seconds)', color='white', fontsize=9)
    ax_class.set_yticks([])
    ax_class.tick_params(colors='white', labelsize=8)

    # Class colour legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=CLASS_COLOURS[c],
                            label=LABEL_NAMES[c]) for c in range(6)]
    ax_class.legend(handles=legend_patches, loc='upper right',
                    fontsize=7, facecolor='#16213e',
                    labelcolor='white', framealpha=0.8, ncol=6)

    # ── Row 3 — Summary panel ─────────────────────────────────────────────
    ax_summary.axis('off')

    # Overall verdict box
    verdict_box = FancyBboxPatch(
        (0.0, 0.55), 0.32, 0.40,
        boxstyle="round,pad=0.02",
        facecolor=overall_colour, edgecolor='white',
        linewidth=1.5, transform=ax_summary.transAxes
    )
    ax_summary.add_patch(verdict_box)
    ax_summary.text(
        0.16, 0.76, overall_label,
        ha='center', va='center',
        fontsize=13, fontweight='bold',
        color='white', transform=ax_summary.transAxes
    )
    if any_fault:
        ax_summary.text(
            0.16, 0.61,
            f"Primary: {LABEL_NAMES[dominant_class]}  "
            f"({dominant_conf*100:.1f}% confidence)",
            ha='center', va='center',
            fontsize=8.5, color='white',
            transform=ax_summary.transAxes
        )

    # Per-sensor anomaly flags
    ax_summary.text(0.36, 0.92, 'Sensor Anomaly Flags',
                    ha='left', va='center', fontsize=9,
                    fontweight='bold', color='white',
                    transform=ax_summary.transAxes)

    x_positions = [0.36, 0.52, 0.63, 0.74, 0.86]
    for i, c in enumerate(range(1, 6)):
        flagged = fault_flagged[c]
        conf    = fault_max_conf[c]
        colour  = CLASS_COLOURS[c] if flagged else '#555555'
        status  = 'ANOMALY' if flagged else 'OK'

        box = FancyBboxPatch(
            (x_positions[i], 0.05), 0.10, 0.78,
            boxstyle="round,pad=0.02",
            facecolor=colour, edgecolor='white',
            linewidth=0.8 if flagged else 0.3,
            alpha=0.9 if flagged else 0.4,
            transform=ax_summary.transAxes
        )
        ax_summary.add_patch(box)
        ax_summary.text(
            x_positions[i] + 0.05, 0.72,
            LABEL_NAMES[c],
            ha='center', va='center', fontsize=7.5,
            fontweight='bold', color='white',
            transform=ax_summary.transAxes
        )
        ax_summary.text(
            x_positions[i] + 0.05, 0.48,
            status,
            ha='center', va='center', fontsize=9,
            fontweight='bold',
            color='white' if flagged else '#aaaaaa',
            transform=ax_summary.transAxes
        )
        ax_summary.text(
            x_positions[i] + 0.05, 0.22,
            f"max: {conf*100:.1f}%",
            ha='center', va='center', fontsize=7.5,
            color='white' if flagged else '#888888',
            transform=ax_summary.transAxes
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"      Report saved → {output_path}")
    else:
        plt.show()

    return fig


# =============================================================================
# SECTION 10 — TERMINAL SUMMARY PRINTOUT
# =============================================================================

def print_terminal_summary(probs: np.ndarray, threshold: float,
                             time_seconds: np.ndarray, ulg_path: str):
    """
    Print a clean maintenance summary to the terminal after inference.
    """
    SEP  = "=" * 60
    sep2 = "-" * 60

    pred_class = np.argmax(probs, axis=1)
    any_fault  = any(
        np.max(probs[:, c]) >= threshold for c in range(1, 6)
    )

    print(f"\n{SEP}")
    print(f"  UAV FLIGHT HEALTH SUMMARY")
    print(f"  File    : {os.path.basename(ulg_path)}")
    print(f"  Duration: {time_seconds[-1]:.1f} seconds")
    print(f"  Windows : {len(probs)}")
    print(SEP)

    if any_fault:
        print("  OVERALL STATUS : *** FAULT DETECTED ***")
    else:
        print("  OVERALL STATUS : NORMAL OPERATION")

    print(sep2)
    print(f"  {'Sensor':<18}  {'Max Confidence':>14}  {'Status':>12}")
    print(sep2)

    for c in range(1, 6):
        conf    = float(np.max(probs[:, c]))
        flagged = conf >= threshold
        status  = "[ ANOMALY ]" if flagged else "OK"
        marker  = " <--" if flagged else ""
        print(f"  {LABEL_NAMES[c]:<18}  {conf*100:>13.1f}%  "
              f"{status:>12}{marker}")

    print(sep2)
    # Show the window with the highest fault confidence
    fault_conf = 1.0 - probs[:, 0]  # confidence that something is wrong
    peak_idx   = int(np.argmax(fault_conf))
    peak_class = int(pred_class[peak_idx])
    peak_conf  = float(probs[peak_idx, peak_class])
    print(f"  Peak anomaly at window {peak_idx} "
          f"→ {LABEL_NAMES[peak_class]} ({peak_conf*100:.1f}%)")
    print(SEP + "\n")


# =============================================================================
# SECTION 11 — MAIN ENTRY POINT
# =============================================================================

def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  UAV POST-FLIGHT SENSOR HEALTH ANALYSIS TOOL")
    print("="*60)

    # ── Validate input paths ─────────────────────────────────────────────
    if not os.path.isfile(args.ulg):
        print(f"[ERROR] ULG file not found: {args.ulg}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.scaler):
        print(f"[ERROR] Scaler file not found: {args.scaler}")
        sys.exit(1)

    # ── Load model and scaler ────────────────────────────────────────────
    print("[0/5] Loading model and scaler...")
    model  = tf.keras.models.load_model(args.model)
    scaler = joblib.load(args.scaler)
    print(f"      Model  : {args.model}")
    print(f"      Scaler : {args.scaler}")
    print(f"      Input shape expected: {model.input_shape}")

    # Validate model input shape matches our pipeline
    expected_shape = (None, WINDOW_SIZE, N_FEATURES)
    if model.input_shape != expected_shape:
        print(f"[ERROR] Model input shape mismatch!\n"
              f"  Model expects : {model.input_shape}\n"
              f"  Pipeline gives: {expected_shape}\n"
              f"  Ensure WINDOW_SIZE={WINDOW_SIZE} and "
              f"N_FEATURES={N_FEATURES} match training config.")
        sys.exit(1)

    # ── Extract CSVs from .ulg into a temp directory ─────────────────────
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            extract_ulg_to_csv(args.ulg, tmp_dir)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"[ERROR] ULG extraction failed: {e}")
            sys.exit(1)

        # ── Preprocess ───────────────────────────────────────────────────
        features, time_seconds = preprocess_flight(tmp_dir)
        if features is None:
            print("[ERROR] Preprocessing failed. Exiting.")
            sys.exit(1)

    # ── Normalise using saved training scaler ────────────────────────────
    # This is NOT fitting — only transforming using training-set statistics.
    features_norm = normalise(features, scaler)

    # ── Sliding window segmentation ──────────────────────────────────────
    print("[3/5] Segmenting into windows...")
    windows, center_idx = sliding_window_inference(features_norm)
    print(f"      {len(windows)} windows  (size={WINDOW_SIZE}, stride={STRIDE})")

    if len(windows) == 0:
        print("[ERROR] No windows produced. Flight too short.")
        sys.exit(1)

    # ── Model inference ──────────────────────────────────────────────────
    # The model receives a clean (N, 50, 31) float32 array.
    # It outputs (N, 6) softmax probabilities.
    # No preprocessing happens inside the model.
    probs = run_inference(model, windows)

    # ── Terminal summary ─────────────────────────────────────────────────
    print_terminal_summary(probs, args.threshold, time_seconds, args.ulg)

    # ── Visual health report ─────────────────────────────────────────────
    build_health_report(
        probs        = probs,
        center_idx   = center_idx,
        time_seconds = time_seconds,
        ulg_filename = args.ulg,
        threshold    = args.threshold,
        output_path  = args.output
    )


if __name__ == '__main__':
    main()
