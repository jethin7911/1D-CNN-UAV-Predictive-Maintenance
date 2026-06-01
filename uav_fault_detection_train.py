
import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1 — CONFIGURATION
# Update the two dataset paths below to match your local drive.
# =============================================================================

# ── Dataset roots ─────────────────────────────────────────────────────────
REAL_SENSORS_PATH  = r"D:\FINAL YEAR PROJECT\Real-Sensors"
REAL_NO_FAULT_PATH = r"D:\FINAL YEAR PROJECT\Real-No_Fault"
MODEL_SAVE_PATH    = r"D:\FINAL YEAR PROJECT\models"

# ── Sliding window ────────────────────────────────────────────────────────
WINDOW_SIZE = 50                                   # 50 samples × 10 Hz = 5 s
OVERLAP     = 0.5                                  # 50 % overlap
STRIDE      = int(WINDOW_SIZE * (1 - OVERLAP))    # = 25 samples

# ── Training hyper-parameters ─────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 0.001
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15

# ── Resampling ────────────────────────────────────────────────────────────
TARGET_HZ       = 10
RESAMPLE_PERIOD = '100ms'   # 1 / 10 Hz

# ── 6-class label map ─────────────────────────────────────────────────────
FAULT_LABELS = {
    'no_fault':      0,
    'accelerometer': 1,
    'gyroscope':     2,
    'magnetometer':  3,
    'barometer':     4,
    'gps':           5,
}
LABEL_NAMES = ['No Fault', 'Accelerometer', 'Gyroscope',
               'Magnetometer', 'Barometer', 'GPS']

# ── Feature column names (post-merge) ─────────────────────────────────────
ACCEL_COLS = ['accel_x',  'accel_y',  'accel_z']
GYRO_COLS  = ['gyro_x',   'gyro_y',   'gyro_z']
MAG_COLS   = ['mag_x',    'mag_y',    'mag_z']
BARO_COLS  = ['baro_pressure', 'baro_temperature']
GPS_COLS   = ['gps_lat',  'gps_lon',  'gps_alt',
              'gps_vel',  'gps_vn',   'gps_ve',   'gps_vd']
MOTOR_COLS = ['motor_0',  'motor_1',  'motor_2',  'motor_3']
ATT_COLS   = ['att_roll', 'att_pitch','att_yaw']
POS_COLS   = ['pos_x',    'pos_y',    'pos_z',
              'vel_x',    'vel_y',    'vel_z']

ALL_FEATURES = (ACCEL_COLS + GYRO_COLS + MAG_COLS + BARO_COLS
                + GPS_COLS + MOTOR_COLS + ATT_COLS + POS_COLS)
N_FEATURES   = len(ALL_FEATURES)   # 31 channels


# =============================================================================
# SECTION 2 — UTILITY / MATH HELPERS
# =============================================================================

def butter_lowpass_filter(signal: np.ndarray,
                           cutoff: float = 4.0,
                           fs: float = TARGET_HZ,
                           order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter applied to IMU channels.

    After resampling to 10 Hz, the Nyquist limit is 5 Hz.
    A 4 Hz cutoff removes residual vibration artefacts while preserving
    all meaningful flight-dynamics frequency content (well below 4 Hz).

    filtfilt applies the filter twice (forward + backward) for zero phase lag.
    """
    nyq           = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)   # must stay strictly below 1.0
    b, a          = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def quaternion_to_euler(q0: np.ndarray, q1: np.ndarray,
                        q2: np.ndarray, q3: np.ndarray):
    """
    Convert PX4 unit quaternion [q0=w, q1=x, q2=y, q3=z] to
    Euler angles [roll, pitch, yaw] in radians using the ZYX convention.

    PX4 EKF2 outputs attitude as a Hamilton quaternion with scalar part first.
    """
    # Roll  (rotation around body x-axis)
    roll  = np.arctan2(2.0*(q0*q1 + q2*q3),
                       1.0 - 2.0*(q1**2 + q2**2))
    # Pitch (rotation around body y-axis — clamp to avoid arcsin domain error)
    sin_p = np.clip(2.0*(q0*q2 - q3*q1), -1.0, 1.0)
    pitch = np.arcsin(sin_p)
    # Yaw   (rotation around body z-axis)
    yaw   = np.arctan2(2.0*(q0*q3 + q1*q2),
                       1.0 - 2.0*(q2**2 + q3**2))
    return roll, pitch, yaw


def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PX4 microsecond timestamps to a pandas DatetimeIndex.
    Mandatory step before time-based resampling with pandas .resample().
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
    return df.set_index('datetime').sort_index()


def resample_topic(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Resample any topic DataFrame to TARGET_HZ (10 Hz / 100 ms).

    method='interpolate'
        For low- and matched-frequency topics:
        GPS (1 Hz), baro (10 Hz), local position (10 Hz), accel/gyro/mag (10 Hz).
        Resamples then linearly interpolates any gaps.

    method='mean'
        For high-frequency topics:
        actuator_outputs (~665 Hz), vehicle_attitude (~192 Hz).
        Takes the arithmetic mean of all samples in each 100 ms bin.
        This acts as a box-filter and prevents aliasing on downsampling.
    """
    if method == 'mean':
        return df.resample(RESAMPLE_PERIOD).mean()
    return (df.resample(RESAMPLE_PERIOD)
              .mean()
              .interpolate(method='linear', limit_direction='both'))


# =============================================================================
# SECTION 3 — FOLDER WALKER & FILE FINDER
# =============================================================================

def find_csv_by_suffix(log_folder: str, suffix: str):
    """
    Locate a single CSV inside log_folder whose filename ends with suffix.
    Works regardless of the date-stamped prefix (log_7_2023-...).
    Returns the full path, or None if not found.
    """
    matches = glob.glob(os.path.join(log_folder, f'*{suffix}'))
    return matches[0] if matches else None


def find_log_subfolder(flight_case_dir: str):
    """
    Find the log_7_* subfolder inside a flight case directory.
    This subfolder contains all 80 uORB-topic CSVs extracted from .ulg.
    Returns full path or None.
    """
    for entry in os.listdir(flight_case_dir):
        full = os.path.join(flight_case_dir, entry)
        if os.path.isdir(full) and entry.startswith('log_'):
            return full
    return None


def folder_name_to_fault_class(folder_name: str) -> int:
    """
    Map a {flight_status}-{fault_type} folder name to an integer fault class.

    Folder naming from the screenshots:
        acce-acce    → accelerometer fault (class 1)
        hover-baro   → barometer fault     (class 4)
        circling-GPS → GPS fault           (class 5)
        ... etc.

    Returns class 0 (no fault) for unrecognised names.
    """
    n = folder_name.lower()
    if any(k in n for k in ('no_fault', 'nofault', 'no-fault')):
        return FAULT_LABELS['no_fault']
    if n.endswith('-acce') or 'accelerometer' in n:
        return FAULT_LABELS['accelerometer']
    if n.endswith('-gyro') or 'gyroscope' in n:
        return FAULT_LABELS['gyroscope']
    if n.endswith('-mag') or 'magnetometer' in n:
        return FAULT_LABELS['magnetometer']
    if n.endswith('-baro') or 'barometer' in n:
        return FAULT_LABELS['barometer']
    if n.endswith('-gps'):
        return FAULT_LABELS['gps']
    return FAULT_LABELS['no_fault']


def collect_flight_cases(sensors_root: str, no_fault_root: str) -> list:
    """
    Recursively collect every valid flight case from both dataset roots.

    Expected directory tree (two levels deep below each root):
        sensors_root /
            acce-acce /           ← Level 1 : {status}-{fault_type}
                431_1 /           ← Level 2 : individual flight case ID
                    log_7_... /   ← Level 3 : per-topic CSVs (target)
                432_1 / ...
            hover-baro / ...
        no_fault_root /
            hover / (or similar structure)
                .../

    Returns a list of dicts with keys:
        log_folder  : str   full path to the log_* subfolder
        fault_class : int   integer class label 0–5
        case_id     : str   human-readable ID used in progress logging
    """
    cases = []

    def _walk(root: str, is_no_fault: bool):
        if not os.path.isdir(root):
            print(f"[WARNING] Directory not found: {root}")
            return

        for lvl1 in sorted(os.listdir(root)):           # {status}-{fault}
            lvl1_path = os.path.join(root, lvl1)
            if not os.path.isdir(lvl1_path):
                continue

            fault_class = (FAULT_LABELS['no_fault'] if is_no_fault
                           else folder_name_to_fault_class(lvl1))

            for lvl2 in sorted(os.listdir(lvl1_path)):  # flight case IDs
                lvl2_path = os.path.join(lvl1_path, lvl2)
                if not os.path.isdir(lvl2_path):
                    continue

                log_folder = find_log_subfolder(lvl2_path)
                if log_folder is None:
                    print(f"[SKIP] No log_* folder in: {lvl2_path}")
                    continue

                cases.append({
                    'log_folder':  log_folder,
                    'fault_class': fault_class,
                    'case_id':     f"{lvl1}/{lvl2}",
                })

    _walk(sensors_root,  is_no_fault=False)
    _walk(no_fault_root, is_no_fault=True)

    print(f"[INFO] Total flight cases found: {len(cases)}")
    return cases


# =============================================================================
# SECTION 4 — INDIVIDUAL TOPIC LOADERS
# Pattern: read CSV → keep required columns → datetime index
#          → resample to 10 Hz → rename columns for clarity.
# =============================================================================

def load_accel(log_folder: str):
    """Accelerometer: x, y, z [m/s²] — native log rate ~10 Hz."""
    f = find_csv_by_suffix(log_folder, '_sensor_accel_0.csv')
    if f is None:
        return None
    df = pd.read_csv(f, usecols=['timestamp', 'x', 'y', 'z'])
    df = to_datetime_index(df)
    df = resample_topic(df[['x', 'y', 'z']], method='interpolate')
    df.columns = ACCEL_COLS
    return df


def load_gyro(log_folder: str):
    """Gyroscope: x, y, z [rad/s] — native log rate ~10 Hz."""
    f = find_csv_by_suffix(log_folder, '_sensor_gyro_0.csv')
    if f is None:
        return None
    df = pd.read_csv(f, usecols=['timestamp', 'x', 'y', 'z'])
    df = to_datetime_index(df)
    df = resample_topic(df[['x', 'y', 'z']], method='interpolate')
    df.columns = GYRO_COLS
    return df


def load_mag(log_folder: str):
    """Magnetometer: x, y, z [Gauss] — native log rate ~10 Hz."""
    f = find_csv_by_suffix(log_folder, '_sensor_mag_0.csv')
    if f is None:
        return None
    df = pd.read_csv(f, usecols=['timestamp', 'x', 'y', 'z'])
    df = to_datetime_index(df)
    df = resample_topic(df[['x', 'y', 'z']], method='interpolate')
    df.columns = MAG_COLS
    return df


def load_baro(log_folder: str):
    """Barometer: pressure [hPa], temperature [°C] — native log rate ~10 Hz."""
    f = find_csv_by_suffix(log_folder, '_sensor_baro_0.csv')
    if f is None:
        return None
    df = pd.read_csv(f, usecols=['timestamp', 'pressure', 'temperature'])
    df = to_datetime_index(df)
    df = resample_topic(df[['pressure', 'temperature']], method='interpolate')
    df.columns = BARO_COLS
    return df


def load_gps(log_folder: str):
    """
    GPS: lat, lon, alt, velocities — native log rate ~1 Hz.

    PX4 stores lat/lon as signed integers scaled by 1e7 inside sensor_gps.
    e.g., 403119653 → 40.3119653 degrees.
    Upsampled from 1 Hz → 10 Hz via linear interpolation.
    """
    f = find_csv_by_suffix(log_folder, '_sensor_gps_0.csv')
    if f is None:
        return None
    cols = ['timestamp', 'lat', 'lon', 'alt',
            'vel_m_s', 'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']
    df = pd.read_csv(f, usecols=cols)
    df['lat'] = df['lat'] / 1e7   # integer → degrees
    df['lon'] = df['lon'] / 1e7
    df = to_datetime_index(df)
    df = resample_topic(
        df[['lat', 'lon', 'alt',
            'vel_m_s', 'vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']],
        method='interpolate'
    )
    df.columns = GPS_COLS
    return df


def load_actuator_outputs(log_folder: str):
    """
    Motor PWM commands: output[0..3] [µs, ~917–2000] — native log rate ~665 Hz.
    Downsampled to 10 Hz via mean (box filter).
    Only the first 4 outputs correspond to quadrotor motors.
    """
    f = find_csv_by_suffix(log_folder, '_actuator_outputs_0.csv')
    if f is None:
        return None
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


def load_attitude(log_folder: str):
    """
    Vehicle attitude quaternion from EKF2 — native log rate ~192 Hz.
    Downsampled to 10 Hz, then converted to Euler angles [roll, pitch, yaw].
    """
    f = find_csv_by_suffix(log_folder, '_vehicle_attitude_0.csv')
    if f is None:
        return None
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


def load_local_position(log_folder: str):
    """NED position (x,y,z) and velocity (vx,vy,vz) — native log rate ~10 Hz."""
    f = find_csv_by_suffix(log_folder, '_vehicle_local_position_0.csv')
    if f is None:
        return None
    df = pd.read_csv(f, usecols=['timestamp',
                                  'x', 'y', 'z', 'vx', 'vy', 'vz'])
    df = to_datetime_index(df)
    df = resample_topic(df[['x', 'y', 'z', 'vx', 'vy', 'vz']],
                        method='interpolate')
    df.columns = POS_COLS
    return df


def load_fault_flag(log_folder: str) -> pd.Series:
    """
    rfly_ctrl_lxl — custom uORB topic injected by RflySim for fault tagging.

    Decoded from our ZIP analysis:
        id == 1500  →  no fault active   →  fault_active = 0
        id != 1500  →  fault injected    →  fault_active = 1

    Labels are resampled with forward-fill (ffill) — never interpolate labels,
    as that would create fractional/intermediate values that do not exist.
    """
    f = find_csv_by_suffix(log_folder, '_rfly_ctrl_lxl_0.csv')
    if f is None:
        return None
    df = pd.read_csv(f, usecols=['timestamp', 'id'])
    df = to_datetime_index(df)
    df['fault_active'] = (df['id'] != 1500).astype(np.int8)
    return df['fault_active'].resample(RESAMPLE_PERIOD).ffill()


# =============================================================================
# SECTION 5 — FLIGHT CASE PROCESSOR
# =============================================================================

def process_one_case(log_folder: str, fault_class: int):
    """
    Process one complete flight case:

    1. Load all 9 required topic CSVs
    2. Apply Butterworth low-pass filter to IMU (accel + gyro) channels
    3. Inner-join all topics on a common 10 Hz datetime index
    4. Build per-sample class labels using rfly_ctrl_lxl timestamps:
           fault_active == 1  →  fault_class  (e.g., 1 for accelerometer)
           fault_active == 0  →  0            (no fault)
    5. Drop NaN rows; skip if flight is shorter than one window

    Returns
    -------
    features : np.ndarray  shape (T, N_FEATURES)  float32
    labels   : np.ndarray  shape (T,)             int32
    or (None, None) if the case must be skipped.
    """
    # ── Load topics ───────────────────────────────────────────────────────
    accel      = load_accel(log_folder)
    gyro       = load_gyro(log_folder)
    mag        = load_mag(log_folder)
    baro       = load_baro(log_folder)
    gps        = load_gps(log_folder)
    motors     = load_actuator_outputs(log_folder)
    att        = load_attitude(log_folder)
    pos        = load_local_position(log_folder)
    fault_flag = load_fault_flag(log_folder)

    # Skip if any critical topic failed to load
    if any(t is None for t in [accel, gyro, mag, baro, gps,
                                motors, att, pos, fault_flag]):
        print(f"  [SKIP] Missing topic — {log_folder}")
        return None, None

    # ── Low-pass filter IMU channels ──────────────────────────────────────
    # cutoff = 4 Hz (Nyquist at 10 Hz = 5 Hz; 4 Hz is safely within range)
    for col in ACCEL_COLS:
        accel[col] = butter_lowpass_filter(accel[col].values,
                                           cutoff=4.0, fs=TARGET_HZ)
    for col in GYRO_COLS:
        gyro[col]  = butter_lowpass_filter(gyro[col].values,
                                           cutoff=4.0, fs=TARGET_HZ)

    # ── Inner-join all topics on common datetime index ────────────────────
    # inner join keeps only rows where ALL sensors have valid data.
    merged = (accel
              .join(gyro,   how='inner')
              .join(mag,    how='inner')
              .join(baro,   how='inner')
              .join(gps,    how='inner')
              .join(motors, how='inner')
              .join(att,    how='inner')
              .join(pos,    how='inner'))

    # Align fault flag to the merged time index
    fault_aligned = fault_flag.reindex(merged.index, method='nearest')

    # Drop rows that still have NaN after joining
    valid         = merged.notna().all(axis=1) & fault_aligned.notna()
    merged        = merged[valid]
    fault_aligned = fault_aligned[valid]

    # Skip flight cases too short to produce even one window
    if len(merged) < WINDOW_SIZE:
        print(f"  [SKIP] Only {len(merged)} rows after merge — {log_folder}")
        return None, None

    # ── Build per-sample class labels ─────────────────────────────────────
    sample_labels = np.where(
        fault_aligned.values == 1,
        fault_class,              # e.g., 1 for accelerometer
        FAULT_LABELS['no_fault']  # 0
    ).astype(np.int32)

    return merged.values.astype(np.float32), sample_labels


# =============================================================================
# SECTION 6 — SLIDING WINDOW
# =============================================================================

def sliding_window(features: np.ndarray, labels: np.ndarray):
    """
    Convert a 2-D time-series (T × N_FEATURES) into overlapping windows
    for 1D-CNN input.

    Window size : 50 samples  (5 seconds at 10 Hz)
    Stride      : 25 samples  (50 % overlap → doubles the number of windows)

    Label strategy — CENTER SAMPLE:
        Each window is assigned the class of its middle (25th) sample.
        This is safer than majority-vote at fault onset/offset boundaries
        because it avoids creating ambiguously-labelled transition windows.

    Returns
    -------
    X : np.ndarray  (n_windows, WINDOW_SIZE, N_FEATURES)
    y : np.ndarray  (n_windows,)
    """
    X, y   = [], []
    center = WINDOW_SIZE // 2
    n      = len(features)

    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        X.append(features[start:end])
        y.append(labels[start + center])

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32))


# =============================================================================
# SECTION 7 — FULL DATASET BUILDER
# =============================================================================

def build_full_dataset():
    """
    Iterate over every flight case, process it, apply windowing,
    and concatenate into one final dataset.

    Returns
    -------
    X : np.ndarray  shape (N, WINDOW_SIZE, N_FEATURES)
    y : np.ndarray  shape (N,)
    """
    cases = collect_flight_cases(REAL_SENSORS_PATH, REAL_NO_FAULT_PATH)

    all_X, all_y = [], []

    for i, case in enumerate(cases):
        print(f"[{i+1:>3}/{len(cases)}] {case['case_id']}", end='  ')

        features, labels = process_one_case(
            case['log_folder'],
            case['fault_class']
        )
        if features is None:
            continue

        X_w, y_w = sliding_window(features, labels)
        if len(X_w) == 0:
            print("→ 0 windows")
            continue

        all_X.append(X_w)
        all_y.append(y_w)

        fault_wins = int(np.sum(y_w > 0))
        print(f"→ {len(X_w):>4} windows  "
              f"(fault={fault_wins}  normal={len(X_w)-fault_wins})")

    if not all_X:
        raise RuntimeError(
            "No valid flight cases processed. "
            "Check REAL_SENSORS_PATH and REAL_NO_FAULT_PATH."
        )

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Print class distribution summary
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"  Total windows : {X.shape[0]}")
    print(f"  Shape         : {X.shape}  (windows × timesteps × features)")
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        pct = 100.0 * n / len(y)
        print(f"  Class {c} ({LABEL_NAMES[c]:>15s}) : {n:>6d}  ({pct:.1f} %)")
    print(f"{'='*60}\n")

    return X, y


# =============================================================================
# SECTION 8 — 1D-CNN MODEL
# =============================================================================

def build_1d_cnn(input_shape: tuple, n_classes: int) -> tf.keras.Model:
    """
    Three-block 1D Convolutional Neural Network for multi-class FDI.

    Input  : (WINDOW_SIZE=50, N_FEATURES=31)

    Architecture
    ────────────
    Block 1 │ Conv1D(64,  k=3) → BatchNorm → ReLU → MaxPool(2)
    Block 2 │ Conv1D(128, k=3) → BatchNorm → ReLU → MaxPool(2)
    Block 3 │ Conv1D(256, k=3) → BatchNorm → ReLU
            │ GlobalAveragePooling1D
    Head    │ Dense(128, relu) → Dropout(0.5) → Dense(6, softmax)

    Design rationale
    ────────────────
    • Conv1D kernel_size=3 : captures local temporal patterns between adjacent
      sensor samples (300 ms at 10 Hz).
    • BatchNormalization : stabilises training; sensor channels have very
      different scales (PWM ~917–2000 vs gyro ~0.001–0.1).
    • MaxPooling : progressively increases temporal receptive field so that
      Block 3 sees patterns spanning ~1 second.
    • GlobalAveragePooling : replaces Flatten — fewer parameters, better
      generalisation on this moderate-sized dataset.
    • Dropout(0.5) : strong regularisation needed because total unique
      flight cases is only ~286.
    • Softmax : probability distribution over 6 fault classes.
    """
    inp = layers.Input(shape=input_shape, name='sensor_input')

    # Block 1
    x = layers.Conv1D(64,  kernel_size=3, padding='same', name='conv1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)

    # Block 2
    x = layers.Conv1D(128, kernel_size=3, padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool2')(x)

    # Block 3
    x = layers.Conv1D(256, kernel_size=3, padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)

    # Temporal aggregation
    x = layers.GlobalAveragePooling1D(name='gap')(x)

    # Classification head
    x   = layers.Dense(128, activation='relu', name='dense1')(x)
    x   = layers.Dropout(0.5, name='dropout')(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    return models.Model(inputs=inp, outputs=out,
                        name='UAV_FaultDetector_1DCNN')


# =============================================================================
# SECTION 9 — MAIN TRAINING PIPELINE
# =============================================================================

def train():
    """
    End-to-end training pipeline.

    Steps
    ─────
    1.  Build full windowed dataset from raw CSV files
    2.  Z-score normalise features (fitted on training set only)
    3.  Stratified train / val / test split (70 / 15 / 15 %)
    4.  Compute class weights to address fault / no-fault imbalance
    5.  Build and compile 1D-CNN
    6.  Train with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    7.  Evaluate on held-out test set
    8.  Plot confusion matrix and training curves
    9.  Save model (.keras) and scaler (.pkl)
    """
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # ── Step 1 : Build dataset ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1 — Building dataset from CSV files")
    print("="*60)
    X, y = build_full_dataset()

    # ── Step 2 : Normalise ────────────────────────────────────────────────
    print("STEP 2 — Z-score normalisation (per feature)")
    n_win, win_len, n_feat = X.shape

    # Split indices FIRST, then fit scaler on training data only.
    # Fitting on the full dataset would leak test statistics into training.
    indices = np.arange(n_win)
    idx_trainval, idx_test = train_test_split(
        indices, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_SIZE / (1.0 - TEST_SIZE),
        stratify=y[idx_trainval],
        random_state=42
    )

    # Fit scaler on training windows only
    scaler = StandardScaler()
    X_train_2d = X[idx_train].reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    # Transform all splits using training statistics
    X_train = scaler.transform(X[idx_train].reshape(-1, n_feat)
                               ).reshape(len(idx_train), win_len, n_feat)
    X_val   = scaler.transform(X[idx_val].reshape(-1, n_feat)
                               ).reshape(len(idx_val),   win_len, n_feat)
    X_test  = scaler.transform(X[idx_test].reshape(-1, n_feat)
                               ).reshape(len(idx_test),  win_len, n_feat)
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # Save scaler — must be loaded and applied identically at inference time
    scaler_path = os.path.join(MODEL_SAVE_PATH, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved → {scaler_path}")
    print(f"  Train : {len(X_train):>5}  |  "
          f"Val : {len(X_val):>5}  |  Test : {len(X_test):>5}  windows")

    # ── Step 3 : Class weights ─────────────────────────────────────────────
    print("\nSTEP 3 — Computing class weights (balanced)")
    classes_present   = np.unique(y_train)
    weights_array     = compute_class_weight(
        class_weight='balanced',
        classes=classes_present,
        y=y_train
    )
    class_weight_dict = dict(zip(classes_present.tolist(),
                                  weights_array.tolist()))
    for cls, w in class_weight_dict.items():
        print(f"  Class {cls} ({LABEL_NAMES[cls]:>15s}) : weight = {w:.3f}")

    # ── Step 4 : Build model ──────────────────────────────────────────────
    print("\nSTEP 4 — Building 1D-CNN model")
    n_classes = len(FAULT_LABELS)
    model     = build_1d_cnn(
        input_shape=(WINDOW_SIZE, N_FEATURES),
        n_classes=n_classes
    )
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ── Step 5 : Callbacks ────────────────────────────────────────────────
    ckpt_path = os.path.join(MODEL_SAVE_PATH, 'best_model.keras')
    cb_list = [
        # Stop when val_loss shows no improvement for 15 consecutive epochs.
        # restore_best_weights automatically rolls back to the best checkpoint.
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        ),
        # Halve the learning rate when val_loss plateaus for 7 epochs.
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
        # Persist the epoch with highest val_accuracy to disk.
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    # ── Step 6 : Train ────────────────────────────────────────────────────
    print("\nSTEP 5 — Training 1D-CNN")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=cb_list,
        verbose=1
    )

    # ── Step 7 : Evaluate ─────────────────────────────────────────────────
    print("\nSTEP 6 — Evaluating on held-out test set")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Accuracy : {test_acc*100:.2f} %")
    print(f"  Test Loss     : {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES,
        zero_division=0
    ))

    # ── Step 8a : Confusion matrix ────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        linewidths=0.5
    )
    plt.title('Confusion Matrix — UAV Sensor Fault Detection (1D-CNN)',
              fontsize=13, pad=12)
    plt.ylabel('True Label',      fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"  Confusion matrix → {cm_path}")

    # ── Step 8b : Training curves ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(history.history['accuracy'],     label='Train',      lw=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', lw=2)
    ax1.set_title('Accuracy',  fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train',      lw=2)
    ax2.plot(history.history['val_loss'], label='Validation', lw=2)
    ax2.set_title('Loss',      fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle('Training Curves — UAV 1D-CNN', fontsize=13)
    plt.tight_layout()
    curves_path = os.path.join(MODEL_SAVE_PATH, 'training_curves.png')
    plt.savefig(curves_path, dpi=150)
    plt.show()
    print(f"  Training curves  → {curves_path}")

    # ── Step 9 : Save ─────────────────────────────────────────────────────
    final_path = os.path.join(MODEL_SAVE_PATH, 'uav_fault_detector.keras')
    model.save(final_path)
    print(f"\n  Final model saved → {final_path}")
    print(f"  Scaler saved      → {scaler_path}")
    print("\n[DONE]")

    return model, scaler, history


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Fix random seeds for reproducibility across runs
    np.random.seed(42)
    tf.random.set_seed(42)

    model, scaler, history = train()
