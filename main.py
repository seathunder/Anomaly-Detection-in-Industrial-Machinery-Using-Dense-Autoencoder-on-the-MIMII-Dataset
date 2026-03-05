#!/usr/bin/env python3
"""
mimii_baseline_rewrite.py

Cleaned, fixed, and ready-to-run version with:
 - deterministic manual_test creation (moves files once during training)
 - infer mode that loads model/scaler/status and accepts absolute paths
 - CLI that accepts pasted Windows paths (quotes/backslashes handled)
 - batch infer option to classify all files under manual_test and print summary
 - robust fallbacks and clear logging
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import glob
import pickle
from typing import List, Tuple, Dict, Optional

import numpy as np
import yaml
from tqdm import tqdm
from sklearn import metrics

import librosa
import hashlib
import random
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import shutil

# --------------------------- Utilities & Logging ---------------------------

def setup_logging(logfile: str = "baseline.log", level: int = logging.INFO) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)
    # Clear handlers
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

def save_pickle(path: str, data) -> None:
    logging.info("Saving pickle: %s", path)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path: str):
    logging.info("Loading pickle: %s", path)
    with open(path, "rb") as f:
        return pickle.load(f)

def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------- Audio IO / Features --------------------------

def load_wav(path: str, mono: bool = True) -> Tuple[np.ndarray, int]:
    try:
        y, sr = librosa.load(path, sr=None, mono=mono)
        return y, sr
    except Exception as e:
        logging.warning("Could not load %s: %s", path, e)
        raise

def demux_wav(path: str, channel: int = 0) -> Tuple[int, np.ndarray]:
    y, sr = load_wav(path, mono=False)
    arr = np.asarray(y)
    if arr.ndim == 1:
        return sr, arr
    if channel >= arr.shape[0]:
        logging.warning("Requested channel %d but audio has %d channels. Using channel 0.", channel, arr.shape[0])
        channel = 0
    return sr, arr[channel, :]

def file_to_vector_array(
    filename: str,
    n_mels: int = 64,
    frames: int = 5,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: float = 2.0,
    cache_mels_dir: Optional[str] = None,
) -> np.ndarray:
    try:
        sr, y = demux_wav(filename)
    except Exception:
        logging.warning("Skipping file due to load error: %s", filename)
        return np.empty((0, n_mels * frames), dtype=np.float32)

    log_mel = None
    if cache_mels_dir:
        os.makedirs(cache_mels_dir, exist_ok=True)
        key = hashlib.sha1(filename.encode("utf-8")).hexdigest()
        key += f"_nm{n_mels}_fr{frames}_nf{n_fft}_hl{hop_length}_p{int(power)}"
        cache_path = os.path.join(cache_mels_dir, f"{key}.npz")
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path) as npz:
                    log_mel = npz["log_mel"]
            except Exception:
                logging.warning("Failed to load cached mel for %s, will recompute.", filename)

    if log_mel is None:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        if cache_mels_dir:
            try:
                np.savez_compressed(cache_path, log_mel=log_mel)
            except Exception:
                logging.warning("Failed to write mel cache for %s", filename)

    vectorarray_size = log_mel.shape[1] - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, n_mels * frames), dtype=np.float32)

    dims = n_mels * frames
    out = np.zeros((vectorarray_size, dims), dtype=np.float32)
    for t in range(frames):
        out[:, n_mels * t : n_mels * (t + 1)] = log_mel[:, t : t + vectorarray_size].T

    return out

def estimate_windows_by_duration(path: str, frames: int, hop_length: int, n_fft: int) -> int:
    try:
        sr, y = demux_wav(path)
    except Exception:
        return 0
    n_samples = len(y)
    if n_samples <= n_fft:
        approx_cols = 0
    else:
        approx_cols = int(np.floor((n_samples - n_fft) / float(hop_length))) + 1
    vectorarray_size = max(0, approx_cols - frames + 1)
    return vectorarray_size

def count_windows_for_files(file_list: List[str], feat_kwargs: dict) -> int:
    total = 0
    fast = feat_kwargs.get("estimate_windows_fast", False)
    for f in file_list:
        try:
            if fast:
                vectorarray_size = estimate_windows_by_duration(
                    f,
                    frames=feat_kwargs.get("frames", 5),
                    hop_length=feat_kwargs.get("hop_length", 512),
                    n_fft=feat_kwargs.get("n_fft", 1024),
                )
            else:
                sr, y = demux_wav(f)
                mel = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_fft=feat_kwargs.get("n_fft", 1024),
                    hop_length=feat_kwargs.get("hop_length", 512),
                    n_mels=feat_kwargs.get("n_mels", 64),
                    power=feat_kwargs.get("power", 2.0),
                )
                vectorarray_size = mel.shape[1] - feat_kwargs.get("frames", 5) + 1
                if vectorarray_size < 0:
                    vectorarray_size = 0
            if vectorarray_size > 0:
                total += vectorarray_size
        except Exception:
            logging.warning("Skipping file in counting pass: %s", f)
    return total

def build_dataset_from_files(file_list: List[str], feat_kwargs: dict) -> np.ndarray:
    if len(file_list) == 0:
        return np.empty((0, feat_kwargs.get("n_mels",64) * feat_kwargs.get("frames",5)), dtype=np.float32)

    logging.info("Counting windows for %d files...", len(file_list))
    total_windows = count_windows_for_files(file_list, feat_kwargs)
    dims = feat_kwargs.get("n_mels", 64) * feat_kwargs.get("frames", 5)
    if total_windows == 0:
        logging.warning("No windows could be generated from files; returning empty dataset.")
        return np.empty((0, dims), dtype=np.float32)

    logging.info("Allocating dataset array: windows=%d, dims=%d", total_windows, dims)
    dataset = np.zeros((total_windows, dims), dtype=np.float32)

    idx = 0
    for f in tqdm(file_list, desc="extract features"):
        arr = file_to_vector_array(f, **feat_kwargs)
        if arr.shape[0] == 0:
            continue
        dataset[idx: idx + arr.shape[0], :] = arr
        idx += arr.shape[0]

    if idx < total_windows:
        dataset = dataset[:idx, :]

    return dataset

# ------------------------- Dataset generator / splitter --------------------

def dataset_generator(
    target_dir: str,
    normal_dir_name: str = "normal",
    abnormal_dir_name: str = "abnormal",
    ext: str = "wav",
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    base = os.path.abspath(target_dir)
    normal_files = sorted(glob.glob(os.path.join(base, normal_dir_name, f"*.{ext}")))
    abnormal_files = sorted(glob.glob(os.path.join(base, abnormal_dir_name, f"*.{ext}")))

    if not normal_files:
        raise FileNotFoundError(f"No normal files under {os.path.join(base, normal_dir_name)}")
    if not abnormal_files:
        raise FileNotFoundError(f"No abnormal files under {os.path.join(base, abnormal_dir_name)}")

    n_ab = len(abnormal_files)
    train_files = normal_files[n_ab:]
    train_labels = np.zeros(len(train_files), dtype=int)

    eval_files = normal_files[:n_ab] + abnormal_files
    eval_labels = np.concatenate((np.zeros(len(normal_files[:n_ab]), dtype=int), np.ones(n_ab, dtype=int)))

    logging.info("Dataset generator: target=%s | normal=%d | abnormal=%d | train=%d | eval=%d",
                 target_dir, len(normal_files), len(abnormal_files), len(train_files), len(eval_files))
    return train_files, train_labels, eval_files, eval_labels

def build_autoencoder(input_dim: int, bottleneck_dim: int = 32, noise_std: float = 0.05) -> Model:
    inp = Input(shape=(input_dim,))
    x = GaussianNoise(noise_std)(inp)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(bottleneck_dim, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    out = Dense(input_dim, activation=None)(x)
    model = Model(inputs=inp, outputs=out)
    return model

def train_autoencoder(
    model: Model,
    train_data: np.ndarray,
    model_path: str,
    epochs: int = 120,
    batch_size: int = 128,
    validation_split: float = 0.1,
    patience: int = 20,
    verbose: int = 1,
    use_tfdata: bool = False,
) -> Dict:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.LogCosh()
    )

    if model_path.endswith(".weights.h5"):
        model_path_full = model_path.replace(".weights.h5", ".h5")
    else:
        model_path_full = model_path + ".h5"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(model_path_full, monitor="val_loss", save_best_only=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, save_weights_only=True),
    ]

    if not use_tfdata:
        history = model.fit(
            train_data,
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        return {"history": history.history}

    n = train_data.shape[0]
    n_val = int(n * validation_split)
    idx = np.random.permutation(n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_data[train_idx].astype(np.float32))
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(train_data[val_idx].astype(np.float32))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    return {"history": history.history}

# ----------------------------- Evaluation ---------------------------------

def compute_anomaly_scores_for_files(
    model: Model,
    file_list: List[str],
    feat_kwargs: dict,
    batch_size: int = 256,
    scaler: Optional[StandardScaler] = None,
    scoring: str = "percentile",
    percentile: int = 95,
    normalize_windows: bool = True,
    hybrid_weights: Tuple[float, float] = (0.7, 0.3),
) -> np.ndarray:
    scores = np.zeros(len(file_list), dtype=float)
    for i, fname in enumerate(tqdm(file_list, desc="eval files")):
        data = file_to_vector_array(fname, **feat_kwargs)
        if data.shape[0] == 0:
            logging.warning("No features for %s -> setting large score", fname)
            scores[i] = 1e6
            continue

        if normalize_windows:
            try:
                data = normalize_windows_energy(data)
            except Exception as e:
                logging.warning("Window normalization failed for %s: %s", fname, e)

        if scaler is not None:
            try:
                data = scaler.transform(data)
            except Exception as e:
                logging.warning("Scaler transform failed for %s: %s. Using unscaled data.", fname, e)

        ds = tf.data.Dataset.from_tensor_slices(data.astype(np.float32)).batch(batch_size)
        recon = model.predict(ds, verbose=0)
        errs = np.mean(np.square(data - recon), axis=1)

        if scoring == "mean":
            scores[i] = float(np.mean(errs))
        elif scoring == "max":
            scores[i] = float(np.max(errs))
        elif scoring == "percentile":
            scores[i] = float(np.percentile(errs, percentile))
        elif scoring == "hybrid":
            w_pct, w_mean = hybrid_weights
            pct_val = float(np.percentile(errs, percentile))
            mean_val = float(np.mean(errs))
            scores[i] = w_pct * pct_val + w_mean * mean_val
        else:
            scores[i] = float(np.percentile(errs, percentile))
    return scores

def normalize_windows_energy(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if X is None or X.size == 0:
        return X
    energy = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (energy + eps)

def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    auc = metrics.roc_auc_score(y_true, y_score)
    try:
        aupr = metrics.average_precision_score(y_true, y_score)
    except Exception:
        aupr = float("nan")
    return {"AUC": float(auc), "AUPR": float(aupr)}

# ------------------------------ Config / Main ------------------------------

def load_config(config_path: Optional[str]) -> Dict:
    default = {
        "feature": {"n_mels": 64, "frames": 5, "n_fft": 1024, "hop_length": 512, "power": 2.0},
        "fit": {"epochs": 120, "batch_size": 128, "validation_split": 0.1, "patience": 20},
        "paths": {"pickle_directory": "./pickle", "model_directory": "./model", "result_directory": "./result"},
        "logging": {"logfile": "baseline.log"},
        "run": {"batch_predict_size": 256}
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if k in default and isinstance(default[k], dict):
                default[k].update(v)
            else:
                default[k] = v
    return default

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass
    if os.environ.get("TF_DETERMINISTIC_OPS", None) is None:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass

def fit_and_save_scaler(train_data: np.ndarray, scaler_path: str) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_data)
    save_pickle(scaler_path, scaler)
    logging.info("Saved scaler to %s", scaler_path)
    return scaler

def load_scaler(scaler_path: str) -> Optional[StandardScaler]:
    if not os.path.exists(scaler_path):
        logging.warning("Scaler path not found: %s", scaler_path)
        return None
    try:
        scaler = load_pickle(scaler_path)
        return scaler
    except Exception as e:
        logging.warning("Failed to load scaler %s: %s", scaler_path, e)
        return None

def make_train_dataset(np_array: np.ndarray, batch_size: int, shuffle_buffer: int = 10000) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(np_array.astype(np.float32))
    ds = ds.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def create_manual_test_split(
    id_dirs: List[str],
    output_dir: str,
    n_normal_per_id: int = 5,
    n_abnormal_per_id: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Selects a deterministic random set of files (per-seed), MOVES them into output_dir,
    and returns a mapping of held_out original paths.

    Returned dict structure:
      held_out[id_name] = {"normal": [orig_path1, ...], "abnormal": [orig_path1, ...]}

    Also writes a manifest (mapping moved_path -> original_path) inside output_dir/manifest.pickle
    """
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)

    held_out = {}
    moved_map: Dict[str, str] = {}  # moved_path -> original_path (for restore)

    for id_dir in id_dirs:
        id_name = os.path.basename(id_dir)

        normal_files = sorted(glob.glob(os.path.join(id_dir, "normal", "*.wav")))
        abnormal_files = sorted(glob.glob(os.path.join(id_dir, "abnormal", "*.wav")))

        if len(normal_files) < n_normal_per_id:
            raise ValueError(f"Not enough normal files in {id_name} to select {n_normal_per_id} samples.")
        if len(abnormal_files) < n_abnormal_per_id:
            raise ValueError(f"Not enough abnormal files in {id_name} to select {n_abnormal_per_id} samples.")

        # deterministic shuffle using RNG instance
        rng.shuffle(normal_files)
        rng.shuffle(abnormal_files)

        sel_normal = normal_files[:n_normal_per_id]
        sel_abnormal = abnormal_files[:n_abnormal_per_id]

        held_out[id_name] = {
            "normal": sel_normal.copy(),
            "abnormal": sel_abnormal.copy(),
        }

        # Move files to manual_test folder (maintain id/class structure)
        for cls, files in [("normal", sel_normal), ("abnormal", sel_abnormal)]:
            dst_dir = os.path.join(output_dir, id_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for f in files:
                fname = os.path.basename(f)
                dst_path = os.path.join(dst_dir, fname)
                # Move the file (safe move: if dst exists, create a unique suffix)
                if os.path.exists(dst_path):
                    # find a non-colliding name
                    base, ext = os.path.splitext(fname)
                    i = 1
                    while True:
                        alt = f"{base}.dup{i}{ext}"
                        alt_path = os.path.join(dst_dir, alt)
                        if not os.path.exists(alt_path):
                            dst_path = alt_path
                            break
                        i += 1
                try:
                    shutil.move(f, dst_path)
                    moved_map[dst_path] = f  # record original
                except Exception as e:
                    logging.exception("Failed to move %s -> %s: %s", f, dst_path, e)
                    raise

    # write manifest for restore
    manifest_path = os.path.join(output_dir, "manifest.pickle")
    manifest = {"held_out": held_out, "moved_map": moved_map}
    try:
        save_pickle(manifest_path, manifest)
    except Exception:
        logging.warning("Failed to save manifest at %s", manifest_path)

    logging.info("Manual test set created (moved files) at %s", output_dir)
    return held_out

def restore_manual_test_split(manual_test_dir: str) -> None:
    """
    Restore moved files back to their original paths using manifest.pickle.
    If original path already exists, the restore will skip and log a warning.
    """
    manifest_path = os.path.join(manual_test_dir, "manifest.pickle")
    if not os.path.exists(manifest_path):
        logging.warning("No manifest found at %s. Nothing to restore.", manifest_path)
        return

    try:
        manifest = load_pickle(manifest_path)
    except Exception as e:
        logging.exception("Failed to load manifest for restore: %s", e)
        return

    moved_map: Dict[str, str] = manifest.get("moved_map", {})
    # Move files back
    for moved_path, orig_path in moved_map.items():
        try:
            if not os.path.exists(moved_path):
                logging.warning("Moved file missing (skipping): %s", moved_path)
                continue
            orig_dir = os.path.dirname(orig_path)
            os.makedirs(orig_dir, exist_ok=True)
            if os.path.exists(orig_path):
                logging.warning("Original path already exists, skipping restore for %s", orig_path)
                continue
            shutil.move(moved_path, orig_path)
            logging.info("Restored %s -> %s", moved_path, orig_path)
        except Exception as e:
            logging.exception("Failed to restore %s -> %s: %s", moved_path, orig_path, e)

    # Optionally remove manifest (keep it for auditing)
    try:
        os.remove(manifest_path)
        logging.info("Removed manifest %s after restore.", manifest_path)
    except Exception:
        pass

def collect_global_dataset(
    id_dirs: List[str],
    held_out: Dict[str, Dict[str, List[str]]],
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Collect train and eval files with STRICT file-label alignment.
    Labels are appended exactly when files are appended.
    This guarantees len(eval_files) == len(eval_labels).
    """

    train_files: List[str] = []
    eval_files: List[str] = []
    eval_labels: List[int] = []

    for id_dir in id_dirs:
        id_name = os.path.basename(id_dir)

        normal_files = sorted(glob.glob(os.path.join(id_dir, "normal", "*.wav")))
        abnormal_files = sorted(glob.glob(os.path.join(id_dir, "abnormal", "*.wav")))

        held_normal = set(held_out.get(id_name, {}).get("normal", []))
        held_abnormal = set(held_out.get(id_name, {}).get("abnormal", []))

        # --- TRAIN: normal only, excluding held-out ---
        for f in normal_files:
            if f not in held_normal:
                train_files.append(f)

        # --- EVAL: normal (label 0), excluding held-out ---
        for f in normal_files:
            if f not in held_normal:
                eval_files.append(f)
                eval_labels.append(0)

        # --- EVAL: abnormal (label 1), excluding held-out ---
        for f in abnormal_files:
            if f not in held_abnormal:
                eval_files.append(f)
                eval_labels.append(1)

    # Hard safety check (never remove this)
    assert len(eval_files) == len(eval_labels), (
        f"Eval mismatch: files={len(eval_files)} labels={len(eval_labels)}"
    )

    return train_files, eval_files, np.array(eval_labels)


# ----------------------- Single-file prediction helper --------------------

def compute_threshold_from_eval(scores: np.ndarray, labels: np.ndarray, percentile: float = 95.0) -> float:
    normal_scores = scores[labels == 0]
    if normal_scores.size > 0:
        return float(np.percentile(normal_scores, percentile))
    return float(np.percentile(scores, percentile))

def predict_single_file_score(model: Model, filepath: str, feat_cfg: dict, scaler: Optional[StandardScaler], batch_size: int = 256) -> float:
    data = file_to_vector_array(filepath, **feat_cfg)
    if data.shape[0] == 0:
        logging.warning("No features for %s", filepath)
        return float("nan")
    try:
        data_n = normalize_windows_energy(data)
    except Exception:
        data_n = data
    if scaler is not None:
        try:
            data_n = scaler.transform(data_n)
        except Exception as e:
            logging.warning("Scaler transform failed for %s: %s. Using unscaled data.", filepath, e)
    ds = tf.data.Dataset.from_tensor_slices(data_n.astype(np.float32)).batch(batch_size)
    recon = model.predict(ds, verbose=0)
    errs = np.mean(np.square(data_n - recon), axis=1)
    score = float(np.percentile(errs, 95))
    return score

# --------------------------- Run signature / status ------------------------

def compute_run_signature(feat_cfg: dict, fit_cfg: dict, seed: int, manual_n_normal: int, manual_n_abnormal: int) -> str:
    sig_obj = {
        "feat": feat_cfg,
        "fit": fit_cfg,
        "seed": seed,
        "manual_n_normal": manual_n_normal,
        "manual_n_abnormal": manual_n_abnormal,
    }
    s = json.dumps(sig_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_status(status_path: str) -> Optional[dict]:
    if not os.path.exists(status_path):
        return None
    try:
        return read_json(status_path)
    except Exception as e:
        logging.warning("Failed to read status file %s: %s", status_path, e)
        return None

def write_status(status_path: str, status: dict) -> None:
    try:
        write_json(status_path, status)
        logging.info("Wrote status to %s", status_path)
    except Exception as e:
        logging.warning("Failed to write status %s: %s", status_path, e)

# --------------------------- CLI / Helpers --------------------------------

def list_manual_test_files(manual_test_dir: str) -> List[str]:
    files = []
    if not os.path.isdir(manual_test_dir):
        return files
    for root, _, fnames in os.walk(manual_test_dir):
        for f in fnames:
            if f == "manifest.pickle":
                continue
            files.append(os.path.join(root, f))
    return sorted(files)

def manual_cli_loop(
    manual_test_dir: str,
    model: Model,
    feat_cfg: dict,
    scaler: Optional[StandardScaler],
    threshold: float,
    run_cfg: dict,
    restore_on_exit: bool = True,
):
    """
    Manual CLI for classifying files.

    If restore_on_exit is True: on 'exit' or KeyboardInterrupt the manifest is used to move files back.
    If restore_on_exit is False: manual_test files remain in place after exiting the CLI.
    """
    print("\nManual testing CLI started.")
    print("Commands: list | show <id/class> | help | exit")
    print("You can also paste an absolute path (quotes allowed) to classify a file.")
    print("To classify all manual_test files in batch, type: batch\n")

    try:
        while True:
            try:
                cmd = input("Enter file path (or command): ").strip()
            except EOFError:
                cmd = "exit"

            # strip surrounding quotes so pasted paths work
            cmd = cmd.strip().strip('"').strip("'")
            if not cmd:
                continue

            if cmd.lower() in ("exit", "quit"):
                logging.info("Exit command received.")
                manifest_path = os.path.join(manual_test_dir, "manifest.pickle")
                if restore_on_exit and os.path.exists(manifest_path):
                    restore_manual_test_split(manual_test_dir)
                    print("Manual test files restored. Exiting.")
                else:
                    if restore_on_exit:
                        print("No manifest found. Exiting without restore.")
                    else:
                        print("Exiting without restoring manual_test files (infer mode).")
                break

            if cmd.lower() == "help":
                print(
                    "Commands:\n"
                    "  list                 - list all manual_test files\n"
                    "  show <id/class>      - show files under id or id/class\n"
                    "  batch                - run classification for all files under manual_test\n"
                    "  exit / quit          - restore files (if manifest exists and restore_on_exit=True) and exit\n"
                    "Or paste a file path to classify it."
                )
                continue

            if cmd.lower() == "list":
                files = list_manual_test_files(manual_test_dir)
                if not files:
                    print("No manual_test files found.")
                else:
                    for p in files:
                        print(p)
                continue

            if cmd.lower().startswith("show "):
                _, _, arg = cmd.partition(" ")
                target = os.path.join(manual_test_dir, arg)
                if os.path.isdir(target):
                    for root, _, files in os.walk(target):
                        for f in files:
                            if f != "manifest.pickle":
                                print(os.path.join(root, f))
                else:
                    print("Invalid id or class:", arg)
                continue

            if cmd.lower() == "batch":
                files = list_manual_test_files(manual_test_dir)
                if not files:
                    print("No manual_test files to classify.")
                    continue
                print(f"Running batch classification on {len(files)} files...")
                results = []
                for p in files:
                    try:
                        score = predict_single_file_score(model, p, feat_cfg, scaler, batch_size=run_cfg.get("batch_predict_size", 256))
                        verdict = "ABNORMAL" if score > threshold else "NORMAL"
                        results.append((p, score, verdict))
                        print(f"{verdict:7s} | {score:.6g} | {p}")
                    except Exception as e:
                        logging.exception("Batch inference error for %s: %s", p, e)
                        print("ERROR for", p, "->", e)
                # summary
                n_ab = sum(1 for _, _, v in results if v == "ABNORMAL")
                n_norm = len(results) - n_ab
                print(f"\nBatch complete. NORMAL: {n_norm} | ABNORMAL: {n_ab}\n")
                continue

            # Resolve file path (normalize Windows backslashes)
            query_path = cmd
            query_path = os.path.normpath(query_path)

            if not os.path.isabs(query_path):
                candidate = os.path.join(manual_test_dir, query_path)
                if os.path.exists(candidate):
                    query_path = candidate
                else:
                    matches = glob.glob(os.path.join(manual_test_dir, "**", query_path), recursive=True)
                    if matches:
                        query_path = matches[0]

            if not os.path.exists(query_path):
                print("File not found:", query_path)
                continue

            # Run inference
            try:
                score = predict_single_file_score(
                    model,
                    query_path,
                    feat_cfg,
                    scaler,
                    batch_size=run_cfg.get("batch_predict_size", 256),
                )

                if np.isnan(score):
                    print("Could not compute score for file:", query_path)
                    continue

                verdict = "ABNORMAL" if score > threshold else "NORMAL"

                print("\n================ RESULT ================")
                print(f"File      : {query_path}")
                print(f"Score     : {score:.6f}")
                print(f"Threshold : {threshold:.6f}")
                print(f"Verdict   : {verdict}")
                print("========================================\n")

            except Exception as e:
                logging.exception("Inference failed: %s", e)
                print("Inference error:", e)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        if restore_on_exit:
            print("Attempting to restore files and exit.")
            manifest_path = os.path.join(manual_test_dir, "manifest.pickle")
            if os.path.exists(manifest_path):
                restore_manual_test_split(manual_test_dir)
        else:
            print("Exiting without restoring manual_test files (infer mode).")
        sys.exit(0)

# --------------------------------- MAIN -----------------------------------

def main():
    print("\n>>> MAIN EXECUTION STARTED <<<\n", flush=True)
    parser = argparse.ArgumentParser(description="MIMII Dense AE (Generalized, Valve Only)")
    parser.add_argument("--base_directory", required=True, help="Path to valve directory (contains id_* folders)")
    parser.add_argument("--config", default="baseline.yaml", help="YAML config file")
    parser.add_argument("--loglevel", default="INFO", help="Logging level (INFO / DEBUG)")
    parser.add_argument("--force_recompute", action="store_true", help="Force recompute features and retrain model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (defaults to 42)")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="Mode: train (default) or infer (no training / no dataset moves)")
    parser.add_argument("--manual_n_normal", type=int, default=5, help="Number of normal files per id to hold out for manual_test")
    parser.add_argument("--manual_n_abnormal", type=int, default=5, help="Number of abnormal files per id to hold out for manual_test")
    parser.add_argument("--auto_batch", action="store_true", help="If infer mode and manual_test exists, run batch classification and exit")
    args = parser.parse_args()

    cfg = load_config(args.config if os.path.exists(args.config) else None)
    logpath = cfg.get("logging", {}).get("logfile", "baseline.log")
    level = getattr(logging, args.loglevel.upper(), logging.INFO)
    setup_logging(logfile=logpath, level=level)

    logging.info("=" * 70)
    logging.info("MIMII GENERALIZED DENSE AUTOENCODER - EXECUTION STARTED")
    logging.info("=" * 70)
    logging.info("Mode: %s | Base dir: %s | Seed: %s", args.mode, args.base_directory, args.seed)

    set_seed(args.seed)

    pickle_dir = cfg["paths"]["pickle_directory"]
    model_dir = cfg["paths"]["model_directory"]
    result_dir = cfg["paths"]["result_directory"]
    cache_mels_dir = os.path.join(pickle_dir, "mels")
    manual_test_dir = os.path.join(result_dir, "manual_test")
    status_path = os.path.join(result_dir, "status.json")

    os.makedirs(pickle_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_mels_dir, exist_ok=True)

    feat_cfg = cfg["feature"].copy()
    feat_cfg["cache_mels_dir"] = cache_mels_dir

    fit_cfg = cfg.get("fit", {}).copy()
    run_cfg = cfg.get("run", {}).copy()

    base_directory = os.path.abspath(args.base_directory)
    id_dirs = sorted(
        d for d in glob.glob(os.path.join(base_directory, "*"))
        if os.path.isdir(d) and os.path.basename(d).lower().startswith("id")
    )

    if not id_dirs:
        logging.error("No id_* directories found in %s", base_directory)
        sys.exit(1)

    logging.info("FOUND %d VALVE MACHINE IDS", len(id_dirs))
    for d in id_dirs:
        logging.info(" - %s", os.path.basename(d))

    # compute run signature
    signature = compute_run_signature(feat_cfg, fit_cfg, args.seed, args.manual_n_normal, args.manual_n_abnormal)
    logging.info("Run signature: %s", signature)

    # load previous status if any
    prev_status = load_status(status_path)
    prev_signature = prev_status.get("signature") if prev_status else None

    model_full = os.path.join(model_dir, "generalized_model.keras")
    model_weights = os.path.join(model_dir, "generalized_model.weights.h5")
    train_pickle = os.path.join(pickle_dir, "generalized_train_data.pickle")
    scaler_pickle = os.path.join(pickle_dir, "generalized_scaler.pickle")

    # MODE: infer -> do not create manual_test split, do not retrain, only load model and start CLI
    if args.mode == "infer":
        if not os.path.exists(model_full) and not os.path.exists(model_weights):
            logging.error("No trained model found. Cannot run in infer mode.")
            sys.exit(1)
        logging.info("Infer mode: loading model and scaler, skipping dataset moves and training.")
        try:
            model = tf.keras.models.load_model(model_full)
        except Exception:
            # fallback: build model architecture and load weights if only weights exist
            if os.path.exists(model_weights):
                # need input_dim - attempt to load train pickle
                if os.path.exists(train_pickle):
                    train_data = load_pickle(train_pickle)
                    input_dim = train_data.shape[1]
                    model = build_autoencoder(input_dim)
                    model.load_weights(model_weights)
                else:
                    logging.error("Weights present but cannot infer input_dim (missing train pickle). Aborting.")
                    sys.exit(1)
            else:
                logging.exception("Failed to load model in infer mode.")
                sys.exit(1)
        scaler = load_scaler(scaler_pickle)

        # Load threshold from status if available; fallback to generalized_results.yaml
        threshold = float("nan")
        if prev_status and "threshold" in prev_status:
            threshold = prev_status.get("threshold", float("nan"))
        else:
            # try generalized_results.yaml
            results_file = os.path.join(result_dir, "generalized_results.yaml")
            if os.path.exists(results_file):
                try:
                    with open(results_file, "r") as f:
                        r = yaml.safe_load(f)
                        threshold = r.get("threshold", float("nan"))
                except Exception:
                    logging.warning("Failed to read generalized_results.yaml for threshold.")

        logging.info("Entering manual CLI (infer mode). Manual_test folder: %s", manual_test_dir)

        # If auto_batch requested, run batch then exit
        if args.auto_batch:
            files = list_manual_test_files(manual_test_dir)
            if not files:
                logging.info("No manual_test files found for batch inference.")
                print("No manual_test files found for batch inference.")
            else:
                print(f"Running batch classification on {len(files)} files...")
                n_ab = 0
                for p in files:
                    score = predict_single_file_score(model, p, feat_cfg, scaler, batch_size=run_cfg.get("batch_predict_size", 256))
                    verdict = "ABNORMAL" if score > threshold else "NORMAL"
                    if verdict == "ABNORMAL":
                        n_ab += 1
                    print(f"{verdict:7s} | {score:.6g} | {p}")
                print(f"\nBatch complete. NORMAL: {len(files)-n_ab} | ABNORMAL: {n_ab}\n")
            return

        # go directly to CLI (do NOT restore manual_test on exit in infer mode)
        manual_cli_loop(manual_test_dir, model, feat_cfg, scaler, threshold, run_cfg, restore_on_exit=False)
        return

    # MODE: train
    # Create manual test split only if manifest does not exist.
    manifest_path = os.path.join(manual_test_dir, "manifest.pickle")
    if os.path.exists(manifest_path):
        logging.info("Detected existing manual_test manifest at %s", manifest_path)
        if prev_signature != signature and not args.force_recompute:
            logging.error("Existing manual_test manifest was created under a different run signature (%s).", prev_signature)
            logging.error("To regenerate manual_test with current settings, re-run with --force_recompute")
            sys.exit(1)
        else:
            try:
                manifest = load_pickle(manifest_path)
                held_out = manifest.get("held_out", {})
                logging.info("Using existing manual_test manifest; held_out loaded.")
            except Exception as e:
                logging.exception("Failed to load existing manifest: %s", e)
                if not args.force_recompute:
                    logging.error("Use --force_recompute to regenerate manual_test split.")
                    sys.exit(1)
                else:
                    logging.info("Proceeding to recreate manual_test split due to force_recompute.")
                    held_out = create_manual_test_split(id_dirs=id_dirs, output_dir=manual_test_dir,
                                                        n_normal_per_id=args.manual_n_normal,
                                                        n_abnormal_per_id=args.manual_n_abnormal,
                                                        seed=args.seed)
    else:
        # create manual_test split (move files)
        try:
            held_out = create_manual_test_split(
                id_dirs=id_dirs,
                output_dir=manual_test_dir,
                n_normal_per_id=args.manual_n_normal,
                n_abnormal_per_id=args.manual_n_abnormal,
                seed=args.seed or 42,
            )
        except Exception as e:
            logging.exception("Failed to create manual test split: %s", e)
            sys.exit(1)

    # Collect train / eval files (global, excluding moved)
    logging.info("Collecting global dataset (excluding held-out files)...")
    train_files, eval_files, eval_labels = collect_global_dataset(id_dirs, held_out)

    logging.info("TOTAL TRAIN FILES (normal, aggregated): %d", len(train_files))
    logging.info("TOTAL EVAL FILES (normal+abnormal): %d", len(eval_files))

    # Decide whether to skip training based on previous status + artifacts
    skip_training = False
    if prev_status and prev_signature == signature and prev_status.get("training_done") and not args.force_recompute:
        # signature matches, training was done previously
        if os.path.exists(model_full):
            logging.info("Found previous run with same signature and trained model. Skipping training.")
            skip_training = True
        else:
            logging.warning("Status indicated training_done but model file missing. Will retrain.")
            skip_training = False

    # Load or build train_data
    if os.path.exists(train_pickle) and not args.force_recompute and not skip_training:
        logging.info("Loading cached generalized training data from %s", train_pickle)
        train_data = load_pickle(train_pickle)
        scaler = load_scaler(scaler_pickle)
    elif skip_training:
        # training skipped; load pickle + scaler mandatory
        if os.path.exists(train_pickle):
            train_data = load_pickle(train_pickle)
            scaler = load_scaler(scaler_pickle)
        else:
            logging.warning("Skipping training but training pickle missing. Will attempt to load model and scaler directly.")
            train_data = np.empty((0, feat_cfg.get("n_mels",64) * feat_cfg.get("frames",5)))
            scaler = load_scaler(scaler_pickle)
    else:
        logging.info("BUILDING GENERALIZED TRAIN DATASET (this may take a while)...")
        train_data = build_dataset_from_files(train_files, feat_cfg)
        logging.info("Raw train data shape: %s", train_data.shape)

        if train_data.size == 0:
            logging.error("No training windows extracted. Check your dataset and feature params.")
            logging.info("Attempting to restore manual_test files before exit...")
            try:
                restore_manual_test_split(manual_test_dir)
            except Exception:
                logging.exception("Restore during failure also failed.")
            sys.exit(1)

        train_data = normalize_windows_energy(train_data)
        scaler = fit_and_save_scaler(train_data, scaler_pickle)
        train_data = scaler.transform(train_data)

        save_pickle(train_pickle, train_data)
        logging.info("Saved generalized train data and scaler to %s and %s", train_pickle, scaler_pickle)

    # Build model (need input_dim)
    if train_data.size > 0:
        input_dim = train_data.shape[1]
    else:
        # fallback: if model exists, load to infer input_dim; else cannot continue
        if os.path.exists(model_full):
            try:
                model = tf.keras.models.load_model(model_full)
                input_dim = model.input_shape[1]
            except Exception:
                logging.error("Cannot determine input_dim. Aborting.")
                restore_manual_test_split(manual_test_dir)
                sys.exit(1)
        else:
            logging.error("No train data and no model available. Aborting.")
            restore_manual_test_split(manual_test_dir)
            sys.exit(1)

    model = build_autoencoder(input_dim)
    model.summary()

    # TRAIN
    try:
        if not skip_training:
            if os.path.exists(model_full) and not args.force_recompute:
                logging.info("Loading existing generalized model from %s", model_full)
                model = tf.keras.models.load_model(model_full)
            else:
                logging.info("TRAINING GENERALIZED AUTOENCODER (verbose output visible)")
                train_autoencoder(
                    model,
                    train_data,
                    model_weights,
                    epochs=fit_cfg.get("epochs", 120),
                    batch_size=fit_cfg.get("batch_size", 128),
                    validation_split=fit_cfg.get("validation_split", 0.1),
                    patience=fit_cfg.get("patience", 20),
                    verbose=1,
                    use_tfdata=False,
                )
                try:
                    model.save(model_full)
                    logging.info("Model saved to %s (and weights backup to %s)", model_full, model_weights)
                except Exception:
                    # Try saving weights
                    try:
                        model.save_weights(model_weights)
                        logging.info("Model weights saved to %s", model_weights)
                    except Exception:
                        logging.exception("Failed to save model or weights.")
        else:
            # skip_training True -> load model
            if os.path.exists(model_full):
                model = tf.keras.models.load_model(model_full)
            elif os.path.exists(model_weights):
                model.load_weights(model_weights)
            else:
                logging.error("skip_training set but no model artifacts present. Aborting.")
                restore_manual_test_split(manual_test_dir)
                sys.exit(1)

    except KeyboardInterrupt:
        logging.warning("Training interrupted by user (KeyboardInterrupt). Attempting to save partial & restore files.")
        if 'model' in locals():
            try:
                model.save_weights(model_weights.replace(".weights.h5", ".partial.weights.h5"))
                logging.info("Partial weights saved.")
            except Exception:
                logging.warning("Failed to save partial weights.")
        restore_manual_test_split(manual_test_dir)
        sys.exit(1)
    except Exception as e:
        logging.exception("Error during training: %s", e)
        restore_manual_test_split(manual_test_dir)
        sys.exit(1)

    # EVALUATION
    logging.info("=" * 60)
    logging.info("EVALUATING GENERALIZED MODEL")
    logging.info("=" * 60)

    scores = compute_anomaly_scores_for_files(
        model=model,
        file_list=eval_files,
        feat_kwargs=feat_cfg,
        batch_size=run_cfg.get("batch_predict_size", 256),
        scaler=scaler,
        scoring="percentile",
        percentile=95,
        normalize_windows=True,
    )

    results = evaluate_scores(np.array(eval_labels), scores)
    logging.info("FINAL RESULTS -> AUC: %.4f | AUPR: %.4f", results["AUC"], results["AUPR"])

    # compute threshold using normal eval files (95th percentile)
    try:
        threshold = compute_threshold_from_eval(scores, np.array(eval_labels), percentile=95.0)
        logging.info("Using anomaly threshold (95th percentile of normal eval scores): %.6g", threshold)
    except Exception as e:
        threshold = float("nan")
        logging.warning("Failed to compute threshold from eval: %s", e)

    # SAVE RESULTS
    result_file = os.path.join(result_dir, "generalized_results.yaml")
    with open(result_file, "w") as f:
        yaml.safe_dump({"results": results, "threshold": threshold}, f)

    # update status
    status = {
        "signature": signature,
        "seed": args.seed,
        "mode": args.mode,
        "training_done": True,
        "results": results,
        "threshold": threshold,
        "timestamp": int(np.floor(np.datetime64('now').astype('int') / 1)),
        "feat_cfg": feat_cfg,
        "fit_cfg": fit_cfg,
        "manual_n_normal": args.manual_n_normal,
        "manual_n_abnormal": args.manual_n_abnormal,
    }
    write_status(status_path, status)

    logging.info("=" * 70)
    logging.info("EXECUTION COMPLETED SUCCESSFULLY")
    logging.info("=" * 70)

    # ------------------ Interactive manual testing loop ------------------
    logging.info("Entering manual CLI. Manual_test folder: %s", manual_test_dir)
    # After training we want to restore manual_test files when user exits
    manual_cli_loop(manual_test_dir, model, feat_cfg, scaler, threshold, run_cfg, restore_on_exit=True)


if __name__ == "__main__":
    main()
