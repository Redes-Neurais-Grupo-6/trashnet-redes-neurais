"""Freeze validation decisions, then evaluate the six frozen checkpoints once.

Run from the repository root::
    python -m src.evaluate --lock
    python -m src.evaluate --evaluate
    python -m src.evaluate --verify-inference

The lock must predate testing. It records the validation predictions used for
selection and SHA-256 hashes of every checkpoint and experimental record. A
second invocation reuses a verified evaluation; it never replaces test results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
EXPERIMENTS = [
    "mlp_128_unweighted", "mlp_128_weighted", "cnn_128_unweighted",
    "cnn_128_weighted", "cnn_256_unweighted", "cnn_512_unweighted",
]
ROOT = Path(__file__).resolve().parents[1]
LOCK_NAME = "selection_lock.json"
_VERIFIED_CACHES = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_once(path: Path, content: bytes):
    """Atomically publish new bytes; an existing different file is an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"Refusing to replace an existing result: {path}")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Cannot publish an empty result table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def load_manifest(root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(root / "data/manifest.csv")
    required = {"path", "class_name", "label", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {required - set(manifest.columns)}")
    if manifest.path.duplicated().any():
        raise ValueError("Duplicate source paths in manifest")
    if not manifest["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Manifest split must be train, val or test")
    if not manifest["label"].isin(range(6)).all():
        raise ValueError("Manifest labels must be integers in [0,5]")
    if not (manifest.label.map(dict(enumerate(CLASSES))) == manifest.class_name).all():
        raise ValueError("Manifest label/class order differs from the fixed class order")
    for column in ("sha256", "pixel_sha256"):
        if column in manifest and (manifest.groupby(column)["split"].nunique() > 1).any():
            raise ValueError(f"Identical images cross data splits ({column})")
    for split in ("train", "val", "test"):
        if set(manifest.loc[manifest.split == split, "label"]) != set(range(6)):
            raise ValueError(f"All six classes must occur in {split}")
    return manifest


def normalize_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept explicit names shared with the training script."""
    aliases = {"true_label": "y_true", "pred_label": "y_pred", "predicted_label": "y_pred", "label": "y_true",
               "prediction": "y_pred", "true": "y_true", "pred": "y_pred"}
    return frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns and v not in frame.columns})


def validate_validation_predictions(root: Path, experiment: str, manifest: pd.DataFrame):
    frame = normalize_prediction_columns(pd.read_csv(root / "runs" / experiment / "val_predictions.csv"))
    expected = manifest.loc[manifest.split == "val"]
    if not {"path", "y_true", "y_pred"}.issubset(frame.columns):
        raise ValueError(f"{experiment}: validation predictions need path, y_true and y_pred")
    if frame.path.duplicated().any() or set(frame.path) != set(expected.path):
        raise ValueError(f"{experiment}: validation predictions do not exactly cover the validation set")
    frame = frame.set_index("path").loc[expected.path].reset_index()
    if not np.array_equal(frame.y_true.to_numpy(), expected.label.to_numpy()):
        raise ValueError(f"{experiment}: validation ground truth differs from manifest")
    if not frame.y_pred.isin(range(6)).all():
        raise ValueError(f"{experiment}: invalid predicted class")
    return {
        "accuracy": float(accuracy_score(frame.y_true, frame.y_pred)),
        "f1_macro": float(f1_score(frame.y_true, frame.y_pred, labels=range(6), average="macro", zero_division=0)),
        "support": len(frame),
    }


def validate_run_provenance(root: Path, validation_metrics: dict):
    """Reject stale/mixed checkpoints before freezing the experimental record."""
    config_document = json.loads((root / "configs/experiments.json").read_text())
    if config_document.get("class_order") != CLASSES:
        raise ValueError("Final configuration has an inconsistent class order")
    configs = {item["id"]: {**config_document["defaults"], **item} for item in config_document["experiments"]}
    if set(configs) != set(EXPERIMENTS) or len(config_document["experiments"]) != len(EXPERIMENTS):
        raise ValueError("Final configuration must contain exactly the six controlled experiments")
    manifest_hash = sha256_file(root / "data/manifest.csv")
    for experiment in EXPERIMENTS:
        run = root / "runs" / experiment
        metadata = json.loads((run / "metadata.json").read_text())
        config = json.loads((run / "config.json").read_text())
        if config != configs[experiment] or metadata.get("config") != config or metadata.get("id") != experiment:
            raise ValueError(f"{experiment}: saved training configuration differs from the final experiment configuration")
        expected_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if metadata.get("config_sha256") != expected_config_hash:
            raise ValueError(f"{experiment}: training configuration SHA-256 is inconsistent")
        if metadata.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"{experiment}: checkpoint was trained with a different manifest")
        if metadata.get("model_file_sha256") != sha256_file(run / "model.pt"):
            raise ValueError(f"{experiment}: checkpoint differs from the model saved by training")
        if metadata.get("class_order") != CLASSES:
            raise ValueError(f"{experiment}: checkpoint class order is inconsistent")
        for metric in ("accuracy", "f1_macro"):
            recorded = metadata.get("restored_val_metrics", {}).get(metric)
            if recorded is None or not np.isclose(recorded, validation_metrics[experiment][metric], rtol=0, atol=1e-12):
                raise ValueError(f"{experiment}: restored checkpoint validation {metric} disagrees with predictions")


def tracked_files(root: Path):
    paths = [root / "data/manifest.csv"]
    paths.extend(root / name for name in ("src/models.py", "src/train.py", "src/prepare.py"))
    paths.extend(root / "data/cache" / resolution / "metadata.json" for resolution in ("128x96", "256x192", "512x384"))
    config_paths = sorted(p for p in (root / "configs").rglob("*") if p.is_file())
    if not config_paths:
        raise FileNotFoundError("No experiment configuration exists in configs/")
    paths.extend(config_paths)
    for experiment in EXPERIMENTS:
        for filename in ("model.pt", "config.json", "metadata.json", "history.csv", "val_predictions.csv"):
            paths.append(root / "runs" / experiment / filename)
    return {str(p.relative_to(root)): sha256_file(p) for p in paths}


def verify_lock(root: Path):
    path = root / LOCK_NAME
    if not path.exists():
        raise FileNotFoundError("Create the validation selection lock first: python -m src.evaluate --lock")
    lock = json.loads(path.read_text())
    if lock.get("experiment_order") != EXPERIMENTS or lock.get("class_order") != CLASSES:
        raise ValueError("Lock experiment/class order is inconsistent")
    actual = tracked_files(root)
    if actual != lock.get("file_sha256"):
        changed = sorted(set(actual) ^ set(lock.get("file_sha256", {})) |
                         {p for p in actual if actual[p] != lock.get("file_sha256", {}).get(p)})
        raise ValueError("Frozen experiment files changed; testing is blocked: " + ", ".join(changed))
    manifest = load_manifest(root)
    metrics = {experiment: validate_validation_predictions(root, experiment, manifest) for experiment in EXPERIMENTS}
    selected = max(EXPERIMENTS, key=lambda e: metrics[e]["f1_macro"])
    selected_cnn = max((e for e in EXPERIMENTS if e.startswith("cnn_")), key=lambda e: metrics[e]["f1_macro"])
    if lock.get("validation_metrics") != metrics or lock.get("selected_experiment") != selected or lock.get("selected_cnn_experiment") != selected_cnn:
        raise ValueError("Frozen model-selection decision disagrees with the saved validation predictions")
    validate_run_provenance(root, metrics)
    return lock


def create_lock(root: Path):
    if (root / LOCK_NAME).exists():
        lock = verify_lock(root)
        print(f"Existing lock verified; selected model: {lock['selected_experiment']}")
        return lock
    if any((root / "runs" / experiment / name).exists() for experiment in EXPERIMENTS for name in ("predictions.csv", "metrics.json")):
        raise ValueError("Test artifacts already exist without a selection lock; freeze decisions before evaluating test")
    manifest = load_manifest(root)
    metrics = {experiment: validate_validation_predictions(root, experiment, manifest) for experiment in EXPERIMENTS}
    validate_run_provenance(root, metrics)
    # Stable order breaks exact ties without consulting any test outcome.
    selected = max(EXPERIMENTS, key=lambda e: metrics[e]["f1_macro"])
    selected_cnn = max((e for e in EXPERIMENTS if e.startswith("cnn_")), key=lambda e: metrics[e]["f1_macro"])
    lock = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "selection_criterion": "maximum validation macro F1 of the saved checkpoints; ties use experiment_order",
        "class_order": CLASSES,
        "experiment_order": EXPERIMENTS,
        "selected_experiment": selected,
        "selected_cnn_experiment": selected_cnn,
        "validation_metrics": metrics,
        "file_sha256": tracked_files(root),
    }
    write_once(root / LOCK_NAME, json_bytes(lock))
    print(f"Validation selection frozen: {selected}; best CNN: {selected_cnn}")
    return lock


def stratified_bootstrap(y_true, y_pred, repetitions=2000, seed=20261127):
    """Percentile CIs conditional on this test split, not training-seed CIs."""
    if repetitions <= 0:
        return None
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == k) for k in range(6)]
    accuracy, macro_f1 = np.empty(repetitions), np.empty(repetitions)
    for i in range(repetitions):
        sampled = np.concatenate([rng.choice(idx, len(idx), replace=True) for idx in class_indices])
        cm = np.bincount(6 * y_true[sampled] + y_pred[sampled], minlength=36).reshape(6, 6)
        tp = np.diag(cm)
        denominator = cm.sum(axis=0) + cm.sum(axis=1)
        scores = np.divide(2 * tp, denominator, out=np.zeros(6, dtype=float), where=denominator != 0)
        accuracy[i] = tp.sum() / cm.sum()
        macro_f1[i] = scores.mean()
    return {
        "method": "stratified nonparametric percentile bootstrap within true class",
        "repetitions": repetitions, "seed": seed, "confidence_level": 0.95,
        "accuracy": np.quantile(accuracy, [0.025, 0.975]).tolist(),
        "f1_macro": np.quantile(macro_f1, [0.025, 0.975]).tolist(),
        "interpretation": "Test-sample uncertainty conditional on the frozen trained model and class supports; excludes training-seed variation and distribution shift.",
    }


def predict_test(root: Path, experiment: str, manifest: pd.DataFrame, device_name: str):
    """Load the saved architecture and run deterministic, unaugmented test inference."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .models import build_model

    run = root / "runs" / experiment
    metadata = json.loads((run / "metadata.json").read_text())
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    config = metadata["config"]
    torch.set_num_threads(int(config.get("threads", 4)))
    torch.use_deterministic_algorithms(True)
    width, height = int(config["width"]), int(config["height"])
    model = build_model(config)
    model.load_state_dict(checkpoint)
    device = torch.device(device_name)
    model.to(device).eval()
    cache_path = root / "data/cache" / f"{width}x{height}" / "images.npy"
    cache_metadata = json.loads(cache_path.with_name("metadata.json").read_text())
    if cache_metadata.get("manifest_sha256") != sha256_file(root / "data/manifest.csv"):
        raise ValueError(f"{experiment}: cache is associated with a different manifest")
    if cache_metadata.get("layout") != "NCHW":
        raise ValueError(f"{experiment}: cache layout must be NCHW")
    stat = cache_path.stat()
    cache_key = (str(cache_path.resolve()), stat.st_size, stat.st_mtime_ns)
    if cache_key not in _VERIFIED_CACHES:
        actual_hash = sha256_file(cache_path)
        if actual_hash != cache_metadata.get("images_sha256"):
            raise ValueError(f"{experiment}: cache contents failed SHA-256 integrity check")
        _VERIFIED_CACHES[cache_key] = actual_hash
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if cache.shape != (len(manifest), 3, height, width) or cache.dtype != np.uint8:
        raise ValueError(f"{experiment}: incompatible cache shape or dtype: {cache.shape} / {cache.dtype}")
    indices = np.flatnonzero(manifest.split.to_numpy() == "test")
    # Advanced indexing makes a writable copy, preserving the read-only cache.
    images = torch.from_numpy(np.asarray(cache[indices]))
    loader = DataLoader(TensorDataset(images), batch_size=int(config.get("batch_size", 32)), shuffle=False)
    probabilities = []
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(device=device, dtype=torch.float32) / 255.0)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError(f"{experiment}: invalid class probabilities")
    return indices, probabilities, metadata


def make_metrics(y_true, y_pred, bootstrap_repetitions, bootstrap_seed):
    cm = confusion_matrix(y_true, y_pred, labels=range(6))
    row_pct = np.divide(100.0 * cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    report = classification_report(y_true, y_pred, labels=range(6), target_names=CLASSES, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=range(6), average="macro", zero_division=0)),
        "support": int(len(y_true)), "class_order": CLASSES,
        "classification_report": report, "zero_division": 0,
        "zero_division_explanation": "Undefined precision, recall or F1 is assigned 0; all six labels remain in the macro average.",
        "confusion_count": cm.tolist(), "confusion_row_percent": row_pct.tolist(),
        "bootstrap_95ci": stratified_bootstrap(y_true, y_pred, bootstrap_repetitions, bootstrap_seed),
    }


def cost_fields(metadata: dict, root: Path, experiment: str):
    """Extract measured cost without estimating missing quantities."""
    required = ["parameters", "trainable_parameters", "epochs_completed", "best_epoch",
                "total_training_seconds", "mean_epoch_seconds", "peak_rss_bytes"]
    missing = [key for key in required if metadata.get(key) is None]
    if missing:
        raise ValueError(f"{experiment}: required measured cost fields missing: {missing}")
    values = {key: metadata[key] for key in required if key not in ("total_training_seconds", "peak_rss_bytes")}
    values["train_seconds"] = metadata["total_training_seconds"]
    values["batch_size"] = metadata["config"]["batch_size"]
    values["peak_rss_mb"] = metadata["peak_rss_bytes"] / 1024**2
    values["peak_gpu_mb"] = None  # The controlled experiments were executed on CPU.
    values["checkpoint_bytes"] = (root / "runs" / experiment / "model.pt").stat().st_size
    return values


def aggregate_reports(root: Path, lock: dict, all_metrics: dict):
    rows, per_class = [], []
    for experiment in EXPERIMENTS:
        metrics = all_metrics[experiment]
        metadata = json.loads((root / "runs" / experiment / "metadata.json").read_text())
        row = {"experiment": experiment, "test_accuracy": metrics["accuracy"], "test_f1_macro": metrics["f1_macro"],
               "val_accuracy": lock["validation_metrics"][experiment]["accuracy"],
               "val_f1_macro": lock["validation_metrics"][experiment]["f1_macro"],
               "selected_by_validation": experiment == lock["selected_experiment"],
               "test_support": metrics["support"], **cost_fields(metadata, root, experiment)}
        ci = metrics.get("bootstrap_95ci")
        for metric in ("accuracy", "f1_macro"):
            row[f"test_{metric}_ci_low"] = ci[metric][0] if ci else None
            row[f"test_{metric}_ci_high"] = ci[metric][1] if ci else None
        rows.append(row)
        for class_name in CLASSES:
            values = metrics["classification_report"][class_name]
            per_class.append({"experiment": experiment, "class_name": class_name, "precision": values["precision"],
                              "recall": values["recall"], "f1": values["f1-score"], "support": int(values["support"])})
    write_once(root / "reports/summary.csv", csv_bytes(rows))
    write_once(root / "reports/per_class.csv", csv_bytes(per_class))
    write_once(root / "reports/summary.json", json_bytes({
        "selection_lock_sha256": sha256_file(root / LOCK_NAME), "selected_experiment": lock["selected_experiment"],
        "selected_cnn_experiment": lock["selected_cnn_experiment"], "class_order": CLASSES,
        "experiments": rows, "per_class": per_class,
    }))


def evaluate(root: Path, device: str, bootstrap_repetitions: int, seed: int):
    import torch
    lock = verify_lock(root)
    lock_hash = sha256_file(root / LOCK_NAME)
    manifest = load_manifest(root)
    all_metrics = {}
    for experiment in EXPERIMENTS:
        run = root / "runs" / experiment
        metrics_path, predictions_path = run / "metrics.json", run / "predictions.csv"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("selection_lock_sha256") != lock_hash or metrics.get("predictions_sha256") != sha256_file(predictions_path):
                raise ValueError(f"{experiment}: existing test artifacts failed integrity checks")
            ci = metrics.get("bootstrap_95ci")
            if bool(ci) != bool(bootstrap_repetitions) or (ci and (ci["repetitions"] != bootstrap_repetitions or ci["seed"] != seed)):
                raise ValueError("Existing evaluation uses a different bootstrap configuration; it will not be overwritten")
            all_metrics[experiment] = metrics
            print(f"{experiment}: reusing verified test results")
            continue
        if predictions_path.exists():
            raise FileExistsError(f"Incomplete evaluation exists: {predictions_path}. Inspect it before manually moving it out of the run directory.")
        indices, probabilities, _ = predict_test(root, experiment, manifest, device)
        subset = manifest.iloc[indices]
        y_true = subset.label.to_numpy(dtype=np.int64)
        y_pred = probabilities.argmax(1).astype(np.int64)
        rows = []
        for i, (global_index, source) in enumerate(subset.iterrows()):
            rows.append({"manifest_index": int(global_index), "path": source.path, "y_true": int(y_true[i]),
                         "y_pred": int(y_pred[i]), "true_class": CLASSES[y_true[i]], "pred_class": CLASSES[y_pred[i]],
                         "pred_probability": float(probabilities[i, y_pred[i]]),
                         **{f"prob_{name}": float(probabilities[i, j]) for j, name in enumerate(CLASSES)}})
        predictions_bytes = csv_bytes(rows)
        metrics = make_metrics(y_true, y_pred, bootstrap_repetitions, seed)
        metrics.update({"experiment": experiment, "evaluated_at_utc": utc_now(), "selection_lock_sha256": lock_hash,
                        "checkpoint_sha256": sha256_file(run / "model.pt"), "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
                        "inference_device": device, "python_version": platform.python_version(), "torch_version": str(torch.__version__)})
        write_once(predictions_path, predictions_bytes)
        write_once(metrics_path, json_bytes(metrics))
        all_metrics[experiment] = metrics
        print(f"{experiment}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['f1_macro']:.4f}")
    aggregate_reports(root, lock, all_metrics)


def verify_inference(root: Path, device: str):
    """Recompute predictions from frozen weights and compare, without writing."""
    verify_lock(root)
    manifest = load_manifest(root)
    lock_hash = sha256_file(root / LOCK_NAME)
    for experiment in EXPERIMENTS:
        for filename in ("metrics.json", "predictions.csv"):
            if not (root / "runs" / experiment / filename).is_file():
                raise FileNotFoundError("All six saved test evaluations must exist before verification")
    for experiment in EXPERIMENTS:
        run = root / "runs" / experiment
        saved = normalize_prediction_columns(pd.read_csv(run / "predictions.csv"))
        metrics = json.loads((run / "metrics.json").read_text())
        if metrics.get("selection_lock_sha256") != lock_hash or metrics.get("predictions_sha256") != sha256_file(run / "predictions.csv"):
            raise ValueError(f"{experiment}: stored evaluation integrity check failed")
        indices, probabilities, _ = predict_test(root, experiment, manifest, device)
        expected = manifest.iloc[indices]
        if saved.path.duplicated().any() or set(saved.path) != set(expected.path):
            raise ValueError(f"{experiment}: stored predictions do not exactly cover the test set")
        saved = saved.set_index("path").loc[expected.path].reset_index()
        y_true = expected.label.to_numpy(dtype=np.int64)
        y_pred = probabilities.argmax(1).astype(np.int64)
        if not np.array_equal(saved.y_true.to_numpy(), y_true):
            raise ValueError(f"{experiment}: ground truth differs from the frozen test manifest")
        if not np.array_equal(saved.y_pred.to_numpy(), y_pred):
            raise ValueError(f"{experiment}: recomputed predicted labels differ")
        old_probabilities = saved[[f"prob_{name}" for name in CLASSES]].to_numpy()
        if not np.allclose(old_probabilities, probabilities, rtol=1e-5, atol=1e-6):
            raise ValueError(f"{experiment}: recomputed probabilities differ beyond rtol=1e-5, atol=1e-6")
        fresh = make_metrics(y_true, y_pred, 0, 0)
        for name in ("accuracy", "f1_macro", "classification_report", "confusion_count", "confusion_row_percent"):
            if fresh[name] != metrics[name]:
                raise ValueError(f"{experiment}: saved metric differs from recomputed metric: {name}")
        delta = float(np.max(np.abs(old_probabilities - probabilities)))
        print(f"{experiment}: labels, all class probabilities and metrics verified; max probability difference={delta:.3g}")
    print("Verification complete. No training, model selection or artifact modification was performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lock", action="store_true", help="Freeze the six saved models and select by validation macro F1")
    mode.add_argument("--evaluate", action="store_true", help="Evaluate the frozen models on test, or reuse verified existing results")
    mode.add_argument("--verify-inference", action="store_true", help="Reload all six frozen models and compare fresh inference with saved test results, without overwriting")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Stratified bootstrap replicates; 0 disables CIs")
    parser.add_argument("--bootstrap-seed", type=int, default=20261127)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    if args.lock:
        create_lock(args.root.resolve())
    elif args.verify_inference:
        verify_inference(args.root.resolve(), args.device)
    else:
        evaluate(args.root.resolve(), args.device, args.bootstrap, args.bootstrap_seed)


if __name__ == "__main__":
    main()
