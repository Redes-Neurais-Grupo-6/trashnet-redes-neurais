"""Protocol regression tests using isolated synthetic records, never TrashNet test.

Run: python -m unittest discover -s tests -p 'test_evaluation.py' -v
These tests do not load real checkpoints, train networks or read dataset images.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src import evaluate as ev


def fixture(root: Path):
    """Six fake checkpoints with intentionally competing accuracy/macro F1."""
    rows = []
    for split in ("train", "val", "test"):
        for label, name in enumerate(ev.CLASSES):
            for sample in range(8 if split == "val" and label == 0 else 1):
                rows.append({"path": f"{name}/{split}{sample}.jpg", "class_name": name,
                             "label": label, "split": split,
                             "sha256": f"unique-file-{split}-{label}-{sample}",
                             "pixel_sha256": f"unique-pixels-{split}-{label}-{sample}"})
    manifest = pd.DataFrame(rows)
    (root / "data").mkdir(parents=True)
    manifest.to_csv(root / "data/manifest.csv", index=False)
    configs = {"class_order": ev.CLASSES, "defaults": {"seed": 42, "batch_size": 32},
               "experiments": [{"id": e, "architecture": e.split("_")[0]} for e in ev.EXPERIMENTS]}
    (root / "configs").mkdir()
    (root / "configs/experiments.json").write_text(json.dumps(configs))
    (root / "src").mkdir()
    for name in ("models.py", "train.py", "prepare.py"):
        (root / "src" / name).write_text("# Synthetic source-provenance fixture\n")
    for resolution in ("128x96", "256x192", "512x384"):
        path = root / "data/cache" / resolution / "metadata.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"manifest_sha256": ev.sha256_file(root / "data/manifest.csv"), "layout": "NCHW"}))
    validation = manifest[manifest.split == "val"]
    for index, experiment in enumerate(ev.EXPERIMENTS):
        run = root / "runs" / experiment
        run.mkdir(parents=True)
        # models 1 and 2 have LOWER accuracy but HIGHER macro F1 than all-zero classifiers.
        truth = validation.label.to_numpy()
        predictions = np.zeros(len(validation), dtype=int)
        if index in (1, 2):
            predictions = truth.copy()
            predictions[:6] = 1
        frame = pd.DataFrame({"path": validation.path, "true_label": truth,
                              "predicted_label": predictions})
        frame.to_csv(run / "val_predictions.csv", index=False)
        config = {**configs["defaults"], **configs["experiments"][index]}
        (run / "config.json").write_text(json.dumps(config))
        (run / "model.pt").write_bytes(f"synthetic-checkpoint-{experiment}".encode())
        (run / "history.csv").write_text("epoch,train_loss,val_loss\n1,2,1\n")
        metadata = {"id": experiment, "class_order": ev.CLASSES, "config": config,
                    "manifest_sha256": ev.sha256_file(root / "data/manifest.csv"),
                    "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    "model_file_sha256": ev.sha256_file(run / "model.pt"),
                    "restored_val_metrics": {"accuracy": accuracy_score(truth, predictions),
                                             "f1_macro": f1_score(truth, predictions, labels=range(6), average="macro", zero_division=0)}}
        (run / "metadata.json").write_text(json.dumps(metadata))
    return manifest


class MetricTests(unittest.TestCase):
    def test_exact_counts_row_percent_and_macro_average(self):
        truth = np.repeat(np.arange(6), 2)
        pred = np.array([0, 1, 1, 1, 2, 2, 3, 4, 4, 0, 5, 0])
        result = ev.make_metrics(truth, pred, 0, 42)
        expected_cm = np.array([[1, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0],
                                [0, 0, 2, 0, 0, 0], [0, 0, 0, 1, 1, 0],
                                [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(result["confusion_count"], expected_cm)
        np.testing.assert_array_equal(result["confusion_row_percent"], expected_cm * 50)
        self.assertAlmostEqual(result["accuracy"], 8 / 12)
        self.assertAlmostEqual(result["f1_macro"], (2/5 + 4/5 + 1 + 2/3 + 1/2 + 2/3) / 6)
        self.assertEqual(result["class_order"], ev.CLASSES)
        self.assertIsNone(result["bootstrap_95ci"])
        for name in ev.CLASSES:
            self.assertEqual(result["classification_report"][name]["support"], 2)

    def test_zero_division_keeps_all_six_classes_in_average(self):
        truth = np.repeat(np.arange(6), 2)
        result = ev.make_metrics(truth, np.zeros(12, dtype=int), 0, 42)
        self.assertAlmostEqual(result["accuracy"], 1/6)
        self.assertAlmostEqual(result["f1_macro"], (2/7) / 6)
        for name in ev.CLASSES[1:]:
            for metric in ("precision", "recall", "f1-score"):
                self.assertEqual(result["classification_report"][name][metric], 0)
        # Missing true classes must remain zero rows, without NaN or division error.
        missing = ev.make_metrics(np.array([0, 0]), np.array([0, 0]), 0, 42)
        self.assertAlmostEqual(missing["f1_macro"], 1/6)
        self.assertEqual(missing["confusion_row_percent"][5], [0.0] * 6)
        json.dumps(missing, allow_nan=False)

    def test_bootstrap_matches_independent_sklearn_reference(self):
        truth = np.repeat(np.arange(6), [7, 2, 3, 2, 2, 1])
        pred = np.array([0, 0, 0, 1, 1, 2, 3, 1, 0, 2, 2, 5, 3, 1, 4, 0, 5])
        result = ev.stratified_bootstrap(truth, pred, repetitions=80, seed=17)
        rng = np.random.default_rng(17)
        accuracy, macro_f1 = [], []
        groups = [np.flatnonzero(truth == k) for k in range(6)]
        original_support = np.bincount(truth, minlength=6)
        for _ in range(80):
            sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
            np.testing.assert_array_equal(np.bincount(truth[sampled], minlength=6), original_support)
            accuracy.append(accuracy_score(truth[sampled], pred[sampled]))
            macro_f1.append(f1_score(truth[sampled], pred[sampled], labels=range(6), average="macro", zero_division=0))
        np.testing.assert_allclose(result["accuracy"], np.quantile(accuracy, [.025, .975]), rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["f1_macro"], np.quantile(macro_f1, [.025, .975]), rtol=0, atol=1e-14)
        self.assertEqual(result, ev.stratified_bootstrap(truth, pred, 80, 17))
        self.assertNotEqual(result["accuracy"], ev.stratified_bootstrap(truth, pred, 80, 18)["accuracy"])

    def test_perfect_predictions_have_degenerate_bootstrap_intervals(self):
        truth = np.repeat(np.arange(6), [8, 4, 3, 2, 2, 1])
        result = ev.stratified_bootstrap(truth, truth.copy(), 40, 42)
        self.assertEqual(result["accuracy"], [1.0, 1.0])
        self.assertEqual(result["f1_macro"], [1.0, 1.0])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)

    def lock(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return ev.create_lock(self.root)

    def test_manifest_rejects_duplicate_paths_class_mismatch_and_leakage(self):
        mutations = [
            ("Duplicate source paths", lambda f: f.assign(path=[f.path.iloc[1]] + f.path.iloc[1:].tolist())),
            ("label/class order", lambda f: f.assign(class_name=["trash"] + f.class_name.iloc[1:].tolist())),
            ("cross data splits", lambda f: f.assign(pixel_sha256=["repeated"] * len(f))),
            ("six classes", lambda f: f[~((f.split == "test") & (f.label == 5))]),
        ]
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                mutate(self.manifest.copy()).to_csv(self.root / "data/manifest.csv", index=False)
                with self.assertRaisesRegex(ValueError, expected):
                    ev.load_manifest(self.root)
        self.manifest.to_csv(self.root / "data/manifest.csv", index=False)
        self.assertEqual(len(ev.load_manifest(self.root)), len(self.manifest))

    def test_lock_selects_macro_f1_not_accuracy_and_breaks_ties_stably(self):
        # Inference must never run while creating the validation lock.
        with patch.object(ev, "predict_test", side_effect=AssertionError("Test access forbidden")):
            lock = self.lock()
        self.assertEqual(lock["selected_experiment"], "mlp_128_weighted")
        self.assertEqual(lock["selected_cnn_experiment"], "cnn_128_unweighted")
        validation = lock["validation_metrics"]
        self.assertLess(validation["mlp_128_weighted"]["accuracy"], validation["mlp_128_unweighted"]["accuracy"])
        self.assertAlmostEqual(validation["mlp_128_weighted"]["f1_macro"], .775)
        before = (self.root / ev.LOCK_NAME).read_bytes()
        self.assertEqual(self.lock(), lock)
        self.assertEqual((self.root / ev.LOCK_NAME).read_bytes(), before)

    def test_validation_coverage_and_truth_are_checked_before_selection(self):
        path = self.root / "runs" / ev.EXPERIMENTS[0] / "val_predictions.csv"
        original = pd.read_csv(path)
        for mode in ("missing", "duplicate", "wrong_truth", "test_row"):
            frame = original.copy()
            if mode == "missing":
                frame = frame.iloc[:-1]
            elif mode == "duplicate":
                frame.loc[0, "path"] = frame.loc[1, "path"]
            elif mode == "wrong_truth":
                frame.loc[0, "true_label"] = 5
            else:
                frame.loc[0, "path"] = self.manifest.loc[self.manifest.split == "test", "path"].iloc[0]
            frame.to_csv(path, index=False)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.lock()
            self.assertFalse((self.root / ev.LOCK_NAME).exists())
        original.to_csv(path, index=False)

    def test_lock_blocks_mutated_checkpoint_config_source_and_cache_metadata(self):
        self.lock()
        for relative in ("runs/mlp_128_unweighted/model.pt", "runs/cnn_128_weighted/config.json",
                         "src/models.py", "data/cache/128x96/metadata.json", "data/manifest.csv"):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\nchanged")
            with self.subTest(path=relative), self.assertRaisesRegex(ValueError, "Frozen experiment files changed"):
                ev.verify_lock(self.root)
            path.write_bytes(original)
        self.assertEqual(ev.verify_lock(self.root)["selected_experiment"], "mlp_128_weighted")

    def test_lock_rejects_tampered_selection_even_with_unchanged_checkpoint_hashes(self):
        lock = self.lock()
        lock["selected_experiment"] = "cnn_512_unweighted"
        (self.root / ev.LOCK_NAME).write_bytes(ev.json_bytes(lock))
        with self.assertRaisesRegex(ValueError, "model-selection decision"):
            ev.verify_lock(self.root)

    def test_stale_training_metadata_and_mixed_configuration_are_rejected(self):
        run = self.root / "runs" / ev.EXPERIMENTS[0]
        original = (run / "metadata.json").read_text()
        for key in ("manifest_sha256", "model_file_sha256", "config_sha256"):
            metadata = json.loads(original)
            metadata[key] = "not-the-recorded-hash"
            (run / "metadata.json").write_text(json.dumps(metadata))
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.lock()
            self.assertFalse((self.root / ev.LOCK_NAME).exists())
        (run / "metadata.json").write_text(original)
        config = json.loads((run / "config.json").read_text())
        config["seed"] = 43
        (run / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.lock()

    def test_existing_test_results_prevent_retroactive_lock(self):
        (self.root / "runs" / ev.EXPERIMENTS[0] / "metrics.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Test artifacts already exist"):
            self.lock()
        self.assertFalse((self.root / ev.LOCK_NAME).exists())

    def test_result_publication_is_idempotent_but_refuses_overwrite(self):
        path = self.root / "reports/result.json"
        ev.write_once(path, b'{"value": 1}\n')
        ev.write_once(path, b'{"value": 1}\n')
        with self.assertRaises(FileExistsError):
            ev.write_once(path, b'{"value": 2}\n')
        self.assertEqual(path.read_bytes(), b'{"value": 1}\n')

    def test_cost_uses_measured_fields_and_mib_without_estimation(self):
        experiment = ev.EXPERIMENTS[0]
        metadata = {"parameters": 100, "trainable_parameters": 100, "epochs_completed": 4,
                    "best_epoch": 3, "total_training_seconds": 16.8, "mean_epoch_seconds": 4.2,
                    "peak_rss_bytes": 512 * 1024**2, "config": {"batch_size": 32}}
        result = ev.cost_fields(metadata, self.root, experiment)
        self.assertEqual(result["train_seconds"], 16.8)
        self.assertEqual(result["mean_epoch_seconds"], 4.2)
        self.assertEqual(result["peak_rss_mb"], 512)
        self.assertEqual(result["batch_size"], 32)
        self.assertIsNone(result["peak_gpu_mb"])
        self.assertEqual(result["checkpoint_bytes"], (self.root / "runs" / experiment / "model.pt").stat().st_size)
        del metadata["total_training_seconds"]
        with self.assertRaisesRegex(ValueError, "required measured cost"):
            ev.cost_fields(metadata, self.root, experiment)


if __name__ == "__main__":
    unittest.main()
