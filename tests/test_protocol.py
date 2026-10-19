"""Invariantes do estudo: execute python -m unittest discover -s tests -v."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from src import train
from src.models import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_fixture(root: Path) -> dict:
    """Pequeno conjunto artificial isolado; nunca usa os resultados reais."""
    cache_dir = root / "data/cache/16x16"
    cache_dir.mkdir(parents=True)
    rows = []
    for split, repetitions in (("train", 2), ("val", 1), ("test", 1)):
        for label, class_name in enumerate(train.CLASSES):
            for repetition in range(repetitions):
                rows.append({"path": f"{class_name}/{split}_{repetition}.jpg",
                             "class_name": class_name, "label": label, "split": split,
                             "sha256": f"synthetic-{split}-{label}-{repetition}"})
    with (root / "data/manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(cache_dir / "images.npy", np.random.default_rng(7).integers(
        0, 256, size=(len(rows), 3, 16, 16), dtype=np.uint8))
    train.write_json(cache_dir / "metadata.json", {
        "layout": "NCHW", "manifest_sha256": train.sha256_file(root / "data/manifest.csv")})
    config = train.load_experiments(PROJECT_ROOT)[0]
    config.update(id="synthetic", width=16, height=16, batch_size=6,
                  max_epochs=3, patience=3)
    return config


class ProtocolTests(unittest.TestCase):
    def test_equal_initialization_and_constant_cnn_size(self) -> None:
        configs = train.load_experiments(PROJECT_ROOT)
        hashes, counts = {}, {}
        for config in configs:
            train.seed_everything(config["seed"], config["threads"])
            model = build_model(config)
            hashes[config["id"]] = train.state_hash(model)
            counts[config["id"]] = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(hashes["mlp_128_unweighted"], hashes["mlp_128_weighted"])
        self.assertEqual(len({value for key, value in hashes.items() if key.startswith("cnn_")}), 1)
        self.assertEqual({value for key, value in counts.items() if key.startswith("cnn_")}, {24086})
        self.assertEqual(counts["mlp_128_unweighted"], 4727366)
        for left, right in ((configs[0], configs[1]), (configs[2], configs[3])):
            self.assertEqual({key: value for key, value in left.items() if key not in {"id", "class_weights"}},
                             {key: value for key, value in right.items() if key not in {"id", "class_weights"}})

    def test_weights_use_training_only_and_loss_normalizes_correctly(self) -> None:
        rows = [{"split": "train", "label": label} for label in range(6)]
        rows.extend([{"split": "train", "label": 0}] * 3)
        expected = torch.tensor([9 / 24] + [9 / 6] * 5)
        weights = train.class_weight_tensor(rows, True)
        self.assertTrue(torch.equal(weights, expected))
        rows.extend([{"split": "val", "label": 5}, {"split": "test", "label": 5}] * 100)
        self.assertTrue(torch.equal(train.class_weight_tensor(rows, True), weights))
        logits = torch.arange(24, dtype=torch.float32).view(4, 6) / 10
        labels = torch.tensor([0, 1, 5, 0])
        numerator, denominator = train.loss_components(logits, labels, weights)
        self.assertTrue(torch.allclose(numerator / denominator,
                                       F.cross_entropy(logits, labels, weight=weights)))
        parts = [train.loss_components(logits[:1], labels[:1], weights),
                 train.loss_components(logits[1:], labels[1:], weights)]
        self.assertTrue(torch.allclose(sum(pair[0] for pair in parts) / sum(pair[1] for pair in parts),
                                       numerator / denominator))

    def test_cache_rejects_manifest_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = synthetic_fixture(root)
            rows = train.read_manifest(root)
            dataset = train.CachedImageDataset(root, config, rows, "train")
            image, label, index = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 16, 16))
            self.assertEqual(len(dataset), 12)
            self.assertGreaterEqual(float(image.min()), 0)
            self.assertLessEqual(float(image.max()), 1)
            self.assertEqual(rows[index]["split"], "train")
            manifest = root / "data/manifest.csv"
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "outro manifest"):
                train.CachedImageDataset(root, config, rows, "train")

    def test_selection_lock_and_test_results_block_training(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train.assert_protocol_open(root)
            for relative in ("selection_lock.json", "reports/selection_lock.json",
                             "runs/example/metrics.json", "runs/example/predictions.csv"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "congelado"):
                    train.assert_protocol_open(root)
                path.unlink()

    def test_resume_reproduces_continuous_weights_and_restores_best(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = synthetic_fixture(root)
            config["id"] = "continuous"
            uninterrupted = train.run_experiment(root, config)
            resumed_config = {**config, "id": "resumed"}
            original_save = train.save_torch_atomic

            def interrupt_after_save(value: object, path: Path) -> None:
                original_save(value, path)
                if path.name == "last.pt" and value["epoch"] == 1:
                    raise KeyboardInterrupt("Interrupção artificial após a primeira época")

            with patch.object(train, "save_torch_atomic", interrupt_after_save):
                with self.assertRaises(KeyboardInterrupt):
                    train.run_experiment(root, resumed_config)
            resumed = train.run_experiment(root, resumed_config, resume=True)
            self.assertEqual(uninterrupted["saved_state_sha256"], resumed["saved_state_sha256"])
            self.assertEqual(uninterrupted["restored_val_metrics"], resumed["restored_val_metrics"])
            with (root / "runs/continuous/history.csv").open() as handle:
                history = list(csv.DictReader(handle))
            best = min(history, key=lambda row: float(row["val_loss"]))
            self.assertEqual(uninterrupted["best_epoch"], int(best["epoch"]))
            self.assertAlmostEqual(uninterrupted["restored_val_metrics"]["loss"], float(best["val_loss"]))
            self.assertFalse(uninterrupted["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
