"""Treino controlado dos seis experimentos. Este módulo nunca avalia o teste.

Uso: python -m src.train --root . --experiment all
Retomada: mesmo comando, acrescentando --resume. Cada época salva last.pt,
enquanto model.pt contém somente os pesos da menor perda de validação.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from .models import build_model

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
HISTORY_FIELDS = ["epoch", "train_loss", "train_accuracy", "train_f1_macro",
                  "val_loss", "val_accuracy", "val_f1_macro", "epoch_seconds",
                  "is_best", "bad_epochs"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    """Hash independente dos metadados de serialização de torch.save."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def seed_everything(seed: int, threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    # set_num_interop_threads só pode ser alterado antes do primeiro trabalho.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def load_experiments(root: Path) -> list[dict[str, Any]]:
    document = json.loads((root / "configs/experiments.json").read_text())
    if document["class_order"] != CLASSES:
        raise ValueError("A ordem das seis classes deve ser a ordem do enunciado.")
    configs = [{**document["defaults"], **item} for item in document["experiments"]]
    if len({config["id"] for config in configs}) != len(configs):
        raise ValueError("IDs de experimentos repetidos.")
    return configs


def read_manifest(root: Path) -> list[dict[str, Any]]:
    with (root / "data/manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest vazio.")
    for index, row in enumerate(rows):
        row["label"] = int(row["label"])
        row["index"] = index
        if not 0 <= row["label"] < len(CLASSES) or CLASSES[row["label"]] != row["class_name"]:
            raise ValueError(f"Classe/rótulo inconsistentes na linha {index}.")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Split inválido na linha {index}.")
    return rows


class CachedImageDataset(Dataset):
    """Leitura sob demanda do cache NCHW; a ordem acompanha o manifest inteiro."""

    def __init__(self, root: Path, config: dict[str, Any],
                 rows: list[dict[str, Any]], split: str) -> None:
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Split sem exemplos: {split}")
        cache = root / "data/cache" / f"{config['width']}x{config['height']}" / "images.npy"
        cache_metadata = json.loads(cache.with_name("metadata.json").read_text())
        if cache_metadata.get("manifest_sha256") != sha256_file(root / "data/manifest.csv"):
            raise ValueError("Cache pertence a outro manifest; execute novamente a preparação.")
        if cache_metadata.get("layout") != "NCHW":
            raise ValueError("O cache deve declarar layout NCHW.")
        self.images = np.load(cache, mmap_mode="r", allow_pickle=False)
        expected = (len(rows), 3, int(config["height"]), int(config["width"]))
        if self.images.shape != expected or self.images.dtype != np.uint8:
            raise ValueError(f"Cache inválido: {cache}; esperado uint8 {expected}, "
                             f"encontrado {self.images.dtype} {self.images.shape}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        # Cópia torna o tensor gravável e evita alterar a memória mapeada.
        image = torch.from_numpy(np.array(self.images[row["index"]], copy=True))
        return image.to(torch.float32).div_(255.0), row["label"], row["index"]


def make_loader(dataset: Dataset, config: dict[str, Any], *, shuffle: bool,
                generator: torch.Generator | None = None) -> DataLoader:
    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=shuffle,
                      num_workers=int(config["num_workers"]), generator=generator,
                      pin_memory=False, drop_last=False)


def class_weight_tensor(rows: list[dict[str, Any]], enabled: bool) -> torch.Tensor | None:
    labels = [row["label"] for row in rows if row["split"] == "train"]
    counts = np.bincount(labels, minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError("Treino deve conter exemplos das seis classes.")
    if not enabled:
        return None
    return torch.tensor(len(labels) / (len(CLASSES) * counts), dtype=torch.float32)


def loss_components(logits: torch.Tensor, targets: torch.Tensor,
                    weights: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = F.cross_entropy(logits, targets, weight=weights, reduction="sum")
    denominator = targets.new_tensor(targets.numel(), dtype=torch.float32)
    if weights is not None:
        denominator = weights[targets].sum()
    return numerator, denominator


def metrics_from_predictions(targets: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, labels=list(range(6)),
                                    average="macro", zero_division=0)),
    }


def train_epoch(model: torch.nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer, weights: torch.Tensor | None,
                max_batches: int | None = None) -> dict[str, Any]:
    model.train()
    loss_sum = denominator_sum = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    batch_times: list[float] = []
    for batch_index, (images, targets, _) in enumerate(loader):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        numerator, denominator = loss_components(logits, targets, weights)
        loss = numerator / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("Perda de treino não finita; execução interrompida.")
        loss.backward()
        optimizer.step()
        loss_sum += float(numerator.detach())
        denominator_sum += float(denominator)
        targets_all.extend(targets.tolist())
        predictions_all.extend(logits.detach().argmax(dim=1).tolist())
        batch_times.append(time.perf_counter() - tick)
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return {"loss": loss_sum / denominator_sum,
            **metrics_from_predictions(targets_all, predictions_all),
            "batch_seconds": batch_times, "samples": len(targets_all)}


@torch.inference_mode()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader,
                    weights: torch.Tensor | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Retorna métricas e previsões; o treino chama esta função somente em val."""
    model.eval()
    loss_sum = denominator_sum = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    records: list[dict[str, Any]] = []
    for images, targets, indices in loader:
        logits = model(images)
        numerator, denominator = loss_components(logits, targets, weights)
        if not torch.isfinite(numerator):
            raise FloatingPointError("Perda de validação não finita.")
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        loss_sum += float(numerator)
        denominator_sum += float(denominator)
        targets_all.extend(targets.tolist())
        predictions_all.extend(predictions.tolist())
        for index, true, predicted, probs in zip(indices.tolist(), targets.tolist(),
                                                  predictions.tolist(), probabilities.tolist()):
            records.append({"index": index, "true_label": true,
                            "predicted_label": predicted,
                            "predicted_probability": probs[predicted],
                            **{f"prob_{name}": probs[i] for i, name in enumerate(CLASSES)}})
    return {"loss": loss_sum / denominator_sum,
            **metrics_from_predictions(targets_all, predictions_all)}, records


class PeakRSS:
    """Pico RSS do processo, amostrado a cada 50 ms; não é memória GPU."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.initial = self.peak
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.05):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRSS":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop_event.set()
        self.thread.join()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def save_torch_atomic(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def hardware_info() -> dict[str, Any]:
    cpu_name = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    return {"cpu": cpu_name, "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "system_ram_bytes": psutil.virtual_memory().total,
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "device": "cpu", "gpu_used": False}


def assert_protocol_open(root: Path) -> None:
    """Congela treinamento assim que seleção ou avaliação do teste foi registrada."""
    guards = [root / "selection_lock.json", root / "reports/selection_lock.json",
              root / "results/test_metrics.json", root / "results/test_predictions.csv"]
    guards.extend(root.glob("runs/*/metrics.json"))
    guards.extend(root.glob("runs/*/predictions.csv"))
    if any(path.exists() for path in guards):
        raise RuntimeError("Protocolo já congelado ou teste já avaliado. "
                           "Não reajuste estes experimentos após a seleção final.")


def run_experiment(root: Path, config: dict[str, Any], *, resume: bool = False,
                   benchmark_batches: int = 0) -> dict[str, Any]:
    root = root.resolve()
    assert_protocol_open(root)
    seed_everything(int(config["seed"]), int(config["threads"]))
    rows = read_manifest(root)
    manifest_sha256 = sha256_file(root / "data/manifest.csv")
    config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    train_data = CachedImageDataset(root, config, rows, "train")
    val_data = CachedImageDataset(root, config, rows, "val")
    train_generator = torch.Generator().manual_seed(int(config["seed"]))
    val_generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
    train_loader = make_loader(train_data, config, shuffle=True, generator=train_generator)
    val_loader = make_loader(val_data, config, shuffle=False, generator=val_generator)
    model = build_model(config)
    initial_state_sha256 = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                                 weight_decay=float(config["weight_decay"]))
    weights = class_weight_tensor(rows, bool(config["class_weights"]))
    if benchmark_batches:
        start = time.perf_counter()
        with PeakRSS() as memory:
            result = train_epoch(model, train_loader, optimizer, weights,
                                 max_batches=benchmark_batches)
        benchmark = {"id": config["id"], "samples": result["samples"],
                     "batch_seconds": result["batch_seconds"],
                     "elapsed_seconds": time.perf_counter() - start,
                     "peak_rss_bytes": memory.peak,
                     "train_batches_per_epoch": len(train_loader),
                     "val_batches_per_epoch": len(val_loader),
                     "note": "Somente lotes de treino; pesos descartados; nenhum teste avaliado."}
        print(json.dumps(benchmark, ensure_ascii=False), flush=True)
        return benchmark

    run_dir = root / "runs" / config["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    best_path = run_dir / "model.pt"
    if (run_dir / "metadata.json").exists():
        if resume:
            completed = json.loads((run_dir / "metadata.json").read_text())
            if completed["config_sha256"] != config_sha256 or completed["manifest_sha256"] != manifest_sha256:
                raise ValueError("Execução concluída pertence a outra configuração/divisão.")
            if completed["model_file_sha256"] != sha256_file(best_path):
                raise ValueError("Arquivo model.pt da execução concluída foi alterado.")
            print(f"{config['id']}: já concluído; preservado.", flush=True)
            return completed
        raise FileExistsError(f"{run_dir} já está concluído. Use --resume para preservar resultados.")
    if last_path.exists() and not resume:
        raise FileExistsError(f"Checkpoint existente em {last_path}; use --resume.")

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = bad_epochs = start_epoch = 0
    prior_seconds = 0.0
    prior_peak = 0
    started_at = datetime.now(timezone.utc).isoformat()
    if resume and last_path.exists():
        # Somente checkpoints locais produzidos por este programa devem ser retomados.
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["config_sha256"] != config_sha256 or checkpoint["manifest_sha256"] != manifest_sha256:
            raise ValueError("Configuração ou divisão foi alterada; retomada recusada.")
        if checkpoint["initial_state_sha256"] != initial_state_sha256:
            raise ValueError("Arquitetura/inicialização foi alterada; retomada recusada.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint["history"]
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
        bad_epochs, start_epoch = checkpoint["bad_epochs"], checkpoint["epoch"]
        prior_seconds, prior_peak = checkpoint["total_seconds"], checkpoint["peak_rss_bytes"]
        started_at = checkpoint["started_at_utc"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        train_generator.set_state(checkpoint["train_loader_rng"])
        val_generator.set_state(checkpoint["val_loader_rng"])
        if not best_path.exists():
            raise FileNotFoundError("Checkpoint de retomada existe, mas model.pt está ausente.")

    write_json(run_dir / "config.json", config)
    start = time.perf_counter()
    with PeakRSS() as memory:
        for epoch in range(start_epoch + 1, int(config["max_epochs"]) + 1):
            if bad_epochs >= int(config["patience"]):
                break
            epoch_start = time.perf_counter()
            train_metrics = train_epoch(model, train_loader, optimizer, weights)
            val_metrics, _ = evaluate_loader(model, val_loader, weights)
            is_best = val_metrics["loss"] < best_loss - float(config["min_delta"])
            if is_best:
                best_loss, best_epoch, bad_epochs = val_metrics["loss"], epoch, 0
                save_torch_atomic(model.state_dict(), best_path)
            else:
                bad_epochs += 1
            record = {"epoch": epoch,
                      **{f"train_{key}": train_metrics[key] for key in ("loss", "accuracy", "f1_macro")},
                      **{f"val_{key}": val_metrics[key] for key in ("loss", "accuracy", "f1_macro")},
                      "epoch_seconds": time.perf_counter() - epoch_start,
                      "is_best": int(is_best), "bad_epochs": bad_epochs}
            history.append(record)
            # Inclui checkpoint de continuidade no custo; histórico é atualizado
            # depois da gravação. Na retomada, o total preservado é conservador:
            # o último custo de gravação não pode estar dentro do próprio arquivo.
            checkpoint = {"model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                          "history": history, "best_loss": best_loss, "best_epoch": best_epoch,
                          "bad_epochs": bad_epochs, "config_sha256": config_sha256,
                          "manifest_sha256": manifest_sha256,
                          "initial_state_sha256": initial_state_sha256,
                          "total_seconds": prior_seconds + time.perf_counter() - start,
                          "peak_rss_bytes": max(prior_peak, memory.peak),
                          "started_at_utc": started_at,
                          "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                          "torch_rng": torch.get_rng_state(),
                          "train_loader_rng": train_generator.get_state(),
                          "val_loader_rng": val_generator.get_state()}
            save_torch_atomic(checkpoint, last_path)
            record["epoch_seconds"] = time.perf_counter() - epoch_start
            write_csv(run_dir / "history.csv", history, HISTORY_FIELDS)
            print(f"{config['id']} epoch={epoch:02d} "
                  f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                  f"val_f1={val_metrics['f1_macro']:.4f} seconds={record['epoch_seconds']:.2f} "
                  f"best={best_epoch}", flush=True)
        total_seconds = prior_seconds + time.perf_counter() - start

    # Métricas finais se referem obrigatoriamente ao checkpoint restaurado.
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    restored_val, predictions = evaluate_loader(model, val_loader, weights)
    for prediction in predictions:
        prediction["path"] = rows[prediction["index"]]["path"]
        prediction["true_class"] = CLASSES[prediction["true_label"]]
        prediction["predicted_class"] = CLASSES[prediction["predicted_label"]]
    write_csv(run_dir / "val_predictions.csv", predictions, list(predictions[0]))
    metadata = {
        "id": config["id"], "config": config, "class_order": CLASSES,
        "manifest_sha256": manifest_sha256, "config_sha256": config_sha256,
        "initial_state_sha256": initial_state_sha256,
        "saved_state_sha256": state_hash(model), "model_file_sha256": sha256_file(best_path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "model_file_bytes": best_path.stat().st_size,
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "stopped_early": bad_epochs >= int(config["patience"]),
        "total_training_seconds": total_seconds,
        "mean_epoch_seconds": total_seconds / len(history),
        "epoch_seconds_sum": sum(row["epoch_seconds"] for row in history),
        "peak_rss_bytes": max(prior_peak, memory.peak),
        "baseline_rss_bytes": memory.initial,
        "memory_method": "RSS absoluto do processo, psutil, amostragem 50 ms; inclui páginas do cache mapeadas; sem GPU.",
        "timing_method": "perf_counter: loop de treino, validação por época e checkpoints; exclui download, cache, inicialização e avaliação do checkpoint restaurado.",
        "resume_timing_note": "Na retomada, pode faltar o custo de gravação do último last.pt antes da interrupção; uma execução contínua não tem essa limitação.",
        "resumed": bool(resume and start_epoch > 0),
        "train_class_counts": np.bincount([row["label"] for row in rows if row["split"] == "train"], minlength=6).tolist(),
        "class_weight_values": None if weights is None else weights.tolist(),
        "restored_val_metrics": restored_val,
        "selection_metric": restored_val["f1_macro"],
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_info(), "test_evaluated": False,
        "training_metric_note": "Métricas de treino agregadas durante atualizações, com dropout ativo; validação em eval(). Perdas ponderadas normalizadas pela soma dos pesos observados.",
    }
    write_json(run_dir / "metadata.json", metadata)
    print(json.dumps({"completed": config["id"], "best_epoch": best_epoch,
                      "restored_val_f1_macro": restored_val["f1_macro"],
                      "total_seconds": total_seconds}, ensure_ascii=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--experiment", required=True, help="ID em experiments.json, ou all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-batches", type=int, default=0,
                        help="Executa somente N lotes de treino e descarta os pesos; não avalia teste.")
    args = parser.parse_args()
    if args.benchmark_batches < 0:
        parser.error("--benchmark-batches deve ser não negativo.")
    configs = load_experiments(args.root)
    selected = [config for config in configs if args.experiment in {"all", config["id"]}]
    if not selected:
        parser.error(f"Experimento desconhecido: {args.experiment}")
    # Um processo por experimento reduz influência do alocador e do RSS anterior.
    if args.experiment == "all":
        import subprocess
        for config in selected:
            command = [sys.executable, "-m", "src.train", "--root", str(args.root.resolve()),
                       "--experiment", config["id"]]
            if args.resume:
                command.append("--resume")
            if args.benchmark_batches:
                command.extend(["--benchmark-batches", str(args.benchmark_batches)])
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
    else:
        run_experiment(args.root, selected[0], resume=args.resume,
                       benchmark_batches=args.benchmark_batches)


if __name__ == "__main__":
    main()
