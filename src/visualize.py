"""Generate the numbered, reproducible study figures from measured artifacts.

Run ``python -m src.visualize`` after ``python -m src.evaluate --evaluate``.
Outputs are PNG (300 dpi) and editable SVG files under reports/figures/.
Only training images are used for illustrative dataset examples. The error
figure always uses the checkpoint selected by validation, never by test score.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trashnet-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .evaluate import CLASSES, EXPERIMENTS, ROOT, load_manifest, sha256_file, verify_lock

DISPLAY = {
    "mlp_128_unweighted": "MLP · 128 × 96 · sem pesos",
    "mlp_128_weighted": "MLP · 128 × 96 · com pesos",
    "cnn_128_unweighted": "CNN · 128 × 96 · sem pesos",
    "cnn_128_weighted": "CNN · 128 × 96 · com pesos",
    "cnn_256_unweighted": "CNN · 256 × 192 · sem pesos",
    "cnn_512_unweighted": "CNN · 512 × 384 · sem pesos",
}
SHORT = {e: DISPLAY[e].replace(" · ", "\n", 1) for e in EXPERIMENTS}
TRAIN_COLOR, VAL_COLOR = "#2166ac", "#b35806"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.2,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "legend.fontsize": 8, "figure.dpi": 120, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "svg.fonttype": "none", "figure.facecolor": "white",
    })


def save(fig, output: Path, stem: str):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def source_image(root: Path, relative_path: str):
    path = root / "data/raw/dataset-resized" / relative_path
    if not path.is_file():
        # Support manifests containing a repository-relative full raw-data path.
        path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Source image not found: {relative_path}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def distribution(root: Path, manifest: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(7.2, 2.7), layout="constrained")
    x, width = np.arange(6), 0.26
    for i, (split, label, color) in enumerate([
        ("train", "Treino", "#2166ac"), ("val", "Validação", "#67a9cf"), ("test", "Teste", "#b35806")
    ]):
        counts = manifest.loc[manifest.split == split, "class_name"].value_counts().reindex(CLASSES).to_numpy()
        bars = ax.bar(x + (i - 1) * width, counts, width=width, label=f"{label} (n={counts.sum()})", color=color)
        ax.bar_label(bars, fontsize=7.5, padding=2)
    ax.set(xticks=x, xticklabels=CLASSES, ylabel="Número de imagens")
    ax.set_ylim(0, manifest[manifest.split == "train"].class_name.value_counts().max() * 1.22)
    ax.legend(loc="upper right", frameon=False, ncols=3, fontsize=7.7)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    save(fig, output, "fig01_distribuicao")


def examples(root: Path, manifest: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 2.8), layout="constrained")
    choices = []
    for column, name in enumerate(CLASSES):
        # A prespecified index rule avoids choosing unusually clean examples.
        subset = manifest[(manifest.split == "train") & (manifest.class_name == name)].sort_values("path")
        selected = subset.iloc[[0, len(subset) // 2]]
        for row, (_, item) in enumerate(selected.iterrows()):
            axes[row, column].imshow(source_image(root, item.path))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(name, fontsize=9.2)
            axes[row, column].text(0.5, -0.04, Path(item.path).name, transform=axes[row, column].transAxes,
                                   ha="center", va="top", fontsize=6.5)
            choices.append({"class_name": name, "path": item.path, "split": "train"})
    save(fig, output, "fig02_exemplos")
    return choices


def resolutions(root: Path, manifest: pd.DataFrame, output: Path):
    selected = manifest[(manifest.split == "train") & (manifest.class_name == "plastic")].sort_values("path").iloc[0]
    manifest_index = int(selected.name)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), layout="constrained")
    for ax, (width, height) in zip(axes, [(128, 96), (256, 192), (512, 384)]):
        cache_path = root / "data/cache" / f"{width}x{height}" / "images.npy"
        metadata = json.loads(cache_path.with_name("metadata.json").read_text())
        if metadata.get("manifest_sha256") != sha256_file(root / "data/manifest.csv"):
            raise ValueError("Resolution figure cache does not match the current manifest")
        cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        resized = np.asarray(cache[manifest_index]).transpose(1, 2, 0)
        ax.imshow(resized, interpolation="nearest")
        ax.set_title(f"{width} × {height}")
        ax.axis("off")
    save(fig, output, "fig03_resolucoes")
    return {"path": selected.path, "split": "train", "display": "Equal physical panel size; nearest-neighbor display of actual NCHW cache images, generated with EXIF transpose and Pillow LANCZOS from the reference source."}


def history_frame(root: Path, experiment: str):
    frame = pd.read_csv(root / "runs" / experiment / "history.csv")
    aliases = {"train_acc": "train_accuracy", "val_acc": "val_accuracy", "accuracy": "train_accuracy", "loss": "train_loss"}
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns and v not in frame.columns})
    required = {"epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{experiment}: history missing {required - set(frame.columns)}")
    return frame


def learning_curves(root: Path, output: Path):
    fig = plt.figure(figsize=(7.2, 7.4), layout="constrained")
    panels = fig.subfigures(3, 2, wspace=0.03, hspace=0.03)
    for panel, experiment in zip(panels.flat, EXPERIMENTS):
        frame = history_frame(root, experiment)
        metadata = json.loads((root / "runs" / experiment / "metadata.json").read_text())
        axes = panel.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.08})
        panel.suptitle(DISPLAY[experiment], fontsize=9, fontweight="bold")
        for ax, metric, ylabel in zip(axes, ["loss", "accuracy"], ["Perda", "Acurácia"]):
            ax.plot(frame.epoch, frame[f"train_{metric}"], color=TRAIN_COLOR, label="Treino", linewidth=1.25)
            ax.plot(frame.epoch, frame[f"val_{metric}"], color=VAL_COLOR, linestyle="--", label="Validação", linewidth=1.25)
            if metadata.get("best_epoch") is not None:
                ax.axvline(metadata["best_epoch"], color="#555555", alpha=0.65, linewidth=0.8, linestyle=":")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axes[0].legend(frameon=False, loc="best", ncols=2, fontsize=6.8, handlelength=1.4)
        axes[1].set_ylim(0, 1)
        axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axes[1].set_xlabel("Época", fontsize=8)
    save(fig, output, "fig04_curvas_aprendizagem")


def confusions(root: Path, output: Path):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.0), layout="constrained")
    for ax, experiment in zip(axes.flat, EXPERIMENTS):
        metrics = json.loads((root / "runs" / experiment / "metrics.json").read_text())
        counts = np.array(metrics["confusion_count"])
        percents = np.array(metrics["confusion_row_percent"])
        ax.imshow(percents, vmin=0, vmax=100, cmap="Blues", interpolation="nearest")
        for real in range(6):
            for pred in range(6):
                color = "white" if percents[real, pred] >= 55 else "#172636"
                ax.text(pred, real, f"{counts[real, pred]}\n{percents[real, pred]:.0f}%", ha="center", va="center", color=color, fontsize=8)
        ax.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Classe prevista", ylabel="Classe real", title=DISPLAY[experiment])
        ax.tick_params(axis="x", rotation=35, labelsize=7.5)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)
    save(fig, output, "fig05_matrizes_confusao")


def resolution_metrics(root: Path, output: Path):
    ids = [f"cnn_{w}_unweighted" for w in (128, 256, 512)]
    metrics = [json.loads((root / "runs" / experiment / "metrics.json").read_text()) for experiment in ids]
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout="constrained")
    x = np.arange(3)
    for metric, label, color, offset in [("accuracy", "Acurácia", TRAIN_COLOR, -0.025), ("f1_macro", "F1 macro", VAL_COLOR, 0.025)]:
        values = np.array([m[metric] for m in metrics])
        ci = [m.get("bootstrap_95ci") for m in metrics]
        ax.plot(x + offset, values, marker="o", color=color, label=label, linewidth=1.5)
        if all(ci):
            bounds = np.array([c[metric] for c in ci]).T
            # Percentile intervals need not include the point estimate. Draw absolute bounds.
            ax.vlines(x + offset, bounds[0], bounds[1], color=color, alpha=0.7, linewidth=1)
            ax.hlines(bounds[0], x + offset - 0.035, x + offset + 0.035, color=color, linewidth=1)
            ax.hlines(bounds[1], x + offset - 0.035, x + offset + 0.035, color=color, linewidth=1)
        for xx, value in zip(x + offset, values):
            ax.annotate(f"{100 * value:.1f}%", (xx, value), xytext=(0, 7 if metric == "accuracy" else -13),
                        textcoords="offset points", ha="center", fontsize=8, color=color)
    ax.set(xticks=x, xticklabels=["128 × 96", "256 × 192", "512 × 384"], xlabel="Resolução da entrada (largura × altura)", ylabel="Desempenho no teste", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.7)
    ax.legend(frameon=False, loc="best", ncols=2)
    save(fig, output, "fig06_resolucao_desempenho")


def resolution_cost(root: Path, output: Path):
    summary = pd.read_csv(root / "reports/summary.csv").set_index("experiment")
    subset = summary.loc[[f"cnn_{w}_unweighted" for w in (128, 256, 512)]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), layout="constrained")
    for ax, column, ylabel, title in zip(axes, ["mean_epoch_seconds", "train_seconds"],
                                        ["Segundos por época", "Segundos de treinamento"], ["Tempo médio por época", "Tempo total"]):
        values = subset[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Measured training time is missing: {column}")
        bars = ax.bar(np.arange(3), values, color=["#9ecae1", "#4292c6", "#08519c"], width=0.6)
        ax.bar_label(bars, labels=[f"{x:.1f}" for x in values], fontsize=8, padding=3)
        ax.set(xticks=np.arange(3), xticklabels=["128 × 96", "256 × 192", "512 × 384"], ylabel=ylabel, title=title)
        ax.set_ylim(0, values.max() * 1.2)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
        ax.set_axisbelow(True)
    save(fig, output, "fig07_resolucao_custo")


def errors(root: Path, lock: dict, output: Path):
    experiment = lock["selected_experiment"]
    predictions = pd.read_csv(root / "runs" / experiment / "predictions.csv")
    # Deterministic confidence ordering exposes the strongest model mistakes.
    mistakes = predictions[predictions.y_true != predictions.y_pred].sort_values(["pred_probability", "path"], ascending=[False, True])
    shown = mistakes.head(6)
    if shown.empty:
        fig, ax = plt.subplots(figsize=(7.2, 1.5), layout="constrained")
        ax.axis("off")
        ax.text(0.5, 0.5, "O modelo selecionado pela validação não cometeu erros no conjunto de teste.", ha="center", va="center", wrap=True)
    else:
        nrows = (len(shown) + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(7.2, 2.05 * nrows), layout="constrained", squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, item) in zip(axes.flat, shown.iterrows()):
            ax.imshow(source_image(root, item.path))
            ax.set_title(f"Real: {item.true_class} → prevista: {item.pred_class}\np(prevista) = {100 * item.pred_probability:.1f}%", fontsize=8.3)
            ax.text(0.5, -0.03, Path(item.path).name, transform=ax.transAxes, ha="center", va="top", fontsize=7)
    save(fig, output, "fig08_analise_erros")
    return {"experiment": experiment, "total_test_errors": len(mistakes), "selection_rule": "up to six highest predicted-class probabilities among mistakes; ties by source path", "shown": shown.to_dict("records")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--eda-only", action="store_true", help="Generate only figures 1–3 before model evaluation")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.root.resolve() / "reports/figures"
    style()
    manifest = load_manifest(root)
    distribution(root, manifest, output)
    example_paths = examples(root, manifest, output)
    resolution_source = resolutions(root, manifest, output)
    figure_record = {"class_order": CLASSES, "examples": example_paths, "resolution_example": resolution_source,
                     "source_manifest_sha256": sha256_file(root / "data/manifest.csv"), "formats": ["png", "svg"], "png_dpi": 300}
    if not args.eda_only:
        lock = verify_lock(root)
        summary = json.loads((root / "reports/summary.json").read_text())
        if summary["selection_lock_sha256"] != sha256_file(root / "selection_lock.json"):
            raise ValueError("Summary does not match the frozen model selection")
        for experiment in EXPERIMENTS:
            run = root / "runs" / experiment
            metrics = json.loads((run / "metrics.json").read_text())
            if metrics["predictions_sha256"] != sha256_file(run / "predictions.csv"):
                raise ValueError(f"Prediction file changed after evaluation: {experiment}")
        learning_curves(root, output)
        confusions(root, output)
        resolution_metrics(root, output)
        resolution_cost(root, output)
        figure_record["errors"] = errors(root, lock, output)
        figure_record["selection_lock_sha256"] = sha256_file(root / "selection_lock.json")
    (output / "figure_sources.json").write_text(json.dumps(figure_record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Figures saved in {output}")


if __name__ == "__main__":
    main()
