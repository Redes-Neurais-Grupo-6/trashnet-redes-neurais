"""Busca exploratória de taxa comum às arquiteturas, somente por validação.

Os três candidatos são comparados pela média do F1 macro de validação da
MLP128 e CNN128 sem pesos, nos checkpoints de menor perda. A busca não usa teste.
As execuções preliminares de 1e-3 já existentes podem ser reutilizadas.
"""
from pathlib import Path
import argparse
import csv
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IDS = ["mlp_128_unweighted", "cnn_128_unweighted"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "selection_lock.json").exists():
        raise RuntimeError("Teste já congelado; não executar nova busca neste estudo.")
    search = root / "search"
    search.mkdir(exist_ok=True)
    document = json.loads((root / "configs/experiments.json").read_text())
    protocol = {"candidate_learning_rates": [0.001, 0.0003, 0.0001],
                "ranking": "mean of restored validation macro F1 for unweighted MLP128 and CNN128",
                "selection": "maximum mean; first candidate wins exact ties",
                "trigger": "Exploratory search after unstable initial validation learning curves; test not accessed.",
                "unchanged": "architecture, seed42, split, optimizer Adam, batch32, max25, patience5, preprocessing",
                "test_used": False}
    (search / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    for rate, folder in [(0.001,"lr_1e-3"),(0.0003,"lr_3e-4"),(0.0001,"lr_1e-4")]:
        trial = search / folder
        (trial / "configs").mkdir(parents=True, exist_ok=True)
        config = json.loads(json.dumps(document))
        config["defaults"]["learning_rate"] = rate
        (trial / "configs/experiments.json").write_text(json.dumps(config, indent=2))
        data = trial / "data"
        if not data.exists():
            data.symlink_to(root / "data", target_is_directory=True)
        for exp in IDS:
            result = trial / "runs" / exp / "metadata.json"
            if rate == .001 and not result.exists():
                origin = root / "runs" / exp
                if origin.exists():
                    shutil.copytree(origin, result.parent, ignore=shutil.ignore_patterns("last.pt"))
            if not result.exists():
                subprocess.run([sys.executable,"-u","-m","src.train","--root",str(trial),"--experiment",exp],
                               cwd=root,check=True)
            meta = json.loads(result.read_text())
            if meta["config"]["learning_rate"] != rate:
                raise ValueError("Taxa do resultado não corresponde à busca")
            rows.append({"learning_rate":rate,"architecture":meta["config"]["architecture"],
                         "experiment":exp,"val_f1_macro":meta["restored_val_metrics"]["f1_macro"],
                         "val_accuracy":meta["restored_val_metrics"]["accuracy"],
                         "best_epoch":meta["best_epoch"],"epochs":meta["epochs_completed"],
                         "seconds":meta["total_training_seconds"]})
        # O cache compartilhado não pertence aos registros finais da busca.
        data.unlink()
    scores = {rate: sum(r["val_f1_macro"] for r in rows if r["learning_rate"] == rate)/2
              for rate in protocol["candidate_learning_rates"]}
    selected = max(scores, key=scores.get)
    with (search / "validation_trials.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (search / "selection.json").write_text(json.dumps({"mean_val_f1_by_rate":scores,"selected_learning_rate":selected,
                          "test_used":False,"decision_before_final_test":True}, indent=2))
    print(json.dumps({"mean_val_f1_by_rate":scores,"selected_learning_rate":selected}),flush=True)

if __name__ == "__main__":
    main()
