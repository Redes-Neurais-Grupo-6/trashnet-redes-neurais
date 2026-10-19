"""Reproduz o protocolo final em pasta nova, preservando o estudo publicado.

Exemplo: python -m src.reproduce --destination ../trashnet_reproducao --prepare-only
Sem --prepare-only, também treina os seis modelos, congela a seleção, avalia e
recria as figuras. A busca exploratória de taxa não é repetida por padrão.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_copy(destination: Path, source: Path = SOURCE_ROOT) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destino já existe; escolha uma pasta nova: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("Escolha uma pasta fora do projeto original para preservar seus artefatos.")
    for required in ("src", "configs", "tests", "requirements.txt", "data/manifest.csv"):
        if not (source / required).exists():
            raise FileNotFoundError(f"Fonte de reprodução incompleta: {source / required}")
    destination.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")
    for folder in ("src", "configs", "tests"):
        shutil.copytree(source / folder, destination / folder, ignore=ignore)
    for filename in ("requirements.txt", "environment.json", "README.md", "GUIA_DA_EQUIPE.md", ".gitignore"):
        if (source / filename).is_file():
            shutil.copy2(source / filename, destination / filename)
    readme = destination / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        text = re.sub(r"## Resultados medidos\n.*?(?=## 2\. Ambiente)", "", text, flags=re.S)
        readme.write_text(
            "> **Cópia para reprodução:** este manual descreve o protocolo de origem. "
            "Os novos resultados serão gerados em `runs/` e `reports/`; artigo e "
            "interpretações numéricas do estudo anterior não são resultados desta execução.\n\n" + text,
            encoding="utf-8")
    if (source / "notebooks").is_dir():
        # As saídas dos notebooks originais pertencem ao estudo de origem; só
        # código/texto vão para a reprodução, que gerará suas próprias saídas.
        (destination / "notebooks").mkdir()
        for notebook in (source / "notebooks").glob("*.ipynb"):
            document = json.loads(notebook.read_text(encoding="utf-8"))
            # Interpretações numéricas pertencem ao estudo original, assim como
            # as saídas: não apresentá-las como conclusões de uma nova execução.
            document["cells"] = [cell for cell in document.get("cells", [])
                                 if cell.get("metadata", {}).get("analysis_source") != "article_final"]
            for cell in document.get("cells", []):
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            (destination / "notebooks" / notebook.name).write_text(
                json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (destination / "data").mkdir()
    for filename in ("manifest.csv", "manifest_before_grouping.csv", "eda.json", "exact_duplicates.csv"):
        if (source / "data" / filename).is_file():
            shutil.copy2(source / "data" / filename, destination / "data" / filename)
    # Os notebooks auditam também as previsões de validação da busca original.
    # Copiar somente registros leves permite essa auditoria sem sugerir que a
    # reprodução executou novamente a busca, nem duplicar pesos ou caches.
    search_source = source / "search"
    if search_source.is_dir():
        for directory, subdirectories, files in os.walk(search_source, followlinks=False):
            subdirectories[:] = [name for name in subdirectories
                                 if name not in {"data", "__pycache__"}]
            for name in files:
                original = Path(directory) / name
                if original.suffix not in {".json", ".csv"}:
                    continue
                target = destination / "search" / original.relative_to(search_source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
        (destination / "search").mkdir(exist_ok=True)
        (destination / "search/ORIGIN.md").write_text(
            "# Proveniência da busca\n\nEstes registros leves foram copiados do estudo de origem. "
            "A reprodução usa sua configuração final e não executou novamente a busca. "
            "Os custos, históricos e previsões aqui pertencem às execuções originais, "
            "não aos novos treinos em `../runs/`. Pesos e caches da busca não foram copiados.\n",
            encoding="utf-8")
    provenance = destination / "provenance"
    provenance.mkdir()
    for filename in ("protocol.json", "selection.json", "validation_trials.csv"):
        original = source / "search" / filename
        if original.is_file():
            shutil.copy2(original, provenance / f"source_search_{filename}")
    original_lock = source / "selection_lock.json"
    if original_lock.is_file():
        shutil.copy2(original_lock, provenance / "source_selection_lock.json")
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": digest(source / "data/manifest.csv"),
        "source_config_sha256": digest(source / "configs/experiments.json"),
        "source_selection_lock_sha256": digest(original_lock) if original_lock.is_file() else None,
        "mode": "reproduction of the fixed final protocol; no new hyperparameter search",
        "copied": "source, final configs, tests, dependencies, original manifest, inspection records and clearly identified lightweight source search records",
        "not_copied": "models, histories, test predictions, test metrics, root selection lock, image caches",
        "interpretation": "Reproducing this already evaluated protocol is not an independent new test. Changes informed by its test results require a separately declared exploratory study and fresh external validation.",
    }
    (provenance / "reproduction.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path,
                        help="Pasta nova, fora do projeto original; nunca sobrescreve um destino.")
    parser.add_argument("--zip", type=Path, help="dataset-resized.zip de referência; sem este argumento, baixa o ZIP.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Copia o protocolo e prepara dados; deixa o treinamento para execução manual.")
    args = parser.parse_args()
    if args.destination.exists() or args.destination.is_symlink():
        parser.error(f"Destino já existe; escolha uma pasta nova: {args.destination}")
    destination = args.destination.resolve()
    archive = args.zip.resolve() if args.zip else None
    if archive is not None and not archive.is_file():
        parser.error(f"ZIP não encontrado: {archive}")
    create_copy(destination)
    prepare = [sys.executable, "-m", "src.prepare", "--root", "."]
    if archive:
        prepare.extend(["--zip", str(archive)])
    commands = [prepare]
    if not args.prepare_only:
        commands.extend([
            [sys.executable, "-m", "src.train", "--root", ".", "--experiment", "all"],
            [sys.executable, "-m", "src.evaluate", "--root", ".", "--lock"],
            [sys.executable, "-m", "src.evaluate", "--root", ".", "--evaluate"],
            [sys.executable, "-m", "src.visualize", "--root", "."],
        ])
    for command in commands:
        print("Executando em", destination, ":", " ".join(command), flush=True)
        subprocess.run(command, cwd=destination, check=True)
    print(f"Reprodução {'preparada' if args.prepare_only else 'concluída'} em: {destination}")
    if args.prepare_only:
        print("Na pasta criada, usando este mesmo ambiente Python, execute:")
        print("python -m src.train --root . --experiment all")
        print("python -m src.evaluate --root . --lock")
        print("python -m src.evaluate --root . --evaluate")
        print("python -m src.visualize --root .")


if __name__ == "__main__":
    main()
