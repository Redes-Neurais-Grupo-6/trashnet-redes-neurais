"""Aquisição, inspeção, divisão fixa e cache do TrashNet de referência.

Execute na raiz do projeto: python -m src.prepare --zip /caminho/dataset-resized.zip
Sem --zip, baixa o arquivo oficial. Nenhuma imagem aumentada é criada.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
EXPECTED = dict(zip(CLASSES, [501, 594, 403, 482, 410, 137]))
URL = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip?download=true"
ZIP_SHA256 = "c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def prepare(root: Path, archive: Path | None = None):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = data / "dataset-resized.zip"
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
    if sha256(archive) != ZIP_SHA256:
        raise ValueError("ZIP diferente da referência verificada. Não continuar silenciosamente.")
    raw = data / "raw" / "dataset-resized"
    raw.mkdir(parents=True, exist_ok=True)
    ignored = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            if (len(parts) == 3 and parts[0] == "dataset-resized" and
                    parts[1] in CLASSES and parts[2].lower().endswith((".jpg", ".png", ".jpeg"))):
                target = raw / parts[1] / parts[2]
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(z.read(info))
            elif not info.is_dir():
                ignored.append(info.filename)
    rows, invalid = [], []
    for label, name in enumerate(CLASSES):
        for path in sorted((raw / name).glob("*")):
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    rgb = ImageOps.exif_transpose(im).convert("RGB")
                    pixels = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
                    rows.append(dict(path=path.relative_to(raw).as_posix(), class_name=name,
                                     label=label, width=im.width, height=im.height, mode=im.mode,
                                     sha256=sha256(path), pixel_sha256=pixels))
            except Exception as exc:
                invalid.append({"path": str(path.relative_to(raw)), "error": repr(exc)})
    manifest = pd.DataFrame(rows)
    counts = manifest.class_name.value_counts().to_dict()
    if counts != EXPECTED or invalid:
        (data / "integrity_failure.json").write_text(json.dumps({"counts": counts, "invalid": invalid}, indent=2))
        raise ValueError("A referência tem arquivos ausentes/inválidos. Ver data/integrity_failure.json.")
    # A comparação de pixels identifica duplicatas exatas mesmo com metadados diferentes.
    duplicates = manifest[manifest.duplicated("pixel_sha256", keep=False)]
    if len(duplicates):
        duplicates.to_csv(data / "exact_duplicates.csv", index=False)
    ids = np.arange(len(manifest))
    train, holdout = train_test_split(ids, test_size=.30, stratify=manifest.label, random_state=42)
    val, test = train_test_split(holdout, test_size=.50, stratify=manifest.loc[holdout, "label"], random_state=42)
    manifest["split"] = ""
    for split, ix in [("train", train), ("val", val), ("test", test)]:
        manifest.loc[ix, "split"] = split
    initial = manifest.copy()
    initial.to_csv(data / "manifest_before_grouping.csv", index=False)
    initial_counts = pd.crosstab(manifest.class_name, manifest.split)
    # Corrige duplicatas cruzando partições sem alterar qualquer contagem por classe.
    # A partição da primeira linha de cada grupo é a âncora; não há seleção por
    # desempenho, preferência pelo treino ou mudança dos rótulos oficiais.
    repairs = []
    duplicate_ids = set(duplicates.index)
    for group_hash, group in duplicates.groupby("pixel_sha256", sort=True):
        anchor = int(min(group.index))
        target_split = manifest.loc[anchor, "split"]
        for ix in sorted(group.index):
            old_split = manifest.loc[ix, "split"]
            if old_split == target_split:
                continue
            candidates = manifest[(manifest.split == target_split) &
                                  (manifest.label == manifest.loc[ix, "label"]) &
                                  (~manifest.index.isin(duplicate_ids))]
            if candidates.empty:
                raise ValueError("Não há singleton compatível para corrigir duplicata.")
            swap = int(candidates.index[0])
            manifest.loc[ix, "split"] = target_split
            manifest.loc[swap, "split"] = old_split
            repairs.append(dict(duplicate=manifest.loc[ix,"path"], singleton=manifest.loc[swap,"path"],
                                original_split=old_split, grouped_split=target_split))
    assert manifest.groupby("pixel_sha256").split.nunique().max() == 1
    assert manifest.groupby("sha256").split.nunique().max() == 1
    assert manifest.path.is_unique
    assert pd.crosstab(manifest.class_name, manifest.split).equals(initial_counts)
    # Preserva ordem estável, não a ordem aleatória das partições.
    out = data / "manifest.csv"
    csv = manifest.to_csv(index=False)
    if out.exists() and out.read_text() != csv:
        raise ValueError("A divisão existente difere. Não sobrescrever resultados associados a outra divisão.")
    out.write_text(csv)
    info = {
        "dataset_url": URL, "zip_sha256": ZIP_SHA256, "manifest_sha256": sha256(out),
        "n_images": len(manifest), "class_order": CLASSES, "class_counts": EXPECTED,
        "dimensions": {f"{w}x{h}": int(n) for (w,h), n in manifest.groupby(["width", "height"]).size().items()},
        "modes": manifest["mode"].value_counts().to_dict(), "invalid_images": invalid,
        "exact_duplicate_pixel_groups": int(duplicates.pixel_sha256.nunique()),
        "conflicting_label_duplicate_groups": int(sum(g.label.nunique() > 1 for _,g in duplicates.groupby("pixel_sha256"))),
        "duplicate_repairs": repairs, "ignored_archive_entries": ignored,
        "duplicate_group_splits": manifest[manifest.index.isin(duplicate_ids)][["path","class_name","split","pixel_sha256"]].to_dict("records"),
        "split_counts": pd.crosstab(manifest.class_name, manifest.split).reindex(CLASSES).to_dict(),
        "split_method": "sklearn train_test_split: estratificado 70/30; holdout 50/50, seed=42 em ambos; duplicatas agrupadas na partição da primeira linha estável, com troca de singleton da mesma classe para preservar contagens",
        "preprocessing": "EXIF transpose; RGB; Pillow LANCZOS a partir da referência; tensor float32 / 255",
        "augmentation": "nenhum", "near_duplicate_audit": "Não há identificadores de objeto. Hashes evitam vazamento de duplicatas exatas; semelhança perceptual e múltiplas fotos do mesmo objeto não são totalmente controladas."
    }
    (data / "eda.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps({k:info[k] for k in ["n_images", "dimensions", "modes", "split_counts"]}, indent=2))
    for width, height in [(128,96), (256,192), (512,384)]:
        folder = data / "cache" / f"{width}x{height}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "images.npy"
        expected_shape = (len(manifest), 3, height, width)
        if target.exists() and (folder / "metadata.json").exists():
            md = json.loads((folder / "metadata.json").read_text())
            if md.get("manifest_sha256") == sha256(out):
                existing = np.load(target, mmap_mode="r", allow_pickle=False)
                if existing.shape != expected_shape or existing.dtype != np.uint8 or md.get("layout") != "NCHW":
                    raise ValueError(f"Cache existente tem shape/dtype/layout inválido: {target}")
                if md.get("images_sha256") != sha256(target):
                    raise ValueError(f"Cache existente falhou na verificação SHA-256: {target}. Não sobrescrever silenciosamente.")
                del existing
                print(f"Cache já pronto e verificado: {width}x{height}", flush=True)
                continue
        cache = np.lib.format.open_memmap(target, mode="w+", dtype=np.uint8, shape=expected_shape)
        for i, row in manifest.iterrows():
            with Image.open(raw / row.path) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB").resize((width,height), Image.Resampling.LANCZOS)
                cache[i] = np.asarray(rgb).transpose(2,0,1)
        cache.flush()
        del cache
        (folder / "metadata.json").write_text(json.dumps({"shape":expected_shape,"dtype":"uint8","layout":"NCHW",
                      "manifest_sha256":sha256(out),"images_sha256":sha256(target),"resize":"Pillow LANCZOS"}, indent=2))
        print(f"Cache concluído: {width}x{height}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--zip", type=Path)
    a = p.parse_args()
    prepare(a.root.resolve(), a.zip.resolve() if a.zip else None)
