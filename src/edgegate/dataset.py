"""Labeled samples for accuracy measurement. Two formats: an .npz with x and y, or an image folder with labels.csv."""
import csv
import pathlib

import numpy as np


class DatasetError(Exception):
    pass


def load(ds):
    """Yield (input_array, label) pairs. input_array has the model's input shape with batch dimension 1."""
    kind, path, limit = ds["type"], pathlib.Path(ds["path"]), ds.get("limit")
    if kind == "npz":
        yield from _npz(path, limit)
    else:
        yield from _image_folder(path, limit, ds.get("preprocess") or {})


def _npz(path, limit):
    try:
        data = np.load(path)
    except FileNotFoundError:
        raise DatasetError(f"dataset not found: {path}") from None
    if "x" not in data or "y" not in data:
        raise DatasetError(f"{path}: expected arrays 'x' and 'y'")
    x, y = data["x"], data["y"]
    if len(x) != len(y):
        raise DatasetError(f"{path}: x has {len(x)} samples but y has {len(y)}")
    n = len(x) if limit is None else min(limit, len(x))
    for i in range(n):
        yield x[i][None, ...].astype(np.float32), int(y[i])


def _image_folder(path, limit, pre):
    try:
        from PIL import Image
    except ImportError:
        raise DatasetError("image_folder datasets need Pillow (pip install edgegate[host])") from None
    labels = path / "labels.csv"
    if not labels.is_file():
        raise DatasetError(f"{labels}: missing (one 'path,label' row per image)")
    size = tuple(pre.get("size", [224, 224]))
    mean = np.array(pre.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(pre.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    scale = float(pre.get("scale", 1.0 / 255.0))
    layout = pre.get("layout", "NCHW")
    with labels.open(newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    if rows and not {"path", "label"} <= set(rows[0]):
        raise DatasetError(f"{labels}: header must be 'path,label'")
    if limit is not None:
        rows = rows[:limit]
    for r in rows:
        img = Image.open(path / r["path"]).convert("RGB").resize(size)
        arr = (np.asarray(img, dtype=np.float32) * scale - mean) / std      # HWC
        if layout == "NCHW":
            arr = arr.transpose(2, 0, 1)
        yield arr[None, ...].astype(np.float32), int(r["label"])
