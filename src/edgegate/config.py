"""Project configuration: a small YAML file next to the model."""
import pathlib

import yaml

DEFAULTS = {
    "benchmark": {"warmup": 20, "iterations": 200, "batch": 1},
    "power": {"sampler": "auto", "interval_ms": 200},
    "thresholds": {},
    "device": {"name": None},
}


class ConfigError(Exception):
    pass


def load(path):
    path = pathlib.Path(path)
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise ConfigError(f"config not found: {path}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"config is not valid YAML: {path}: {e}") from None

    if cfg.get("version") != 1:
        raise ConfigError("config: 'version: 1' is required")
    model = cfg.get("model")
    if not isinstance(model, dict) or "path" not in model:
        raise ConfigError("config: model.path is required")
    if model.get("backend") not in ("onnxruntime", "trtexec"):
        raise ConfigError("config: model.backend must be 'onnxruntime' or 'trtexec'")
    inp = model.get("input") or {}
    if "shape" not in inp or not isinstance(inp["shape"], list) or not all(isinstance(d, int) and d > 0 for d in inp["shape"]):
        raise ConfigError("config: model.input.shape must be a list of positive integers (batch first)")
    model["input"] = {"name": inp.get("name"), "shape": inp["shape"], "dtype": inp.get("dtype", "float32")}
    model.setdefault("precision", "fp16")

    ds = cfg.get("dataset")
    if ds is not None:
        if ds.get("type") not in ("npz", "image_folder"):
            raise ConfigError("config: dataset.type must be 'npz' or 'image_folder'")
        if "path" not in ds:
            raise ConfigError("config: dataset.path is required")
        ds.setdefault("limit", None)
        ds.setdefault("preprocess", {})

    for section, defaults in DEFAULTS.items():
        merged = dict(defaults)
        merged.update(cfg.get(section) or {})
        cfg[section] = merged

    t = cfg["thresholds"]
    for key in ("latency_p99_ms_max", "memory_peak_mb_max", "power_w_max", "accuracy_min",
                "latency_p99_regression_pct", "memory_regression_pct", "power_regression_pct", "accuracy_drop_max",
                "latency_regression_floor_ms", "memory_regression_floor_mb", "power_regression_floor_w"):
        if key in t and not isinstance(t[key], (int, float)):
            raise ConfigError(f"config: thresholds.{key} must be a number")

    cfg["_root"] = path.resolve().parent
    cfg["model"]["path"] = str((cfg["_root"] / model["path"]).resolve())
    if ds is not None:
        ds["path"] = str((cfg["_root"] / ds["path"]).resolve())
    return cfg
