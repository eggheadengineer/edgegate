"""Project configuration: a small YAML file next to the model."""
import pathlib
import re

import yaml

DEFAULTS = {
    "benchmark": {"warmup": 20, "iterations": 200, "batch": 1, "duration_s": 0},
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
    if model.get("backend") not in ("onnxruntime", "trtexec", "command"):
        raise ConfigError("config: model.backend must be 'onnxruntime', 'trtexec' or 'command'")
    if model["backend"] == "command":
        if not isinstance(model.get("args", []), list) or not all(isinstance(x, str) for x in model.get("args", [])):
            raise ConfigError("config: model.args must be a list of strings")
        if not isinstance(model.get("env", {}), dict):
            raise ConfigError("config: model.env must be a mapping")
        model["input"] = {"name": None, "shape": None, "dtype": None}
        model.setdefault("precision", None)
        if cfg.get("dataset") is not None:
            raise ConfigError("config: dataset is not supported with the command backend (count metrics instead)")
    else:
        inp = model.get("input") or {}
        if "shape" not in inp or not isinstance(inp["shape"], list) or not all(isinstance(d, int) and d > 0 for d in inp["shape"]):
            raise ConfigError("config: model.input.shape must be a list of positive integers (batch first)")
        model["input"] = {"name": inp.get("name"), "shape": inp["shape"], "dtype": inp.get("dtype", "float32")}
        model.setdefault("precision", "fp16")

    metrics = cfg.get("metrics") or []
    if not isinstance(metrics, list):
        raise ConfigError("config: metrics must be a list")
    for i, m in enumerate(metrics):
        if not isinstance(m, dict) or not all(isinstance(m.get(k), str) and m.get(k) for k in ("name", "file", "pattern")):
            raise ConfigError(f"config: metrics[{i}] needs string fields name, file and pattern")
        try:
            re.compile(m["pattern"])
        except re.error as e:
            raise ConfigError(f"config: metrics[{i}].pattern is not a valid regex: {e}") from None
    cfg["metrics"] = metrics

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
    if "ok_exit_codes" in t and (not isinstance(t["ok_exit_codes"], list) or not all(isinstance(x, int) for x in t["ok_exit_codes"])):
        raise ConfigError("config: thresholds.ok_exit_codes must be a list of integers")
    for key in ("metric_min", "metric_regression_pct", "metric_regression_floor"):
        if key in t and (not isinstance(t[key], dict) or not all(isinstance(v, (int, float)) for v in t[key].values())):
            raise ConfigError(f"config: thresholds.{key} must map metric names to numbers")

    cfg["_root"] = path.resolve().parent
    cfg["model"]["path"] = str((cfg["_root"] / model["path"]).resolve())
    if ds is not None:
        ds["path"] = str((cfg["_root"] / ds["path"]).resolve())
    return cfg
