"""Run the benchmark and the accuracy pass; produce the results document."""
import datetime as dt
import os
import pathlib
import platform
import subprocess

import numpy as np

from . import __version__, dataset
from .backends import BackendError, make_backend, model_sha256
from .sampler import make_sampler, peak_rss_mb


def _now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def latency_stats(samples):
    a = np.asarray(samples, dtype=np.float64)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p90": round(float(np.percentile(a, 90)), 3),
            "p99": round(float(np.percentile(a, 99)), 3), "mean": round(float(a.mean()), 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3), "n": int(a.size)}


def _device_info(cfg):
    info = {"name": cfg["device"].get("name") or platform.node(), "hostname": platform.node(),
            "machine": platform.machine(), "python": platform.python_version()}
    for p, key in (("/etc/nv_tegra_release", "l4t"), ("/proc/device-tree/model", "model")):
        try:
            info[key] = pathlib.Path(p).read_text(errors="replace").strip().split("\n")[0].rstrip("\x00")
        except OSError:
            pass
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "NV Power Mode" in line:
                info["power_mode"] = line.split(":", 1)[-1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def accuracy_top1(backend, ds):
    correct = total = 0
    for x, label in dataset.load(ds):
        out = backend.infer(x)
        pred = int(np.argmax(np.asarray(out).reshape(out.shape[0], -1), axis=1)[0])
        correct += int(pred == label)
        total += 1
    if total == 0:
        raise BackendError("dataset yielded no samples")
    return {"metric": "top1", "value": round(correct / total, 4), "correct": correct, "n": total}


def run(cfg, notes=None):
    backend = make_backend(cfg)
    bench = cfg["benchmark"]
    with make_sampler(cfg) as sampler:
        raw = backend.bench(bench["warmup"], bench["iterations"], bench.get("duration_s", 0))
    power = sampler.summary()

    if isinstance(raw, dict):          # trtexec summary
        latency = {"p50": raw["median"], "p90": None, "p99": raw.get("p99"), "mean": raw["mean"],
                   "min": raw["min"], "max": raw["max"], "n": bench["iterations"], "source": "trtexec summary"}
        throughput = raw.get("throughput_qps")
    else:
        latency = latency_stats(raw)
        throughput = round(1000.0 / latency["mean"], 2) if latency["mean"] else None

    acc = None
    acc_note = None
    if cfg.get("dataset"):
        if backend.can_infer:
            acc = accuracy_top1(backend, cfg["dataset"])
        else:
            acc_note = f"accuracy skipped: backend '{backend.name}' cannot run arbitrary inputs; install onnxruntime on the device to measure it"

    memory = {"peak_rss_mb": peak_rss_mb()}
    if power and power.get("ram_peak_mb") is not None:
        memory["board_ram_peak_mb"] = power.pop("ram_peak_mb")
    elif power:
        power.pop("ram_peak_mb", None)

    return {
        "edgegate_version": __version__,
        "created_at": _now(),
        "device": _device_info(cfg),
        "model": {"path": os.path.relpath(cfg["model"]["path"], cfg["_root"]), "sha256": model_sha256(cfg["model"]["path"]),
                  "input_shape": cfg["model"]["input"]["shape"], **backend.describe()},
        "benchmark": {"warmup": bench["warmup"], "iterations": bench["iterations"], "duration_s": bench.get("duration_s", 0)},
        "latency_ms": latency,
        "throughput_ips": throughput,
        "memory": memory,
        "power": power,
        "accuracy": acc,
        "notes": [n for n in (notes, acc_note) if n],
    }
