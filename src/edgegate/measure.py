"""Run the benchmark and the accuracy pass; produce the results document."""
import datetime as dt
import os
import pathlib
import platform
import re
import signal
import subprocess
import threading
import time

import numpy as np

from . import __version__, dataset
from .backends import BackendError, make_backend, model_sha256
from .sampler import make_sampler, peak_rss_mb, write_timeline


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


class ProcessTreeRss:
    """Polls /proc for every process in the session started for the command and records the peak summed RSS."""

    def __init__(self, sid, interval_s=0.5):
        self.sid, self.interval_s, self.peak_mb = sid, interval_s, 0.0
        self.series = []                      # (t_s, rss_mb) for the timeline
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        total_kb = 0
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
                if int(stat.rsplit(")", 1)[1].split()[3]) != self.sid:     # field 6 = session id
                    continue
                for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total_kb += int(line.split()[1]); break
            except (OSError, ValueError, IndexError):
                continue
        return total_kb / 1024.0

    def _loop(self):
        while not self._stop.is_set():
            mb = self._sample()
            self.series.append((round(time.monotonic() - self._t0, 3), round(mb, 1)))
            self.peak_mb = max(self.peak_mb, mb)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start(); return self

    def __exit__(self, *exc):
        self._stop.set(); self._thread.join(timeout=3)
        self.peak_mb = max(self.peak_mb, self._sample())


def _count_metrics(metrics, out_dir, wall_s):
    found = {}
    for m in metrics:
        path = pathlib.Path(m["file"].replace("{out}", str(out_dir)))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            found[m["name"]] = {"count": None, "per_s": None, "file": str(path), "note": "file not found"}
            continue
        n = len(re.findall(m["pattern"], text, re.MULTILINE))   # ^ and $ anchor per line, as in grep
        found[m["name"]] = {"count": n, "per_s": round(n / wall_s, 3) if wall_s else None, "file": str(path)}
    return found


def forensics(samples, rss_series, end_t_s, window_s=30.0):
    """What the board looked like in the last `window_s` seconds before the process died (or before the window
    closed). `samples` are tegrastats samples with t_s; `rss_series` is [(t_s, rss_mb)] for the process tree.
    death_t_s = last moment the process tree still held more than half its peak RSS; None if it never fell."""
    death = None
    if rss_series:
        peak = max(v for _, v in rss_series)
        alive = [t for t, v in rss_series if v > peak * 0.5]
        if alive and alive[-1] < rss_series[-1][0] - 1.0:      # it fell more than a second before sampling stopped
            death = alive[-1]
    t_end = death if death is not None else end_t_s
    win = [x for x in samples if x.get("t_s") is not None and t_end - window_s <= x["t_s"] <= t_end]
    out = {"window_s": window_s, "death_t_s": round(death, 1) if death is not None else None, "samples": len(win)}
    rams = [x["ram_mb"] for x in win if x.get("ram_mb") is not None]
    if rams:
        out["ram_mb"] = {"min": min(rams), "max": max(rams), "mean": round(sum(rams) / len(rams))}
        # swings: how many times RAM moved by more than a quarter of its range inside the window.
        # 0 or 1 = a plateau or a single step; many = allocation cycling.
        span = max(rams) - min(rams)
        swings, last = 0, rams[0]
        for v in rams[1:]:
            if span and abs(v - last) > span * 0.25:
                swings += 1; last = v
        out["ram_swings"] = swings
        # plateau: steady (a healthy engine after warm-up). step: one excursion, up or up-and-back (a load
        # spike, or a crash right after one). cycling: repeated large swings (an allocation that keeps failing).
        out["ram_pattern"] = "plateau" if span < 512 or swings <= 1 else ("step" if swings == 2 else "cycling")
    rails = sorted({n for x in win for n in x["rails"]})
    if rails:
        out["rails_w"] = {r: round(sum(x["rails"].get(r, 0) for x in win) / len(win) / 1000.0, 2) for r in rails}
    rss_win = [v for t, v in rss_series if t_end - window_s <= t <= t_end]
    if rss_win:
        out["process_rss_mb"] = {"min": round(min(rss_win)), "max": round(max(rss_win))}
    return out


def run_command(cfg, backend, out_dir, notes=None):
    """The command backend: run the process for benchmark.duration_s (or until it exits), sampling meanwhile."""
    bench = cfg["benchmark"]
    duration = bench.get("duration_s") or 0
    out_dir = pathlib.Path(out_dir).resolve()      # absolute: the command runs with its own cwd (the config dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = lambda x: x.replace("{out}", str(out_dir))
    cmd = [backend.path, *[sub(a) for a in backend.args]]
    env = {**os.environ, **{k: sub(str(v)) for k, v in backend.env.items()}}
    stdout_path = out_dir / "stdout.log"
    timed_out = False
    started_at = _now()
    t0 = time.monotonic()
    with open(stdout_path, "wb") as log, make_sampler(cfg) as sampler:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=sub(backend.cwd) if backend.cwd else None,
                                env=env, start_new_session=True)
        with ProcessTreeRss(proc.pid) as rss:
            try:
                proc.wait(timeout=duration if duration > 0 else None)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=10)
    wall_s = round(time.monotonic() - t0, 2)
    exit_code = 124 if timed_out else proc.returncode      # 124 mirrors coreutils timeout: ran the full window
    power = sampler.summary()
    memory = {"peak_rss_mb": round(rss.peak_mb, 1)}
    if power and power.get("ram_peak_mb") is not None:
        memory["board_ram_peak_mb"] = power.pop("ram_peak_mb")
    elif power:
        power.pop("ram_peak_mb", None)
    metrics = _count_metrics(cfg.get("metrics", []), out_dir, wall_s)
    timeline = None
    if write_timeline(out_dir / "timeline.csv", sampler.timeline(), {"process_rss_mb": rss.series}):
        timeline = str(out_dir / "timeline.csv")
    last = forensics(sampler.timeline(), rss.series, wall_s)
    none = {"p50": None, "p90": None, "p99": None, "mean": None, "min": None, "max": None, "n": 0,
            "source": "n/a: command backend has no per-inference latency; see metrics[*].per_s"}
    return {
        "edgegate_version": __version__,
        "created_at": _now(),
        "device": _device_info(cfg),
        "model": {"path": os.path.relpath(cfg["model"]["path"], cfg["_root"]), "sha256": model_sha256(cfg["model"]["path"]),
                  "input_shape": None, **backend.describe()},
        "benchmark": {"warmup": 0, "iterations": 0, "duration_s": duration},
        "command": {"exit_code": exit_code, "timed_out": timed_out, "wall_s": wall_s, "stdout": str(stdout_path),
                    "argv": cmd, "timeline": timeline, "started_at": started_at, "forensics": last},
        "metrics": metrics,
        "latency_ms": none,
        "throughput_ips": None,
        "memory": memory,
        "power": power,
        "accuracy": None,
        "notes": [n for n in (notes,) if n],
    }


def run(cfg, notes=None, out_dir=None):
    backend = make_backend(cfg)
    if backend.name == "command":
        return run_command(cfg, backend, out_dir or (pathlib.Path(cfg["_root"]) / "edgegate-out"), notes)
    bench = cfg["benchmark"]
    with make_sampler(cfg) as sampler:
        raw = backend.bench(bench["warmup"], bench["iterations"], bench.get("duration_s", 0))
    power = sampler.summary()
    if power and out_dir:
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        if write_timeline(pathlib.Path(out_dir) / "timeline.csv", sampler.timeline()):
            power["timeline"] = str(pathlib.Path(out_dir) / "timeline.csv")

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
