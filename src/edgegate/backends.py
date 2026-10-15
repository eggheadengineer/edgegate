"""Inference backends. Each exposes bench() -> latency samples and, if it can run arbitrary inputs, infer()."""
import hashlib
import pathlib
import re
import shutil
import subprocess
import time

import numpy as np


class BackendError(Exception):
    pass


def model_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _random_input(shape, dtype):
    return np.random.default_rng(0).standard_normal(shape).astype(dtype)


# ------------------------------------------------------------------ ONNX Runtime (host, or Jetson with the NVIDIA wheel)

class OnnxRuntimeBackend:
    name = "onnxruntime"
    can_infer = True

    def __init__(self, model_path, input_spec, precision="fp32"):
        try:
            import onnxruntime as ort
        except ImportError:
            raise BackendError("onnxruntime is not installed (pip install edgegate[host])") from None
        self.ort = ort
        providers = ort.get_available_providers()
        preferred = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
        self.session = ort.InferenceSession(str(model_path), providers=preferred)
        self.provider = self.session.get_providers()[0]
        meta = self.session.get_inputs()[0]
        self.input_name = input_spec["name"] or meta.name
        self.shape = tuple(input_spec["shape"])
        self.dtype = np.dtype(input_spec["dtype"])
        self.output_name = self.session.get_outputs()[0].name
        self.precision = "fp32"   # ORT runs the graph as exported

    def describe(self):
        return {"backend": self.name, "provider": self.provider, "version": self.ort.__version__, "precision": self.precision}

    def infer(self, x):
        return self.session.run([self.output_name], {self.input_name: x})[0]

    def bench(self, warmup, iterations, duration_s=0):
        x = _random_input(self.shape, self.dtype)
        for _ in range(warmup):
            self.infer(x)
        samples = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            self.infer(x)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        return samples


# ------------------------------------------------------------------ TensorRT via trtexec (Jetson)

TRTEXEC_LATENCY_RE = re.compile(
    r"Latency:\s*min\s*=\s*([\d.]+)\s*ms,\s*max\s*=\s*([\d.]+)\s*ms,\s*mean\s*=\s*([\d.]+)\s*ms,"
    r"\s*median\s*=\s*([\d.]+)\s*ms,\s*percentile\((\d+)%\)\s*=\s*([\d.]+)\s*ms")
TRTEXEC_THROUGHPUT_RE = re.compile(r"Throughput:\s*([\d.]+)\s*qps")


def parse_trtexec_output(text):
    """Pull the latency summary out of trtexec's stdout. Returns dict or raises BackendError."""
    m = TRTEXEC_LATENCY_RE.search(text)
    if not m:
        raise BackendError("could not find a 'Latency:' summary line in trtexec output")
    pct = int(m.group(5))
    out = {"min": float(m.group(1)), "max": float(m.group(2)), "mean": float(m.group(3)),
           "median": float(m.group(4)), f"p{pct}": float(m.group(6))}
    t = TRTEXEC_THROUGHPUT_RE.search(text)
    if t:
        out["throughput_qps"] = float(t.group(1))
    return out


class TrtexecBackend:
    """Builds (and caches) a TensorRT engine with trtexec, then benchmarks it. Accuracy needs onnxruntime."""
    name = "trtexec"
    can_infer = False

    def __init__(self, model_path, input_spec, precision="fp16", cache_dir=None):
        self.trtexec = shutil.which("trtexec") or ("/usr/src/tensorrt/bin/trtexec" if pathlib.Path("/usr/src/tensorrt/bin/trtexec").exists() else None)
        if not self.trtexec:
            raise BackendError("trtexec not found on PATH or at /usr/src/tensorrt/bin/trtexec")
        self.model_path = pathlib.Path(model_path)
        self.shape = tuple(input_spec["shape"])
        self.input_name = input_spec["name"]
        self.precision = precision
        self.cache_dir = pathlib.Path(cache_dir or self.model_path.parent / ".edgegate-engines")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tag = model_sha256(self.model_path)[7:19]
        self.engine = self.cache_dir / f"{self.model_path.stem}-{tag}-{precision}-{'x'.join(map(str, self.shape))}.engine"
        self.version = self._version()

    def _version(self):
        try:
            out = subprocess.run([self.trtexec, "--help"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        m = re.search(r"TensorRT\s+v?([\d.]+)", out)
        return m.group(1) if m else None

    def describe(self):
        return {"backend": self.name, "trtexec": self.trtexec, "version": self.version,
                "precision": self.precision, "engine": str(self.engine)}

    def _run(self, args, timeout):
        try:
            p = subprocess.run([self.trtexec, *args], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise BackendError(f"trtexec timed out after {timeout}s") from None
        if p.returncode != 0:
            tail = "\n".join((p.stdout + p.stderr).splitlines()[-15:])
            raise BackendError(f"trtexec failed (exit {p.returncode}):\n{tail}")
        return p.stdout

    def build(self):
        if self.engine.exists():
            return False
        args = [f"--onnx={self.model_path}", f"--saveEngine={self.engine}"]
        if self.precision == "fp16":
            args.append("--fp16")
        elif self.precision == "int8":
            args.append("--int8")
        if self.input_name:
            dims = "x".join(map(str, self.shape))
            args += [f"--shapes={self.input_name}:{dims}"]
        self._run(args, timeout=1800)
        return True

    def bench(self, warmup, iterations, duration_s=0):
        """trtexec runs for at least `iterations` AND at least `duration_s` seconds (its own default is 3 s).
        Power sampling needs a window long enough for DVFS to settle, so the Jetson example sets 10 s."""
        self.build()
        out = self._run([f"--loadEngine={self.engine}", f"--warmUp={max(warmup * 10, 200)}",
                         f"--iterations={iterations}", f"--duration={duration_s}", "--percentile=99"],
                        timeout=max(600, duration_s * 3 + 120))
        self.last = parse_trtexec_output(out)
        # trtexec reports a summary, not per-iteration samples; synthesize the fields the gate needs.
        return self.last


def make_backend(cfg):
    m = cfg["model"]
    if m["backend"] == "onnxruntime":
        return OnnxRuntimeBackend(m["path"], m["input"], m.get("precision", "fp32"))
    return TrtexecBackend(m["path"], m["input"], m.get("precision", "fp16"))
