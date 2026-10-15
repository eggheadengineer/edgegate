"""Power and memory sampling during a benchmark.

On Jetson, `tegrastats` prints one line per interval with per-rail power as "NAME cur_mW/avg_mW" and
RAM as "RAM used/totalMB". Orin rails: VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0. Xavier rails: VDD_IN,
VDD_CPU_GPU_CV, VDD_SOC. The parser is rail-agnostic; the total is the sum of whatever rails appear,
except that an input rail (VIN_SYS_5V0 or VDD_IN) is reported on its own as the board total when present.
"""
import re
import resource
import shutil
import subprocess
import threading
import time

RAIL_RE = re.compile(r"\b([A-Z][A-Z0-9_]+)\s+(\d+)mW/(\d+)mW")
RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
INPUT_RAILS = ("VIN_SYS_5V0", "VDD_IN", "VDD_IN_SYS")


def parse_tegrastats_line(line):
    """Return {'rails': {name: cur_mW}, 'ram_mb': used or None} for one tegrastats line."""
    rails = {m.group(1): int(m.group(2)) for m in RAIL_RE.finditer(line)}
    ram = RAM_RE.search(line)
    return {"rails": rails, "ram_mb": int(ram.group(1)) if ram else None}


def summarize(samples):
    """Average rails over samples; compute a board total. Returns None if no samples."""
    samples = [s for s in samples if s["rails"]]
    if not samples:
        return None
    names = sorted({n for s in samples for n in s["rails"]})
    rails = {n: round(sum(s["rails"].get(n, 0) for s in samples) / len(samples) / 1000.0, 3) for n in names}
    inp = next((n for n in INPUT_RAILS if n in rails), None)
    total = rails[inp] if inp else round(sum(rails.values()), 3)
    rams = [s["ram_mb"] for s in samples if s["ram_mb"] is not None]
    return {"rails_w": rails, "total_w": total, "total_rail": inp or "sum", "samples": len(samples),
            "ram_peak_mb": max(rams) if rams else None}


class TegrastatsSampler:
    """Runs tegrastats in the background and collects lines until stopped."""

    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.samples = []
        self._proc = None
        self._thread = None

    @staticmethod
    def available():
        return shutil.which("tegrastats") is not None

    def __enter__(self):
        self._proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        time.sleep(self.interval_ms / 1000.0 * 2)   # let the first sample land
        return self

    def _pump(self):
        for line in self._proc.stdout:
            self.samples.append(parse_tegrastats_line(line))

    def __exit__(self, *exc):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self):
        return summarize(self.samples)


class NoSampler:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self):
        return None


def make_sampler(cfg):
    want = cfg["power"]["sampler"]
    if want == "none":
        return NoSampler()
    if want == "tegrastats" and not TegrastatsSampler.available():
        raise RuntimeError("power.sampler is 'tegrastats' but tegrastats is not on PATH")
    if TegrastatsSampler.available():
        return TegrastatsSampler(cfg["power"]["interval_ms"])
    return NoSampler()


def peak_rss_mb():
    """Peak resident set size of this process, in MB (Linux reports KB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
