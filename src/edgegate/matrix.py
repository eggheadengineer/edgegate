"""Run a matrix of variants x repeats and tabulate crash rate and resources per cell.

Matrix file (YAML):
    version: 1
    base: edgegate.yaml              # the config every cell starts from (path relative to this file)
    repeats: 6
    duration_s: 150                  # optional: overrides benchmark.duration_s for every cell
    stale_processes: [myengine]      # optional: kill processes whose /proc comm starts with these before each run
    cells:
      - { name: baseline }
      - { name: b2,   args: { set: { "-b": "2" } } }       # replace the value after a flag (or append flag+value)
      - { name: trt,  args: { add: ["-S"] }, duration_s: 240 }
      - { name: fp32, env: { PRECISION: fp32 } }
      - { name: pm25, power_mode: 3 }                      # nvpmodel id; must switch without a reboot
Cells are interleaved (rep 1 of every cell, then rep 2, ...) so slow drift cannot favour one cell, and a run
whose results file already exists is skipped, so an interrupted matrix resumes where it stopped.
"""
import copy
import datetime as dt
import json
import os
import pathlib
import re
import signal
import subprocess
import time

import yaml

from .compare import evaluate, status
from .config import ConfigError, load as load_config
from .measure import run as measure_run


class MatrixError(Exception):
    pass


def load_matrix(path):
    path = pathlib.Path(path)
    try:
        m = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise ConfigError(f"matrix file not found: {path}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"matrix file is not valid YAML: {e}") from None
    if m.get("version") != 1:
        raise ConfigError("matrix: 'version: 1' is required")
    if not m.get("base"):
        raise ConfigError("matrix: base (a config file) is required")
    cells = m.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ConfigError("matrix: cells must be a non-empty list")
    names = set()
    for i, c in enumerate(cells):
        if not isinstance(c, dict) or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(c.get("name", ""))):
            raise ConfigError(f"matrix: cells[{i}].name is required (letters, digits, _ . -)")
        if c["name"] in names:
            raise ConfigError(f"matrix: duplicate cell name {c['name']}")
        names.add(c["name"])
        a = c.get("args") or {}
        if not isinstance(a, dict) or not isinstance(a.get("set", {}), dict) or not isinstance(a.get("add", []), list):
            raise ConfigError(f"matrix: cells[{i}].args must have 'set' (mapping) and/or 'add' (list)")
        if "power_mode" in c and not isinstance(c["power_mode"], int):
            raise ConfigError(f"matrix: cells[{i}].power_mode must be an nvpmodel id (integer)")
    m["repeats"] = int(m.get("repeats", 1))
    if m["repeats"] < 1:
        raise ConfigError("matrix: repeats must be >= 1")
    m["_root"] = path.resolve().parent
    m["base"] = str((m["_root"] / m["base"]).resolve())
    m.setdefault("stale_processes", [])
    m.setdefault("max_crash_rate", 0.0)        # a cell fails its JUnit case when crashed/n exceeds this
    return m


def apply_cell(base_cfg, cell, matrix):
    """A fresh config for one cell: base config with the cell's argument, env, duration and note edits applied."""
    cfg = copy.deepcopy(base_cfg)
    model = cfg["model"]
    a = cell.get("args") or {}
    if a and model["backend"] != "command":
        raise ConfigError(f"matrix: cell {cell['name']} edits args but the base backend is {model['backend']}")
    args = list(model.get("args") or [])
    for flag, val in (a.get("set") or {}).items():
        val = str(val)
        if flag in args and args.index(flag) + 1 < len(args):
            args[args.index(flag) + 1] = val
        else:
            args += [flag, val]
    args += [str(x) for x in (a.get("add") or [])]
    model["args"] = args
    if cell.get("env"):
        model["env"] = {**(model.get("env") or {}), **{k: str(v) for k, v in cell["env"].items()}}
    dur = cell.get("duration_s", matrix.get("duration_s"))
    if dur is not None:
        cfg["benchmark"]["duration_s"] = dur
    return cfg


# ---- nvpmodel helpers (Jetson); all tolerate a host without nvpmodel

def power_modes(conf_text):
    """{id: name} from /etc/nvpmodel.conf text."""
    return {int(m.group(1)): m.group(2) for m in re.finditer(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=([A-Za-z0-9_]+)", conf_text)}


def current_power_mode():
    try:
        out = subprocess.run(["sudo", "-n", "nvpmodel", "-q"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "NV Power Mode" in line:
            return line.split(":", 1)[-1].strip()
    return None


def set_power_mode(mode_id, modes, wait_s=20):
    """Switch nvpmodel and wait until it reports the expected name. Refuses modes that need a reboot: the
    matrix process would not survive one. Returns the mode name."""
    name = modes.get(mode_id)
    if name is None:
        raise MatrixError(f"nvpmodel id {mode_id} is not in /etc/nvpmodel.conf (known: {modes})")
    if current_power_mode() == name:
        return name
    p = subprocess.run(["sudo", "-n", "nvpmodel", "-m", str(mode_id)], input="no\n", capture_output=True, text=True, timeout=60)
    if "Reboot required" in (p.stdout + p.stderr):
        raise MatrixError(f"nvpmodel mode {mode_id} ({name}) requires a reboot on this board; run it as a separate matrix after rebooting into it")
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if current_power_mode() == name:
            time.sleep(3)
            return name
        time.sleep(1)
    raise MatrixError(f"nvpmodel did not report {name} within {wait_s} s (now: {current_power_mode()})")


def kill_stale(prefixes):
    """SIGKILL every process whose /proc/<pid>/comm starts with one of the prefixes. Returns how many."""
    n = 0
    if not prefixes:
        return 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            comm = pathlib.Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            continue
        if any(comm.startswith(pfx) for pfx in prefixes):
            try:
                os.kill(int(pid), signal.SIGKILL); n += 1
            except (ProcessLookupError, PermissionError):
                pass
    return n


# ---- the run

def _now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_matrix(matrix, out_dir, log=print, dry_run=False):
    base_cfg = load_config(matrix["base"])
    out_dir = pathlib.Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "LOG", "a")
    def say(msg):
        logf.write(f"{_now()} {msg}\n"); logf.flush(); log(msg)
    needs_pm = any("power_mode" in c for c in matrix["cells"])
    modes, start_mode = {}, None
    if needs_pm:
        try:
            modes = power_modes(pathlib.Path("/etc/nvpmodel.conf").read_text())
        except OSError:
            raise MatrixError("a cell sets power_mode but /etc/nvpmodel.conf is not readable (not a Jetson?)") from None
        start_mode = current_power_mode()
        start_id = next((i for i, n in modes.items() if n == start_mode), None)
    say(f"matrix start: {len(matrix['cells'])} cells x {matrix['repeats']} repeats -> {out_dir}")
    done = skipped = 0
    try:
        for rep in range(1, matrix["repeats"] + 1):
            for cell in matrix["cells"]:
                cdir = out_dir / cell["name"]; cdir.mkdir(exist_ok=True)
                rpath = cdir / f"r{rep}.json"
                if rpath.exists():
                    skipped += 1; continue
                cfg = apply_cell(base_cfg, cell, matrix)
                if dry_run:
                    say(f"[dry] {cell['name']} r{rep}: {cfg['model']['path']} {' '.join(cfg['model'].get('args') or [])} duration={cfg['benchmark'].get('duration_s')}")
                    continue
                try:
                    if "power_mode" in cell:
                        set_power_mode(cell["power_mode"], modes)
                    elif needs_pm and start_id is not None:
                        set_power_mode(start_id, modes)
                except MatrixError as e:
                    say(f"== {cell['name']} r{rep}: SKIP {e}")
                    rpath.write_text(json.dumps({"skipped": str(e), "cell": cell["name"], "rep": rep, "created_at": _now()}, indent=1))
                    continue
                k = kill_stale(matrix["stale_processes"])
                if k:
                    time.sleep(3)
                say(f"== {cell['name']} r{rep} mode={current_power_mode() or '-'} killed_stale={k} {_now()}")
                results = measure_run(cfg, notes=f"matrix cell {cell['name']} rep {rep}" + (f": {cell['note']}" if cell.get("note") else ""),
                                      out_dir=cdir / f"r{rep}.d")
                checks = evaluate(results, cfg["thresholds"])
                results["gate"] = {"status": status(checks), "failed": [c.name for c in checks if c.status == "fail"],
                                   "checked": [c.name for c in checks if c.status != "skip"], "baseline": None}
                results["matrix"] = {"cell": cell["name"], "rep": rep}
                rpath.write_text(json.dumps(results, indent=1) + "\n")
                done += 1
                say("   " + summarize_run(results))
    finally:
        if needs_pm and start_id is not None and not dry_run:
            try:
                set_power_mode(start_id, modes)
            except MatrixError as e:
                say(f"could not restore power mode {start_mode}: {e}")
        kill_stale(matrix["stale_processes"])
        say(f"matrix done: {done} new runs, {skipped} already present")
        logf.close()
    return tabulate(matrix, out_dir)


def summarize_run(r):
    if r.get("command"):
        c = r["command"]
        line = f"exit {c['exit_code']}{' (full window)' if c.get('timed_out') else ''} after {c['wall_s']} s"
        for name, m in (r.get("metrics") or {}).items():
            if m.get("count") is not None:
                line += f" · {name} {m['count']}" + (f" ({m['per_s']}/s)" if m.get("per_s") is not None else "")
    else:
        lat = r["latency_ms"]
        line = f"latency p50 {lat.get('p50')} ms, p99 {lat.get('p99')} ms"
    if r.get("power"):
        line += f" · power {r['power']['total_w']} W"
    line += f" · gate {r.get('gate', {}).get('status', '-')}"
    return line


def _crashed(r):
    """A run is a crash when the exit-code check failed; without that rule, when a command run exited early
    with a non-zero code; for latency runs, when the gate failed. None for skipped cells."""
    if "skipped" in r:
        return None
    gate = r.get("gate") or {}
    if r.get("command"):
        if gate.get("failed") is not None and any(c == "exit_code" for c in gate["failed"]):
            return True
        if "exit_code" in gate.get("checked", []):
            return False
        return not r["command"].get("timed_out") and r["command"].get("exit_code") != 0
    return gate.get("status") == "fail"


def tabulate(matrix, out_dir):
    """Per-cell summary from whatever result files exist. Returns the summary dict and writes summary.json."""
    out_dir = pathlib.Path(out_dir)
    cells = []
    for cell in matrix["cells"]:
        rows = []
        for f in sorted((out_dir / cell["name"]).glob("r*.json"), key=lambda p: int(p.stem[1:])):
            r = json.loads(f.read_text())
            if "skipped" in r:
                rows.append({"rep": r["rep"], "skipped": r["skipped"]}); continue
            cr = _crashed(r)
            row = {"rep": r.get("matrix", {}).get("rep", int(f.stem[1:])), "crashed": cr,
                   "gate": r.get("gate", {}).get("status"), "failed": r.get("gate", {}).get("failed", []),
                   "power_w": (r.get("power") or {}).get("total_w"), "board_ram_peak_mb": r["memory"].get("board_ram_peak_mb"),
                   "peak_rss_mb": r["memory"].get("peak_rss_mb")}
            if r.get("command"):
                row.update({"exit_code": r["command"]["exit_code"], "wall_s": r["command"]["wall_s"],
                            "forensics": r["command"].get("forensics"),
                            "metrics": {k: v.get("count") for k, v in (r.get("metrics") or {}).items()},
                            "rates": {k: v.get("per_s") for k, v in (r.get("metrics") or {}).items() if v.get("per_s") is not None}})
            else:
                row.update({"latency_p99_ms": r["latency_ms"].get("p99"), "throughput_ips": r.get("throughput_ips")})
            rows.append(row)
        measured = [x for x in rows if "skipped" not in x]
        crashed = [x for x in measured if x["crashed"]]
        clean = [x for x in measured if not x["crashed"]]
        def mean(key, src):
            vals = [x[key] for x in src if x.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None
        rates = {}
        for k in {k for x in clean for k in (x.get("rates") or {})}:
            vals = [x["rates"][k] for x in clean if x.get("rates", {}).get(k) is not None]
            rates[k] = round(sum(vals) / len(vals), 3) if vals else None
        rate = len(crashed) / len(measured) if measured else None
        cells.append({"name": cell["name"], "note": cell.get("note"), "n": len(measured), "skipped": len(rows) - len(measured),
                      "crashed": len(crashed), "crash_rate": round(rate, 3) if rate is not None else None,
                      "time_to_crash_s": [x["wall_s"] for x in crashed if "wall_s" in x],
                      "clean_power_w": mean("power_w", clean), "clean_rates": rates,
                      "clean_latency_p99_ms": mean("latency_p99_ms", clean),
                      "board_ram_peak_mb": max((x["board_ram_peak_mb"] or 0 for x in measured), default=None),
                      "forensic_patterns": sorted({(x.get("forensics") or {}).get("ram_pattern") for x in crashed if x.get("forensics")} - {None}),
                      "status": "skip" if not measured else ("fail" if rate > matrix.get("max_crash_rate", 0.0) else "pass"),
                      "runs": rows})
    summary = {"edgegate_matrix": 1, "created_at": _now(), "base": matrix["base"], "repeats": matrix["repeats"],
               "max_crash_rate": matrix.get("max_crash_rate", 0.0), "cells": cells}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def markdown(summary):
    L = [f"# edgegate matrix: {pathlib.Path(summary['base']).name}, {summary['repeats']} repeats per cell", "",
         "| cell | crashed | time to crash | clean rate | clean power W | board RAM peak MB | RAM before crash | verdict |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in summary["cells"]:
        tc = c["time_to_crash_s"]
        tcs = f"{min(tc):.0f}–{max(tc):.0f} s" if len(tc) > 1 else (f"{tc[0]:.0f} s" if tc else "—")
        rate = ", ".join(f"{k} {v}/s" for k, v in (c["clean_rates"] or {}).items()) or (f"p99 {c['clean_latency_p99_ms']} ms" if c.get("clean_latency_p99_ms") else "—")
        L.append(f"| {c['name']}{' (' + c['note'] + ')' if c.get('note') else ''} | {c['crashed']} / {c['n']} | {tcs} | {rate} | {c['clean_power_w'] or '—'} | {c['board_ram_peak_mb'] or '—'} | {', '.join(c['forensic_patterns']) or '—'} | {c['status'].upper()} |")
    L += ["", f"A cell FAILS when its crash rate exceeds {summary['max_crash_rate']:.0%}. 'RAM before crash' is the pattern in the last 30 s of each crashed run: "
          "plateau (steady, like a healthy run) or cycling (repeated large swings, typically an allocation that keeps failing)."]
    return "\n".join(L) + "\n"


def junit(summary):
    import xml.etree.ElementTree as ET
    cells = summary["cells"]
    ts = ET.Element("testsuite", name="edgegate-matrix", tests=str(len(cells)),
                    failures=str(sum(c["status"] == "fail" for c in cells)), skipped=str(sum(c["status"] == "skip" for c in cells)),
                    errors="0", timestamp=summary["created_at"].rstrip("Z"))
    for c in cells:
        tc = ET.SubElement(ts, "testcase", classname="edgegate.matrix", name=c["name"])
        msg = f"{c['crashed']} of {c['n']} runs crashed (limit {summary['max_crash_rate']:.0%})"
        if c["status"] == "fail":
            ET.SubElement(tc, "failure", message=msg, type="crash_rate").text = msg
        elif c["status"] == "skip":
            ET.SubElement(tc, "skipped", message="no measured runs")
        props = ET.SubElement(tc, "properties")
        for k in ("crashed", "n", "crash_rate", "clean_power_w", "board_ram_peak_mb"):
            if c.get(k) is not None:
                ET.SubElement(props, "property", name=k, value=str(c[k]))
    return ET.tostring(ts, encoding="unicode", xml_declaration=True)
