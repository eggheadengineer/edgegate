"""Command-line interface."""
import argparse
import json
import pathlib
import sys

from . import __version__, config
from .backends import BackendError
from .compare import evaluate, status
from .dataset import DatasetError
from .measure import run as measure_run
from .report import junit, markdown

EXIT_OK, EXIT_ERROR, EXIT_FAIL = 0, 1, 2


def _err(msg):
    print(msg, file=sys.stderr)


def _write(path, text):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _load_json(path, what):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise config.ConfigError(f"{what} {path}: {e}") from None


def _emit(a, results, checks, baseline):
    for c in checks:
        print(f"{c.status.upper():5} {c.name}: {c.message}")
    print(f"{status(checks).upper()}: {sum(c.status == 'fail' for c in checks)} failing check(s), "
          f"{sum(c.status == 'skip' for c in checks)} skipped")
    if a.junit:
        _write(a.junit, junit(results, checks)); print(f"wrote {a.junit}")
    if a.markdown:
        _write(a.markdown, markdown(results, checks, baseline)); print(f"wrote {a.markdown}")
    return EXIT_FAIL if status(checks) == "fail" else EXIT_OK


def cmd_run(a):
    cfg = config.load(a.config)
    results = measure_run(cfg, notes=a.note)
    _write(a.out, json.dumps(results, indent=1) + "\n")
    print(f"wrote {a.out}")
    lat = results["latency_ms"]
    line = f"latency p50 {lat['p50']} ms, p99 {lat['p99']} ms, mean {lat['mean']} ms"
    if results.get("power"):
        line += f" · power {results['power']['total_w']} W"
    if results.get("accuracy"):
        line += f" · {results['accuracy']['metric']} {results['accuracy']['value']}"
    print(line)
    for n in results.get("notes", []):
        print(f"note: {n}")
    if a.check:
        baseline = _load_json(a.baseline, "baseline") if a.baseline else None
        return _emit(a, results, evaluate(results, cfg["thresholds"], baseline), baseline)
    return EXIT_OK


def cmd_check(a):
    cfg = config.load(a.config)
    results = _load_json(a.results, "results")
    baseline = _load_json(a.baseline, "baseline") if a.baseline else None
    return _emit(a, results, evaluate(results, cfg["thresholds"], baseline), baseline)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="edgegate", description="CI gate for model-plus-firmware builds on edge devices.")
    ap.add_argument("--version", action="version", version=f"edgegate {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="measure latency, memory, power and accuracy on this device")
    p.add_argument("-c", "--config", default="edgegate.yaml")
    p.add_argument("-o", "--out", default="results/latest.json")
    p.add_argument("--note", help="free-text note stored with the results (e.g. firmware build id)")
    p.add_argument("--check", action="store_true", help="also evaluate thresholds (and --baseline if given); exit 2 on failure")
    p.add_argument("--baseline", help="results JSON to compare against")
    p.add_argument("--junit", help="write JUnit XML for the checks")
    p.add_argument("--markdown", help="write a Markdown summary")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check", help="evaluate an existing results file against thresholds and a baseline")
    p.add_argument("-c", "--config", default="edgegate.yaml")
    p.add_argument("--results", required=True)
    p.add_argument("--baseline")
    p.add_argument("--junit")
    p.add_argument("--markdown")
    p.set_defaults(func=cmd_check)

    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except (config.ConfigError, BackendError, DatasetError, RuntimeError) as e:
        _err(f"error: {e}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
