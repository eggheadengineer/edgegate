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
from . import matrix as matrix_mod

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
    out_dir = pathlib.Path(a.out).with_suffix(".d")
    results = measure_run(cfg, notes=a.note, out_dir=out_dir)
    _write(a.out, json.dumps(results, indent=1) + "\n")
    print(f"wrote {a.out}")
    lat = results["latency_ms"]
    if results.get("command"):
        c = results["command"]
        line = f"exit {c['exit_code']}{' (full window)' if c['timed_out'] else ''} after {c['wall_s']} s"
        for name, m in results.get("metrics", {}).items():
            line += f" · {name} {m['count']}" + (f" ({m['per_s']}/s)" if m.get("per_s") is not None else "")
    else:
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
        checks = evaluate(results, cfg["thresholds"], baseline)
        # record the verdict in the results file so a later run can tell whether this one is fit to be a baseline
        results["gate"] = {"status": status(checks), "failed": [c.name for c in checks if c.status == "fail"],
                           "checked": [c.name for c in checks if c.status != "skip"], "baseline": a.baseline}
        _write(a.out, json.dumps(results, indent=1) + "\n")
        return _emit(a, results, checks, baseline)
    return EXIT_OK


def cmd_check(a):
    cfg = config.load(a.config)
    results = _load_json(a.results, "results")
    baseline = _load_json(a.baseline, "baseline") if a.baseline else None
    return _emit(a, results, evaluate(results, cfg["thresholds"], baseline), baseline)


def cmd_matrix(a):
    m = matrix_mod.load_matrix(a.config)
    summary = matrix_mod.run_matrix(m, a.out, dry_run=a.dry_run)
    if a.dry_run:
        return EXIT_OK
    md = matrix_mod.markdown(summary)
    _write(pathlib.Path(a.out) / "summary.md", md)
    print(md)
    if a.junit:
        _write(a.junit, matrix_mod.junit(summary)); print(f"wrote {a.junit}")
    if a.markdown:
        _write(a.markdown, md); print(f"wrote {a.markdown}")
    print(f"wrote {pathlib.Path(a.out) / 'summary.json'}")
    return EXIT_FAIL if any(c["status"] == "fail" for c in summary["cells"]) else EXIT_OK


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

    p = sub.add_parser("matrix", help="run variants x repeats (interleaved, resumable) and tabulate crash rate per cell")
    p.add_argument("-c", "--config", default="matrix.yaml", help="matrix file (see edgegate.matrix docstring)")
    p.add_argument("-o", "--out", default="results/matrix", help="results directory; existing r<N>.json files are skipped")
    p.add_argument("--junit", help="one JUnit testcase per cell")
    p.add_argument("--markdown", help="copy of the summary table")
    p.add_argument("--dry-run", action="store_true", help="print the command line of every pending run and exit")
    p.set_defaults(func=cmd_matrix)

    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except (config.ConfigError, BackendError, DatasetError, RuntimeError, matrix_mod.MatrixError) as e:
        _err(f"error: {e}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
