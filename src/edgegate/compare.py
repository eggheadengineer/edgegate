"""Turn a results document, optional baseline, and thresholds into pass/fail checks."""
from dataclasses import dataclass, asdict


@dataclass
class Check:
    name: str
    status: str          # pass | fail | skip
    value: object = None
    limit: object = None
    baseline: object = None
    message: str = ""

    def as_dict(self):
        return asdict(self)


# A percentage regression on a tiny absolute value is noise (a 15 µs model "regresses" 150% from
# scheduler jitter). Regressions must also exceed an absolute floor to count. Overridable per metric.
FLOORS = {"latency_p99_regression_pct": ("latency_regression_floor_ms", 0.5),
          "memory_regression_pct": ("memory_regression_floor_mb", 16.0),
          "power_regression_pct": ("power_regression_floor_w", 0.25)}


def _pct_change(cur, base):
    if base in (None, 0) or cur is None:
        return None
    return (cur - base) / base * 100.0


def baseline_problem(baseline, thresholds):
    """Why this baseline must not anchor regression checks, or None. A run that failed its own gate, or a
    command run that exited outside ok_exit_codes, is not a reference: comparing against it produces verdicts
    like 'power regressed 66%' when the baseline simply died early."""
    if not baseline or not (thresholds or {}).get("baseline_must_pass", True):
        return None
    gate = baseline.get("gate")
    if gate and gate.get("status") == "fail":
        return "baseline failed its own gate: " + ", ".join(gate.get("failed", [])[:4])
    cmd = baseline.get("command")
    ok = (thresholds or {}).get("ok_exit_codes")
    if cmd is not None and ok is not None and cmd.get("exit_code") not in ok:
        return f"baseline process exited {cmd.get('exit_code')}, not in accepted {ok}"
    return None


def evaluate(results, thresholds, baseline=None):
    checks = []
    t = thresholds or {}
    bad_baseline = baseline_problem(baseline, t)
    if bad_baseline:
        checks.append(Check("baseline_valid", "fail", None, None, baseline.get("created_at"), message=bad_baseline + "; regression checks skipped"))
        baseline = None
    lat = results["latency_ms"].get("p99")
    mem = results["memory"].get("peak_rss_mb")
    board_mem = results["memory"].get("board_ram_peak_mb")
    pw = results["power"]["total_w"] if results.get("power") else None
    acc = results["accuracy"]["value"] if results.get("accuracy") else None
    b = baseline or {}
    b_lat = b.get("latency_ms", {}).get("p99") if b else None
    b_mem = b.get("memory", {}).get("peak_rss_mb") if b else None
    b_pw = b["power"]["total_w"] if b and b.get("power") else None
    b_acc = b["accuracy"]["value"] if b and b.get("accuracy") else None

    def absolute(name, value, limit, higher_is_bad=True, unit=""):
        if limit is None:
            return
        if value is None:
            checks.append(Check(name, "skip", None, limit, message="not measured")); return
        ok = value <= limit if higher_is_bad else value >= limit
        word = "within" if ok else ("exceeds" if higher_is_bad else "below")
        checks.append(Check(name, "pass" if ok else "fail", value, limit,
                            message=f"{value}{unit} {word} limit {limit}{unit}"))

    def regression(name, value, base, pct_limit, unit=""):
        if pct_limit is None:
            return
        if value is None or base is None:
            checks.append(Check(name, "skip", value, pct_limit, base,
                                message="no baseline" if base is None else "not measured")); return
        change = _pct_change(value, base)
        floor_key, floor_default = FLOORS[name]
        floor = t.get(floor_key, floor_default)
        over_pct = change > pct_limit
        over_floor = (value - base) > floor
        ok = not (over_pct and over_floor)
        note = f" (under the {floor}{unit} floor)" if over_pct and not over_floor else ""
        checks.append(Check(name, "pass" if ok else "fail", value, pct_limit, base,
                            message=f"{value}{unit} vs baseline {base}{unit} ({change:+.1f}%, limit +{pct_limit}%){note}"))

    absolute("latency_p99_ms_max", lat, t.get("latency_p99_ms_max"), unit=" ms")
    absolute("memory_peak_mb_max", board_mem if board_mem is not None else mem, t.get("memory_peak_mb_max"), unit=" MB")
    absolute("power_w_max", pw, t.get("power_w_max"), unit=" W")
    absolute("accuracy_min", acc, t.get("accuracy_min"), higher_is_bad=False)

    regression("latency_p99_regression_pct", lat, b_lat, t.get("latency_p99_regression_pct"), unit=" ms")
    regression("memory_regression_pct", mem, b_mem, t.get("memory_regression_pct"), unit=" MB")
    regression("power_regression_pct", pw, b_pw, t.get("power_regression_pct"), unit=" W")

    drop = t.get("accuracy_drop_max")
    if drop is not None:
        if acc is None or b_acc is None:
            checks.append(Check("accuracy_drop_max", "skip", acc, drop, b_acc,
                                message="no baseline" if b_acc is None else "not measured"))
        else:
            d = b_acc - acc
            checks.append(Check("accuracy_drop_max", "pass" if d <= drop else "fail", acc, drop, b_acc,
                                message=f"{acc} vs baseline {b_acc} (drop {d:+.4f}, limit {drop})"))

    cmd = results.get("command")
    ok_codes = t.get("ok_exit_codes")
    if cmd is not None and ok_codes is not None:
        rc = cmd["exit_code"]
        checks.append(Check("exit_code", "pass" if rc in ok_codes else "fail", rc, ok_codes,
                            message=f"exit {rc}{' after the full window' if cmd.get('timed_out') else ''}, accepted {ok_codes}"))
    metrics = results.get("metrics") or {}
    b_metrics = (b.get("metrics") or {}) if b else {}
    for name, limit in (t.get("metric_min") or {}).items():
        absolute(f"metric_min:{name}", (metrics.get(name) or {}).get("count"), limit, higher_is_bad=False)
    for name, pct_limit in (t.get("metric_regression_pct") or {}).items():
        cur = (metrics.get(name) or {}).get("count")
        base = (b_metrics.get(name) or {}).get("count")
        if cur is None or base is None:
            checks.append(Check(f"metric_regression_pct:{name}", "skip", cur, pct_limit, base,
                                message="no baseline" if base is None else "not measured")); continue
        drop_pct = (base - cur) / base * 100.0 if base else 0.0           # positive when the count fell
        floor = (t.get("metric_regression_floor") or {}).get(name, 0)
        ok = not (drop_pct > pct_limit and (base - cur) > floor)
        checks.append(Check(f"metric_regression_pct:{name}", "pass" if ok else "fail", cur, pct_limit, base,
                            message=f"{cur} vs baseline {base} ({-drop_pct:+.1f}%, limit -{pct_limit}%)"))

    if baseline and baseline.get("model", {}).get("sha256") == results["model"]["sha256"] \
            and baseline.get("device", {}).get("name") != results["device"]["name"]:
        checks.append(Check("baseline_device", "skip", results["device"]["name"], None, baseline["device"].get("name"),
                            message="baseline was recorded on a different device; regression checks compare across devices"))
    return checks


def status(checks):
    return "fail" if any(c.status == "fail" for c in checks) else "pass"
