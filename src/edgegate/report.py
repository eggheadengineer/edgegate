"""JUnit XML and Markdown output."""
import datetime as dt
import xml.etree.ElementTree as ET


def junit(results, checks, suite="edgegate"):
    ts = ET.Element("testsuite", name=suite, tests=str(len(checks)),
                    failures=str(sum(c.status == "fail" for c in checks)),
                    skipped=str(sum(c.status == "skip" for c in checks)), errors="0",
                    timestamp=results["created_at"].rstrip("Z"), hostname=results["device"].get("hostname", ""))
    for c in checks:
        tc = ET.SubElement(ts, "testcase", classname=f"{suite}.{results['device']['name']}", name=c.name)
        if c.status == "fail":
            ET.SubElement(tc, "failure", message=c.message, type="threshold").text = c.message
        elif c.status == "skip":
            ET.SubElement(tc, "skipped", message=c.message)
        props = ET.SubElement(tc, "properties")
        for k, v in (("value", c.value), ("limit", c.limit), ("baseline", c.baseline)):
            if v is not None:
                ET.SubElement(props, "property", name=k, value=str(v))
    return ET.tostring(ET.ElementTree(ts).getroot(), encoding="unicode", xml_declaration=True)


def markdown(results, checks, baseline=None):
    r = results
    lines = [f"# edgegate: {r['model']['path']} on {r['device']['name']}", ""]
    lines.append(f"**{status_word(checks)}** · {r['model']['backend']} {r['model'].get('version') or ''} · "
                 f"{r['model'].get('precision', '')} · {r['created_at']}")
    dev = r["device"]
    extra = [dev.get(k) for k in ("model", "l4t", "power_mode") if dev.get(k)]
    if extra:
        lines.append("Device: " + " · ".join(extra))
    lines += ["", "| Metric | Value | Baseline |", "| --- | --- | --- |"]
    b = baseline or {}
    lat, blat = r["latency_ms"], b.get("latency_ms", {})
    for k in ("p50", "p90", "p99", "mean"):
        if lat.get(k) is not None:
            lines.append(f"| latency {k} | {lat[k]} ms | {blat.get(k, '—')} ms |" if blat.get(k) is not None else f"| latency {k} | {lat[k]} ms | — |")
    if r.get("throughput_ips"):
        lines.append(f"| throughput | {r['throughput_ips']} inf/s | {b.get('throughput_ips', '—')} |")
    mem = r["memory"]
    lines.append(f"| peak RSS | {mem.get('peak_rss_mb')} MB | {b.get('memory', {}).get('peak_rss_mb', '—')} |")
    if mem.get("board_ram_peak_mb") is not None:
        lines.append(f"| board RAM peak | {mem['board_ram_peak_mb']} MB | {b.get('memory', {}).get('board_ram_peak_mb', '—')} |")
    if r.get("power"):
        p = r["power"]
        lines.append(f"| power total ({p['total_rail']}) | {p['total_w']} W | {b.get('power', {}).get('total_w', '—') if b.get('power') else '—'} |")
        for rail, w in p["rails_w"].items():
            lines.append(f"| power {rail} | {w} W | {b.get('power', {}).get('rails_w', {}).get(rail, '—') if b.get('power') else '—'} |")
    if r.get("accuracy"):
        a = r["accuracy"]
        lines.append(f"| accuracy {a['metric']} (n={a['n']}) | {a['value']} | {b['accuracy']['value'] if b.get('accuracy') else '—'} |")
    if r.get("command"):
        c, bc = r["command"], b.get("command") or {}
        lines.append(f"| exit code | {c['exit_code']}{' (full window)' if c.get('timed_out') else ''} | {bc.get('exit_code', '—')} |")
        lines.append(f"| wall time | {c['wall_s']} s | {bc.get('wall_s', '—')} |")
    for name, m in (r.get("metrics") or {}).items():
        bm = (b.get("metrics") or {}).get(name) or {}
        per_s = f" ({m['per_s']}/s)" if m.get("per_s") is not None else ""
        lines.append(f"| {name} | {m.get('count', '—')}{per_s} | {bm.get('count', '—')} |")
    lines += ["", "## Checks", "", "| Check | Status | Detail |", "| --- | --- | --- |"]
    for c in checks:
        lines.append(f"| {c.name} | {c.status.upper()} | {c.message} |")
    if r.get("notes"):
        lines += ["", *[f"- {n}" for n in r["notes"]]]
    return "\n".join(lines) + "\n"


def status_word(checks):
    return "FAIL" if any(c.status == "fail" for c in checks) else "PASS"
