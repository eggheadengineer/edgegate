"""Hermetic tests: a tiny ONNX model built on the fly, an npz dataset, no hardware."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np

from edgegate import compare, config, measure, report, sampler
from edgegate.backends import BackendError, parse_trtexec_output

ROOT = pathlib.Path(__file__).resolve().parent.parent


def make_tiny_model(path, n=4):
    """y = x @ W with W = identity: argmax(y) == argmax(x). Deterministic, instant."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.eye(n, dtype=np.float32), name="W")
    node = helper.make_node("MatMul", ["x", "W"], ["y"])
    graph = helper.make_graph([node], "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, n])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, n])], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def make_dataset(path, n=4, samples=50, wrong=0):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((samples, n)).astype(np.float32)
    y = x.argmax(axis=1)
    y[:wrong] = (y[:wrong] + 1) % n          # deliberately mislabel the first `wrong` samples
    np.savez(path, x=x, y=y)


class Project:
    def __init__(self, thresholds="", dataset=True, wrong=0):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        make_tiny_model(self.root / "tiny.onnx")
        if dataset:
            make_dataset(self.root / "sample.npz", wrong=wrong)
        lines = ["version: 1",
                 "model: { path: tiny.onnx, backend: onnxruntime, input: { shape: [1, 4] } }"]
        if dataset:
            lines.append("dataset: { type: npz, path: sample.npz }")
        lines += ["benchmark: { warmup: 5, iterations: 50 }", "power: { sampler: none }", "device: { name: test-host }"]
        if thresholds:
            lines += ["thresholds:"] + ["  " + l for l in thresholds.splitlines()]
        else:
            lines.append("thresholds: {}")
        (self.root / "edgegate.yaml").write_text("\n".join(lines) + "\n")
        self.config = self.root / "edgegate.yaml"

    def cfg(self):
        return config.load(self.config)

    def cleanup(self):
        self.tmp.cleanup()


class Measure(unittest.TestCase):
    def test_run_produces_latency_memory_accuracy(self):
        p = Project()
        try:
            r = measure.run(p.cfg())
            self.assertEqual(r["latency_ms"]["n"], 50)
            self.assertGreater(r["latency_ms"]["p99"], 0)
            self.assertGreaterEqual(r["latency_ms"]["p99"], r["latency_ms"]["p50"])
            self.assertGreater(r["memory"]["peak_rss_mb"], 0)
            self.assertEqual(r["accuracy"], {"metric": "top1", "value": 1.0, "correct": 50, "n": 50})
            self.assertIsNone(r["power"])
            self.assertTrue(r["model"]["sha256"].startswith("sha256:"))
            self.assertEqual(r["device"]["name"], "test-host")
        finally:
            p.cleanup()

    def test_mislabeled_dataset_lowers_accuracy(self):
        p = Project(wrong=5)
        try:
            self.assertEqual(measure.run(p.cfg())["accuracy"]["value"], 0.9)
        finally:
            p.cleanup()

    def test_no_dataset_means_no_accuracy(self):
        p = Project(dataset=False)
        try:
            self.assertIsNone(measure.run(p.cfg())["accuracy"])
        finally:
            p.cleanup()


class Thresholds(unittest.TestCase):
    def results(self, **over):
        r = {"model": {"sha256": "sha256:abc"}, "device": {"name": "d"}, "latency_ms": {"p99": 10.0},
             "memory": {"peak_rss_mb": 500.0}, "power": {"total_w": 8.0, "rails_w": {}, "total_rail": "sum", "samples": 5},
             "accuracy": {"metric": "top1", "value": 0.95, "n": 100}, "created_at": "2026-10-15T00:00:00Z"}
        r.update(over)
        return r

    def test_absolute_limits(self):
        c = {x.name: x for x in compare.evaluate(self.results(), {"latency_p99_ms_max": 9, "memory_peak_mb_max": 600,
                                                                   "power_w_max": 8, "accuracy_min": 0.96})}
        self.assertEqual(c["latency_p99_ms_max"].status, "fail")
        self.assertEqual(c["memory_peak_mb_max"].status, "pass")
        self.assertEqual(c["power_w_max"].status, "pass")       # equal to the limit passes
        self.assertEqual(c["accuracy_min"].status, "fail")

    def test_regressions_against_baseline(self):
        base = self.results(latency_ms={"p99": 8.0}, memory={"peak_rss_mb": 500.0},
                            power={"total_w": 7.0, "rails_w": {}, "total_rail": "sum", "samples": 5},
                            accuracy={"metric": "top1", "value": 0.96, "n": 100})
        c = {x.name: x for x in compare.evaluate(self.results(), {"latency_p99_regression_pct": 10, "memory_regression_pct": 5,
                                                                   "power_regression_pct": 20, "accuracy_drop_max": 0.005}, base)}
        self.assertEqual(c["latency_p99_regression_pct"].status, "fail")   # +25%
        self.assertEqual(c["memory_regression_pct"].status, "pass")        # +0%
        self.assertEqual(c["power_regression_pct"].status, "pass")         # +14.3% < 20
        self.assertEqual(c["accuracy_drop_max"].status, "fail")            # drop 0.01 > 0.005

    def test_regression_under_absolute_floor_passes(self):
        base = self.results(latency_ms={"p99": 0.020})
        c = {x.name: x for x in compare.evaluate(self.results(latency_ms={"p99": 0.080}), {"latency_p99_regression_pct": 10}, base)}
        self.assertEqual(c["latency_p99_regression_pct"].status, "pass")      # +300% but only +0.06 ms
        self.assertIn("floor", c["latency_p99_regression_pct"].message)
        c = {x.name: x for x in compare.evaluate(self.results(latency_ms={"p99": 0.080}),
                                                 {"latency_p99_regression_pct": 10, "latency_regression_floor_ms": 0.01}, base)}
        self.assertEqual(c["latency_p99_regression_pct"].status, "fail")      # floor lowered: now it counts

    def test_missing_measurement_skips_not_fails(self):
        r = self.results(power=None, accuracy=None)
        c = {x.name: x for x in compare.evaluate(r, {"power_w_max": 5, "accuracy_min": 0.9, "latency_p99_regression_pct": 10})}
        self.assertEqual(c["power_w_max"].status, "skip")
        self.assertEqual(c["accuracy_min"].status, "skip")
        self.assertEqual(c["latency_p99_regression_pct"].status, "skip")   # no baseline
        self.assertEqual(compare.status(list(c.values())), "pass")

    def test_cross_device_baseline_is_flagged(self):
        base = self.results(device={"name": "orin-nx"})
        c = {x.name: x for x in compare.evaluate(self.results(device={"name": "xavier-nx"}), {"latency_p99_regression_pct": 10}, base)}
        self.assertIn("baseline_device", c)
        self.assertEqual(c["baseline_device"].status, "skip")


class Parsers(unittest.TestCase):
    def test_tegrastats_orin_line(self):
        line = ("10-15-2026 20:11:03 RAM 3012/7620MB (lfb 2x4MB) SWAP 0/3810MB (cached 0MB) CPU [12%@1510,8%@1510,0%@1510,0%@1510,0%@729,0%@729] "
                "GR3D_FREQ 45% VDD_GPU_SOC 2840mW/2700mW VDD_CPU_CV 1120mW/1000mW VIN_SYS_5V0 6810mW/6500mW")
        s = sampler.parse_tegrastats_line(line)
        self.assertEqual(s["ram_mb"], 3012)
        self.assertEqual(s["rails"], {"VDD_GPU_SOC": 2840, "VDD_CPU_CV": 1120, "VIN_SYS_5V0": 6810})

    def test_tegrastats_xavier_line_and_summary(self):
        lines = ["RAM 2000/7772MB VDD_IN 5200mW/5100mW VDD_CPU_GPU_CV 2000mW/1900mW VDD_SOC 1400mW/1350mW",
                 "RAM 2100/7772MB VDD_IN 5400mW/5200mW VDD_CPU_GPU_CV 2200mW/2000mW VDD_SOC 1400mW/1375mW"]
        summ = sampler.summarize([sampler.parse_tegrastats_line(l) for l in lines])
        self.assertEqual(summ["total_rail"], "VDD_IN")
        self.assertAlmostEqual(summ["total_w"], 5.3)
        self.assertAlmostEqual(summ["rails_w"]["VDD_CPU_GPU_CV"], 2.1)
        self.assertEqual(summ["ram_peak_mb"], 2100)
        self.assertEqual(summ["samples"], 2)

    def test_tegrastats_without_input_rail_sums(self):
        summ = sampler.summarize([sampler.parse_tegrastats_line("VDD_GPU_SOC 2000mW/2000mW VDD_CPU_CV 1000mW/1000mW")])
        self.assertEqual(summ["total_rail"], "sum")
        self.assertAlmostEqual(summ["total_w"], 3.0)

    def test_summarize_empty(self):
        self.assertIsNone(sampler.summarize([]))
        self.assertIsNone(sampler.summarize([{"rails": {}, "ram_mb": 100}]))

    def test_trtexec_output(self):
        out = textwrap.dedent("""\
            [10/15/2026-20:00:01] [I] === Performance summary ===
            [10/15/2026-20:00:01] [I] Throughput: 412.337 qps
            [10/15/2026-20:00:01] [I] Latency: min = 2.1123 ms, max = 3.9 ms, mean = 2.4201 ms, median = 2.4 ms, percentile(99%) = 3.1 ms
            [10/15/2026-20:00:01] [I] Enqueue Time: min = 0.5 ms, max = 1.2 ms, mean = 0.6 ms, median = 0.6 ms, percentile(99%) = 1.0 ms
            """)
        p = parse_trtexec_output(out)
        self.assertEqual(p, {"min": 2.1123, "max": 3.9, "mean": 2.4201, "median": 2.4, "p99": 3.1, "throughput_qps": 412.337})

    def test_trtexec_output_missing_summary(self):
        with self.assertRaises(BackendError):
            parse_trtexec_output("nothing useful here")


class Reports(unittest.TestCase):
    def test_junit_and_markdown(self):
        p = Project(thresholds="latency_p99_ms_max: 0.000001\naccuracy_min: 0.5\npower_w_max: 10")
        try:
            r = measure.run(p.cfg())
            checks = compare.evaluate(r, p.cfg()["thresholds"])
            x = report.junit(r, checks)
            self.assertIn('<testsuite name="edgegate" tests="3" failures="1" skipped="1"', x)
            self.assertIn('<testcase classname="edgegate.test-host" name="latency_p99_ms_max"', x)
            self.assertIn("<failure", x)
            md = report.markdown(r, checks)
            self.assertTrue(md.startswith("# edgegate: tiny.onnx on test-host"))
            self.assertIn("| latency_p99_ms_max | FAIL |", md)
            self.assertIn("| power_w_max | SKIP |", md)
        finally:
            p.cleanup()


class Config(unittest.TestCase):
    def check(self, text):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "edgegate.yaml"
            f.write_text(textwrap.dedent(text))
            return config.load(f)

    def test_rejects_bad_backend(self):
        with self.assertRaises(config.ConfigError):
            self.check("version: 1\nmodel: { path: m.onnx, backend: pytorch, input: { shape: [1, 3] } }\n")

    def test_rejects_bad_shape(self):
        with self.assertRaises(config.ConfigError):
            self.check("version: 1\nmodel: { path: m.onnx, backend: onnxruntime, input: { shape: [1, -3] } }\n")

    def test_rejects_non_numeric_threshold(self):
        with self.assertRaises(config.ConfigError):
            self.check("version: 1\nmodel: { path: m.onnx, backend: onnxruntime, input: { shape: [1, 3] } }\nthresholds: { accuracy_min: high }\n")

    def test_defaults_applied(self):
        cfg = self.check("version: 1\nmodel: { path: m.onnx, backend: trtexec, input: { shape: [1, 3, 224, 224] } }\n")
        self.assertEqual(cfg["benchmark"]["iterations"], 200)
        self.assertEqual(cfg["model"]["precision"], "fp16")
        self.assertEqual(cfg["power"]["sampler"], "auto")


class Cli(unittest.TestCase):
    def run_cli(self, *args, cwd):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        return subprocess.run([sys.executable, "-m", "edgegate.cli", *args], capture_output=True, text=True, env=env, cwd=cwd)

    def test_run_then_check_with_baseline_and_ratchet(self):
        p = Project(thresholds="latency_p99_regression_pct: 100000\naccuracy_drop_max: 0.0\naccuracy_min: 0.9")
        try:
            r = self.run_cli("run", "-o", "results/base.json", cwd=p.root)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((p.root / "results/base.json").is_file())
            r = self.run_cli("run", "-o", "results/new.json", "--check", "--baseline", "results/base.json",
                             "--junit", "results/junit.xml", "--markdown", "results/report.md", cwd=p.root)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("PASS: 0 failing", r.stdout)
            self.assertTrue((p.root / "results/junit.xml").is_file())
            # degrade: mislabel the dataset so accuracy drops below the baseline
            make_dataset(p.root / "sample.npz", wrong=10)
            r = self.run_cli("check", "--results", "results/new.json", "--baseline", "results/base.json", cwd=p.root)
            self.assertEqual(r.returncode, 0)   # stored results unchanged
            r = self.run_cli("run", "-o", "results/bad.json", "--check", "--baseline", "results/base.json", cwd=p.root)
            self.assertEqual(r.returncode, 2, r.stdout)
            self.assertIn("FAIL  accuracy_drop_max", r.stdout)
            self.assertIn("FAIL  accuracy_min", r.stdout)
        finally:
            p.cleanup()

    def test_missing_config_exits_1(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_cli("run", "-c", "nope.yaml", cwd=d)
            self.assertEqual(r.returncode, 1)
            self.assertIn("config not found", r.stderr)


if __name__ == "__main__":
    unittest.main()


class Command(unittest.TestCase):
    """The command backend: any process as the DUT, metrics counted from its output files."""

    def project(self, sleep_s, lines, duration_s, thresholds=""):
        d = pathlib.Path(tempfile.mkdtemp(prefix="edgegate-cmd-"))
        (d / "work.py").write_text(textwrap.dedent(f"""
            import sys, time, pathlib
            out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
            with open(out / "events.log", "w") as f:
                for i in range({lines}):
                    f.write(f"event {{i}} detected\\n")
            print("ready", flush=True)
            time.sleep({sleep_s})
        """))
        thr = "\n".join("  " + ln for ln in textwrap.dedent(thresholds).strip().splitlines()) if thresholds.strip() else "  {}"
        (d / "edgegate.yaml").write_text(textwrap.dedent(f"""
            version: 1
            model:
              backend: command
              path: {sys.executable}
              args: ["{d / 'work.py'}", "{{out}}"]
              env: {{ EDGEGATE_TEST: "1" }}
            benchmark: {{ duration_s: {duration_s} }}
            power: {{ sampler: none }}
            device: {{ name: host }}
            metrics:
              - {{ name: detections, file: "{{out}}/events.log", pattern: "^event .* detected$" }}
              - {{ name: missing, file: "{{out}}/nope.log", pattern: "x" }}
            thresholds:
            __THR__
        """).replace("__THR__", thr.replace("\n", "\n")))
        return d

    def test_counts_metrics_and_exit_code(self):
        d = self.project(sleep_s=0, lines=7, duration_s=30)
        r = measure.run(config.load(d / "edgegate.yaml"), out_dir=d / "out")
        self.assertEqual(r["command"]["exit_code"], 0)
        self.assertFalse(r["command"]["timed_out"])
        self.assertEqual(r["metrics"]["detections"]["count"], 7)
        self.assertIsNone(r["metrics"]["missing"]["count"])
        self.assertIsNone(r["latency_ms"]["p99"])
        self.assertGreater(r["memory"]["peak_rss_mb"], 0)
        self.assertTrue((d / "out" / "stdout.log").read_text().startswith("ready"))

    def test_full_window_is_exit_124(self):
        d = self.project(sleep_s=30, lines=3, duration_s=1)
        r = measure.run(config.load(d / "edgegate.yaml"), out_dir=d / "out")
        self.assertTrue(r["command"]["timed_out"])
        self.assertEqual(r["command"]["exit_code"], 124)
        self.assertLess(r["command"]["wall_s"], 15)
        self.assertEqual(r["metrics"]["detections"]["count"], 3)

    def test_checks_exit_code_and_metric_thresholds(self):
        d = self.project(sleep_s=0, lines=5, duration_s=30, thresholds="""
            ok_exit_codes: [124]
            metric_min: { detections: 10, missing: 1 }
            metric_regression_pct: { detections: 10 }
        """)
        cfg = config.load(d / "edgegate.yaml")
        r = measure.run(cfg, out_dir=d / "out")
        base = json.loads(json.dumps(r)); base["metrics"]["detections"]["count"] = 8
        by = {c.name: c for c in compare.evaluate(r, cfg["thresholds"], base)}
        self.assertEqual(by["exit_code"].status, "fail")                       # exited 0, only 124 accepted
        self.assertEqual(by["metric_min:detections"].status, "fail")           # 5 < 10
        self.assertEqual(by["metric_min:missing"].status, "skip")              # file absent: not measured
        self.assertEqual(by["metric_regression_pct:detections"].status, "fail")  # 5 vs 8 is -37.5%
        base["metrics"]["detections"]["count"] = 5
        by = {c.name: c for c in compare.evaluate(r, cfg["thresholds"], base)}
        self.assertEqual(by["metric_regression_pct:detections"].status, "pass")
        md = report.markdown(r, list(by.values()), base)
        self.assertIn("| detections | 5", md)
        self.assertIn("| exit code | 0", md)
        report.junit(r, list(by.values()))

    def test_config_rejects_bad_metrics_and_dataset(self):
        d = self.project(sleep_s=0, lines=1, duration_s=5)
        text = (d / "edgegate.yaml").read_text()
        (d / "bad.yaml").write_text(text.replace('pattern: "^event .* detected$"', 'pattern: "(unclosed"'))
        with self.assertRaises(config.ConfigError):
            config.load(d / "bad.yaml")
        (d / "bad2.yaml").write_text(text + "\ndataset: { type: npz, path: x.npz }\n")
        with self.assertRaises(config.ConfigError):
            config.load(d / "bad2.yaml")

    def test_cli_run_with_command_backend(self):
        d = self.project(sleep_s=0, lines=4, duration_s=30, thresholds="ok_exit_codes: [0]\nmetric_min: { detections: 1 }")
        p = subprocess.run([sys.executable, "-m", "edgegate.cli", "run", "-c", "edgegate.yaml", "-o", "r.json", "--check", "--markdown", "r.md"],
                           cwd=d, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("detections 4", p.stdout)
        self.assertTrue((d / "r.d" / "events.log").exists())

    def test_relative_args_resolve_against_config_dir(self):
        d = self.project(sleep_s=0, lines=2, duration_s=10)
        text = (d / "edgegate.yaml").read_text().replace(f'"{d / "work.py"}"', '"work.py"')
        (d / "edgegate.yaml").write_text(text)
        cfg = config.load(d / "edgegate.yaml")
        self.assertEqual(cfg["model"]["cwd"], str(d))
        r = measure.run(cfg, out_dir=d / "out")
        self.assertEqual(r["command"]["exit_code"], 0)
        self.assertEqual(r["metrics"]["detections"]["count"], 2)
