# edgegate

A CI gate for model-plus-firmware builds on edge devices. It runs a model on the real target, measures
latency (p50/p90/p99), memory, power, and accuracy, compares the numbers to a stored baseline, and fails
the build on regression. Output is a JSON results file, a JUnit report, and a Markdown summary.

Built for Jetson (TensorRT through `trtexec`, power through `tegrastats`), but the host backend runs the
same gate anywhere with ONNX Runtime, so the pipeline can be developed and tested without the board.

```
pip install "edgegate[host] @ git+https://github.com/eggheadengineer/edgegate"
python examples/host/make_sample.py
edgegate run -c examples/host/edgegate.yaml --check --junit results/junit.xml --markdown results/report.md
```

That generates a tiny model and dataset, benchmarks it, checks accuracy, and evaluates the thresholds in the
config. Mislabel the dataset (`make_sample.py --wrong 20`) and the accuracy gate fails with exit code 2.

## What it measures

| Measurement | Host (ONNX Runtime) | Jetson (trtexec) |
| --- | --- | --- |
| Latency p50 / p90 / p99 / mean | per-iteration timing | trtexec summary (median, mean, p99) |
| Throughput | derived | trtexec qps |
| Memory | peak RSS of the process | board RAM peak from tegrastats |
| Power | tegrastats if present | per-rail and board total from tegrastats |
| Accuracy (top-1) | yes | needs onnxruntime on the device; otherwise skipped, not failed |

Results record the model hash, backend and version, precision, device name, L4T release and `nvpmodel`
power mode, so a number is never separated from what produced it.

## Thresholds

Absolute limits and regressions against a baseline, all optional:

```yaml
thresholds:
  latency_p99_ms_max: 15
  memory_peak_mb_max: 1500
  power_w_max: 20
  accuracy_min: 0.75
  latency_p99_regression_pct: 10
  memory_regression_pct: 10
  power_regression_pct: 10
  accuracy_drop_max: 0.005
```

A percentage regression only counts if the absolute change also clears a floor (defaults: 0.5 ms latency,
16 MB memory, 0.25 W power; override with `latency_regression_floor_ms` and friends). Without that, a
15-microsecond model "regresses" 150% from scheduler jitter. A measurement that could not be taken skips its
check rather than failing it, and a baseline recorded on a different device is flagged so a hardware
difference is not mistaken for a regression.

## Commands

| Command | Does | Exit |
| --- | --- | --- |
| `edgegate run` | measure on this device; with `--check` also evaluate thresholds and `--baseline` | 0 pass · 1 error · 2 gate failed |
| `edgegate check` | evaluate an existing results file against thresholds and a baseline | same |

Both write `--junit` and `--markdown` if asked. The JUnit output plugs into anything that reads JUnit,
including [reqthread](https://github.com/eggheadengineer/reqthread), where each check can be traced to a
performance requirement.

## On a Jetson

See `docs/JETSON.md`. Short version: `./examples/jetson/fetch_model.sh`, then
`edgegate run -c examples/jetson/edgegate.yaml`. The TensorRT engine is built once and cached by model hash,
precision and input shape. Looping over `nvpmodel` modes gives a latency-versus-power table per device.

## Tests

```
pip install -e ".[dev]"
python -m unittest discover -s tests
```

The suite builds a tiny ONNX model on the fly, so it needs no downloads and no hardware. The trtexec and
tegrastats parsers are tested against captured output formats for Orin and Xavier.

## License

Apache-2.0. See `LICENSE`.
