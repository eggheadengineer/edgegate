# Running edgegate on a Jetson

Written before the hardware was on the bench, so treat the exact commands as the plan and adjust to what the
board actually has. Target: Orin NX on JetPack 6.2.x (TensorRT 10, CUDA 12). Xavier NX on JetPack 5 works the
same way with TensorRT 8.5.

## 1. Install

```
ssh jetson@192.168.55.1
python3 -m venv ~/.venvs/edgegate && . ~/.venvs/edgegate/bin/activate
pip install git+https://github.com/eggheadengineer/edgegate
which trtexec || ls /usr/src/tensorrt/bin/trtexec     # edgegate finds either
tegrastats --interval 500 & sleep 2; kill %1           # confirm rail names (VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0 on Orin)
```

For accuracy on the device you also need onnxruntime with CUDA. NVIDIA publishes Jetson wheels (search
"onnxruntime jetson wheel" for the JetPack version). Without it, latency, memory and power still run and the
accuracy check reports as skipped.

## 2. First run

```
git clone https://github.com/eggheadengineer/edgegate && cd edgegate
./examples/jetson/fetch_model.sh
sudo nvpmodel -q                                        # note the current mode
edgegate run -c examples/jetson/edgegate.yaml -o results/orin-nx-$(date +%Y%m%d)-modeA.json --check \
    --junit results/junit.xml --markdown results/report.md
```

The first run builds the TensorRT engine (minutes). It is cached next to the model under `.edgegate-engines/`
keyed by model hash, precision and input shape, so later runs start immediately.

## 3. The table worth having

Latency and power across power modes is the slide. On Orin NX:

```
for mode in 0 1 2 3; do
  sudo nvpmodel -m $mode && sudo jetson_clocks
  edgegate run -c examples/jetson/edgegate.yaml -o results/orin-nx-mode$mode.json --note "nvpmodel $mode"
done
```

Then compare any two:

```
edgegate check -c examples/jetson/edgegate.yaml --results results/orin-nx-mode3.json --baseline results/orin-nx-mode0.json --markdown results/mode3-vs-mode0.md
```

`nvpmodel -m` numbering differs between boards; `nvpmodel -q --verbose` lists them. The results file records
the mode name under `device.power_mode` so the numbers stay attributable.

## 4. Regression gating in CI

Commit a baseline per device (`results/baseline-orin-nx.json`). A self-hosted GitHub Actions runner on the
Jetson, or a `cron` job that pulls and runs, then:

```
edgegate run -c edgegate.yaml -o results/latest.json --check --baseline results/baseline-orin-nx.json --junit results/junit.xml
```

Exit 2 fails the pipeline. The JUnit file plugs into anything that reads JUnit, including reqthread, where
each check can be traced to a performance requirement.

## 5. Cross-device

Run the same config on the Xavier NX with `device: { name: xavier-nx }`. Comparing a Xavier result against an
Orin baseline is allowed, but edgegate adds a `baseline_device` skip note so nobody mistakes a hardware
difference for a regression.

## Things to confirm on first contact with the board

- `trtexec --help` prints a `TensorRT v10.x` line; the parser reads it for the results file.
- `tegrastats` rail names match the three Orin rails above; if not, the parser still sums whatever rails it finds.
- `nvpmodel -q` output contains "NV Power Mode"; otherwise `device.power_mode` is simply absent.
- The ResNet-50 input is named `data`; `trtexec --shapes` needs that name.
