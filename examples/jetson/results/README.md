# Orin NX 16 GB results: ResNet-50 v2, fp16, TensorRT 10.3, JetPack 6.2.2

Measured 2026-10-15 with `edgegate run -c examples/jetson/edgegate.yaml` (500 iterations, 10 s window, tegrastats at
200 ms, DVFS left alone, no `jetson_clocks`). One JSON per nvpmodel mode; use any of them as a `--baseline`.

| nvpmodel | p50 ms | p99 ms | inf/s | VDD_IN W | power samples |
| --- | --- | --- | --- | --- | --- |
| MAXN_SUPER | 2.32 | 2.33 | 440 | 18.5 | 54 |
| 10W | 6.26 | 6.28 | 161 | 7.8 | 57 |
| 15W | 6.25 | 6.27 | 162 | 8.0 | 54 |
| 25W | 5.90 | 5.92 | 172 | 8.2 | 56 |
| 40W | 2.32 | 2.33 | 440 | 18.3 | 54 |

Two observations worth knowing before you trust a number:

- The first run used trtexec's default 3 s window and read 6.6 W at MAXN_SUPER. With a 10 s window the same mode reads
  18.5 W. That is why `benchmark.duration_s` exists and why the example sets it to 10.
- 10W and 15W change the CPU core count and `nvpmodel -m` reboots the board to apply them. 25W, 40W and MAXN_SUPER
  switch live. Give the board a minute after boot before measuring.
