#!/usr/bin/env python3
"""Stand-in for an inference engine replaying a file: emits 'detect' events at --rate per second until killed.
--crash-after N makes it die with exit 3 after N seconds (to exercise crash-rate gating); --batch above 1
makes it crash early and emit nothing, mimicking a batch-size bug."""
import argparse, os, pathlib, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--rate", type=float, default=40)
ap.add_argument("--crash-after", type=float, default=float(os.environ.get("WORKLOAD_CRASH_AFTER", "0")))
ap.add_argument("--batch", type=int, default=1)
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.monotonic()
with open(out / "events.log", "w", buffering=1) as f:
    if a.batch > 1:
        for i in range(3):
            f.write("error batch allocation failed\n"); time.sleep(0.2)
        print("fatal: batch allocation failed", flush=True); sys.exit(3)
    i = 0
    while True:
        if a.crash_after and time.monotonic() - t0 >= a.crash_after:
            print("fatal: simulated crash", flush=True); sys.exit(3)
        f.write(f"detect {i} bearing={(i * 7) % 360}\n"); i += 1
        time.sleep(1.0 / a.rate)
