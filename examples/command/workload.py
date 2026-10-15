#!/usr/bin/env python3
"""Stand-in for an inference engine replaying a file: emits 'detect' events at --rate per second until killed."""
import argparse, pathlib, time
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--rate", type=float, default=40)
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "events.log", "w", buffering=1) as f:
    i = 0
    while True:
        f.write(f"detect {i} bearing={(i * 7) % 360}\n"); i += 1
        time.sleep(1.0 / a.rate)
