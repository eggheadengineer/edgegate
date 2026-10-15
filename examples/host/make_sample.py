#!/usr/bin/env python3
"""Generate a tiny classifier and a labeled dataset so the host example runs in seconds with no downloads.

The model is y = x @ I (identity), so top-1 accuracy on a correctly labeled set is 1.0. Pass --wrong N to
mislabel N samples and watch the accuracy gate fail. Needs: pip install onnx numpy
"""
import argparse
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument("--classes", type=int, default=8)
ap.add_argument("--samples", type=int, default=200)
ap.add_argument("--wrong", type=int, default=0, help="number of samples to mislabel")
a = ap.parse_args()
here = pathlib.Path(__file__).parent

w = numpy_helper.from_array(np.eye(a.classes, dtype=np.float32), name="W")
g = helper.make_graph([helper.make_node("MatMul", ["x", "W"], ["y"])], "tiny",
                      [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, a.classes])],
                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, a.classes])], initializer=[w])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
onnx.save(m, str(here / "tiny.onnx"))

rng = np.random.default_rng(42)
x = rng.standard_normal((a.samples, a.classes)).astype(np.float32)
y = x.argmax(axis=1)
y[:a.wrong] = (y[:a.wrong] + 1) % a.classes
np.savez(here / "sample.npz", x=x, y=y)
print(f"wrote {here/'tiny.onnx'} and {here/'sample.npz'} ({a.samples} samples, {a.wrong} mislabeled)")
