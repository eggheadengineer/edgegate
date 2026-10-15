#!/usr/bin/env bash
# Fetch a public ImageNet classifier (ResNet-50 v2, ONNX model zoo, ~100 MB) for the Jetson example.
set -euo pipefail
cd "$(dirname "$0")"
URL="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
[ -f resnet50-v2-7.onnx ] || curl -L --fail -o resnet50-v2-7.onnx "$URL"
echo "model: $(du -h resnet50-v2-7.onnx | cut -f1) resnet50-v2-7.onnx (input 'data', 1x3x224x224, NCHW, ImageNet mean/std)"
