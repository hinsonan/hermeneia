#!/bin/bash
# Development script for running Tauri app with CUDA
# Starts vite in background, runs the CUDA binary, cleans up on exit

set -e

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $VITE_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start vite in background
npm run dev &
VITE_PID=$!

# Wait for vite to be ready
echo "Waiting for vite..."
sleep 2

# Run the CUDA binary
# cuda-libs: CUDA runtime libs (libcudart, libcublas, etc.)
# target-cuda/debug: sherpa-onnx shared libs (libsherpa-onnx-c-api.so, etc.)
echo "Starting Tauri app with CUDA..."
LD_LIBRARY_PATH=./src-tauri/cuda-libs:./src-tauri/target-cuda/debug \
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
WEBKIT_DISABLE_DMABUF_RENDERER=1 \
./src-tauri/target-cuda/debug/hermeneia

cleanup
