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
echo "Starting Tauri app with CUDA..."
LD_LIBRARY_PATH=./src-tauri/cuda-libs \
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
WEBKIT_DISABLE_DMABUF_RENDERER=1 \
./src-tauri/target-cuda/debug/hermeneia

cleanup
