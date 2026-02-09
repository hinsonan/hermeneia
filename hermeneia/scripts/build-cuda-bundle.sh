#!/bin/bash
# Build Hermeneia with CUDA as .deb, .rpm, .AppImage
# CUDA libraries are bundled inside each package — end users only need NVIDIA drivers
set -e

# Ensure CUDA libs are extracted (one-time setup)
if [ ! -d "src-tauri/cuda-libs" ] || [ -z "$(ls -A src-tauri/cuda-libs 2>/dev/null)" ]; then
    echo "CUDA libraries not found. Extracting from Docker image..."
    cd src-tauri && docker-compose -f docker-compose.cuda.yml run --rm extract-cuda-libs && cd ..
    echo ""
fi

# Build the Docker image (bundler stage with Node.js), then run the bundle
echo "Building CUDA bundles (.deb, .rpm, .AppImage)..."
echo ""
cd src-tauri && \
    docker-compose -f docker-compose.cuda.yml build bundle-cuda && \
    docker-compose -f docker-compose.cuda.yml run --rm bundle-cuda && \
    cd ..

echo ""
echo "Bundles are at: src-tauri/cuda-bundles/"
ls -lh src-tauri/cuda-bundles/deb/ 2>/dev/null
ls -lh src-tauri/cuda-bundles/rpm/ 2>/dev/null
ls -lh src-tauri/cuda-bundles/appimage/ 2>/dev/null
