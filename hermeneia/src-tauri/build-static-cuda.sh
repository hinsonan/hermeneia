#!/bin/bash
# Build Hermeneia with bundled CUDA 12.8 libraries
# Output: transcribe binary + lib/ folder with CUDA .so files
# Only requires NVIDIA drivers on host, NOT full CUDA toolkit

set -e

echo "🔨 Building Hermeneia with bundled CUDA 12.8 libraries..."
echo ""

# Build using docker-compose
docker-compose -f docker-compose.cuda.yml build build-cuda-static
docker-compose -f docker-compose.cuda.yml run --rm build-cuda-static

echo ""
echo "✅ Build complete!"
echo ""
echo "📦 Distribution package location: ./dist/"
echo "   Binary: ./dist/transcribe"
echo "   CUDA libs: ./dist/lib/*.so"
echo ""
echo "📏 Package size:"
du -sh ./dist
echo ""
echo "🔍 RPATH configuration:"
patchelf --print-rpath ./dist/transcribe || echo "   (patchelf not installed on host)"
echo ""
echo "🚀 To run on ANY machine with NVIDIA drivers:"
echo "   cd dist && ./transcribe --input audio.mp3"
echo ""
echo "   OR copy the entire dist/ folder and run from inside it"
echo ""
echo "✨ Requirements on target machine:"
echo "   - NVIDIA drivers (any recent version with CUDA 12.x support)"
echo "   - NO CUDA toolkit installation needed!"
echo "   - Keep transcribe and lib/ folder together"
echo ""
