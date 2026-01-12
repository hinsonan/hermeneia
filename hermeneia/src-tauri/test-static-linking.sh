#!/bin/bash
# Test if true static CUDA linking works with CUDA 12.8 / Blackwell

set -e

echo "🧪 Testing TRUE static CUDA linking..."
echo "⚠️  This is experimental and may fail with CUDA 12.8"
echo ""

# Build using the test Dockerfile (target builder stage only)
docker build -f Dockerfile.cuda-static-test --target builder -t hermeneia-static-test:12.8 .

echo ""
echo "✅ Build completed!"
echo ""
echo "📦 Extracting binary..."
# Create a container from the builder stage image
CONTAINER_ID=$(docker create hermeneia-static-test:12.8)
docker cp $CONTAINER_ID:/dist/transcribe ./transcribe-static
docker rm $CONTAINER_ID > /dev/null

echo ""
echo "✅ Binary extracted to: ./transcribe-static"
echo ""
echo "📊 Binary info:"
ls -lh ./transcribe-static
echo ""
echo "🔍 Dependencies (should only show libcuda.so.1 for driver):"
ldd ./transcribe-static 2>&1 | grep -E "cuda|nvidia" || echo "   ✓ No CUDA toolkit dependencies found!"
echo ""
