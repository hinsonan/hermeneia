#!/bin/sh
# Removes the bundled libcuda.so stub from the AppImage.
# libcuda.so.1 is the NVIDIA *driver* library — it must come from the host
# system, not from the CUDA toolkit. During the Docker build we symlink
# the CUDA stub so linuxdeploy can resolve the dependency, but if it gets
# bundled the AppImage will fail at runtime with CUDA_ERROR_STUB_LIBRARY.

set -e

BUNDLE_DIR="${1:-/project/src-tauri/target/release/bundle/appimage}"

APPIMAGE=$(find "$BUNDLE_DIR" -maxdepth 1 -name '*.AppImage' ! -name 'linuxdeploy*' | head -1)

if [ -z "$APPIMAGE" ]; then
  echo "No AppImage found in $BUNDLE_DIR — skipping."
  exit 0
fi

echo "Found AppImage: $APPIMAGE"

APPDIR=$(dirname "$APPIMAGE")
APPNAME=$(basename "$APPIMAGE")

cd "$APPDIR"

echo "Extracting AppImage..."
"$APPDIR/$APPNAME" --appimage-extract

echo "Removing libcuda.so stub from squashfs-root/usr/lib/..."
rm -f squashfs-root/usr/lib/libcuda.so*

echo "Repacking AppImage..."
ARCH=x86_64 appimagetool squashfs-root "$APPNAME"

rm -rf squashfs-root

echo "Done — libcuda.so.1 removed from $APPNAME"
