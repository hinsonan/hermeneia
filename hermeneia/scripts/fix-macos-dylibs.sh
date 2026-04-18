#!/usr/bin/env bash
set -euo pipefail

app_path="${1:?usage: fix-macos-dylibs.sh <path-to-app-bundle>}"
frameworks_dir="$app_path/Contents/Frameworks"
macos_dir="$app_path/Contents/MacOS"

required_libs=(
  "libsherpa-onnx-c-api.dylib"
  "libsherpa-onnx-cxx-api.dylib"
  "libonnxruntime.dylib"
  "libonnxruntime.1.17.1.dylib"
)

is_macho_file() {
  local target="$1"
  file "$target" | grep -q "Mach-O"
}

ensure_frameworks_present() {
  local lib

  test -d "$frameworks_dir"
  for lib in "${required_libs[@]}"; do
    test -f "$frameworks_dir/$lib"
  done
}

ensure_rpath() {
  local target="$1"
  local rpath="$2"

  if ! otool -l "$target" | grep -q "$rpath"; then
    install_name_tool -add_rpath "$rpath" "$target"
  fi
}

rewrite_deps_to_rpath() {
  local target="$1"
  local dep
  local base
  local changed=0

  while IFS= read -r dep; do
    [ -n "$dep" ] || continue
    base="$(basename "$dep")"

    case "$base" in
      libsherpa-onnx-c-api.dylib|libsherpa-onnx-cxx-api.dylib|libonnxruntime.dylib|libonnxruntime.1.17.1.dylib)
        if [[ "$dep" != "@rpath/$base" ]]; then
          install_name_tool -change "$dep" "@rpath/$base" "$target"
          changed=1
        fi
        ;;
    esac
  done < <(otool -L "$target" | awk 'NR > 1 {print $1}')

  return "$changed"
}

echo "[macos-fixup] Validating required bundled dylibs"
ensure_frameworks_present

echo "[macos-fixup] Normalizing dylib install ids and internal links"
for lib in "${required_libs[@]}"; do
  current_id="$(otool -D "$frameworks_dir/$lib" | awk 'NR==2 {print $1}')"
  if [[ "$current_id" != "@rpath/$lib" ]]; then
    install_name_tool -id "@rpath/$lib" "$frameworks_dir/$lib"
  fi
  rewrite_deps_to_rpath "$frameworks_dir/$lib" || true
done

echo "[macos-fixup] Rewriting app executable and sidecar links"
shopt -s nullglob
for bin in "$macos_dir"/*; do
  [ -f "$bin" ] || continue
  if is_macho_file "$bin"; then
    ensure_rpath "$bin" "@executable_path/../Frameworks"
    rewrite_deps_to_rpath "$bin" || true
  fi
done

echo "[macos-fixup] Re-signing app bundle (ad-hoc)"
codesign --force --deep --sign - "$app_path"

echo "[macos-fixup] Done"
