# Hermeneia - Divine Word Transcription & Translation

A desktop application for audio transcription and translation, built with Tauri 2, SolidJS, and Rust. Hermeneia uses OpenAI Whisper for speech-to-text and MarianMT/MADLAD-400 for neural machine translation across 38 languages, with optional GPU acceleration via CUDA or Metal.

## Prerequisites

### All Platforms
- [Node.js](https://nodejs.org/) >= 16
- [Rust](https://rustup.rs/) >= 1.70 (via rustup)
- Git

### Linux (tested)
```bash
sudo apt install pkg-config build-essential libssl-dev \
  libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev \
  librsvg2-dev libasound2-dev pciutils
```

### macOS (untested)
```bash
xcode-select --install
```

### Windows (untested)
Install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/) with the "Desktop development with C++" workload and Windows SDK.

> **Note:** Only Linux builds have been verified so far. The codebase uses cross-platform dependencies and all platform-specific code is properly gated, so macOS and Windows builds should work in principle but have not been tested.

---

## Quick Start

```bash
# Install frontend dependencies
npm install

# Run the full app in development mode (Vite frontend + Tauri backend)
npm run dev:tauri
```

The app window opens at 1200x800. The Vite dev server runs on port 1420 with HMR on 1421. Frontend changes hot-reload; Rust changes require a restart.

---

## Development

### Standard Development

```bash
# Full app with Tauri (recommended)
npm run dev:tauri

# Full app with debug logging
npm run dev:tauri:debug

# Frontend only (no Rust backend)
npm run dev
```

### NVIDIA GPU on Linux

The app auto-detects NVIDIA GPUs and applies PRIME offload for hybrid laptops. If you experience rendering issues:

```bash
npm run dev:nvidia
```

This sets `__NV_PRIME_RENDER_OFFLOAD=1`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, and `WEBKIT_DISABLE_DMABUF_RENDERER=1`.

### Running Tests

```bash
cd src-tauri && cargo test

# Run a specific test
cd src-tauri && cargo test test_name
```

---

## Building for Production

### CPU-Only Build

```bash
npm run build:tauri
```

Output location: `src-tauri/target/release/bundle/`

Produces platform-specific installers (builds for the current platform only):
- **Linux** (tested): `.deb`, `.rpm`, `.AppImage`
- **macOS** (untested): `.dmg`
- **Windows** (untested): `.msi`

### Optimized CPU Build

```bash
npm run build:cpu
npm run dev:cpu    # run it
```

The release profile uses `opt-level = 3`, fat LTO, and single codegen unit for maximum runtime performance.

---

## CUDA Builds (Docker)

CUDA builds compile inside Docker so you don't need a local CUDA toolkit -- only NVIDIA drivers on the host machine. The build targets CUDA 12.8 with PTX compilation for compute capability 7.5+, which covers RTX 20-series through 50-series and datacenter GPUs (A100, H100).

### Initial Setup (one-time)

Extract CUDA runtime libraries from the official NVIDIA Docker image:

```bash
cd src-tauri && docker-compose -f docker-compose.cuda.yml run --rm extract-cuda-libs
```

This copies `libcudart`, `libcublas`, `libcublasLt`, `libnvrtc`, and `libcurand` shared objects into `src-tauri/cuda-libs/`.

### Development with CUDA

```bash
# Build the Rust backend with CUDA inside Docker
npm run build:cuda

# Run the app with CUDA acceleration (starts Vite + the CUDA binary)
npm run dev:cuda
```

`build:cuda` mounts your source files into the container and uses persistent Cargo cache volumes, so incremental rebuilds are fast. The output binary lands in `src-tauri/target-cuda/debug/hermeneia`.

`dev:cuda` runs `scripts/dev-cuda.sh`, which starts the Vite frontend, then launches the CUDA binary with `LD_LIBRARY_PATH` pointing to the extracted CUDA libraries.

### Building CUDA Bundles (.deb, .rpm, .AppImage)

```bash
npm run build:cuda:bundle
```

This runs `tauri build --features cuda` inside Docker and produces Linux packages with CUDA libraries bundled inside. Output goes to `src-tauri/cuda-bundles/`:

```
cuda-bundles/
  deb/hermeneia_0.1.0_amd64.deb
  rpm/hermeneia-0.1.0-1.x86_64.rpm
  appimage/hermeneia_0.1.0_amd64.AppImage
```

Each package includes the CUDA runtime libraries (`libcudart`, `libcublas`, `libcublasLt`, `libnvrtc`, `libcurand`). The binary's RPATH is set at link time so it finds the bundled libs automatically. End users only need NVIDIA drivers -- no CUDA toolkit required.

The build uses a separate Docker target directory (`cuda-bundle-target` volume) so it doesn't interfere with CPU builds.

### Building for Distribution (Standalone CLI with Bundled CUDA)

For a portable standalone CLI binary (not a full Tauri app package):

```bash
cd src-tauri && ./build-static-cuda.sh
```

This produces a portable distribution package in `src-tauri/dist/`:

```
dist/
  transcribe          # Release binary with RPATH set to $ORIGIN/lib
  lib/
    libcudart.so.12
    libcublas.so.12
    libcublasLt.so.12
    libnvrtc.so.12
    libcurand.so.12
```

The binary uses `patchelf` to set `RPATH=$ORIGIN/lib`, so it finds the bundled CUDA libraries relative to itself.

### Docker Services Reference

The `src-tauri/docker-compose.cuda.yml` defines these services:

| Service | Purpose | Command |
|---------|---------|---------|
| `extract-cuda-libs` | One-time extraction of CUDA 12.8 runtime libs | `docker-compose -f docker-compose.cuda.yml run --rm extract-cuda-libs` |
| `build-dev` | Incremental dev builds with Cargo caching | `docker-compose -f docker-compose.cuda.yml run --rm build-dev` |
| `build-cuda-static` | Standalone CLI with bundled CUDA libs | `docker-compose -f docker-compose.cuda.yml run --rm build-cuda-static` |
| `bundle-cuda` | Full app bundles (.deb/.rpm/.AppImage) with CUDA | `docker-compose -f docker-compose.cuda.yml run --rm bundle-cuda` |

---

## CLI Binary Tools

Hermeneia includes standalone CLI tools that can be run independently of the GUI. All use `clap` for argument parsing and `tracing` for logging.

### transcribe

Transcribe or translate audio files using OpenAI Whisper models.

```bash
cd src-tauri

# Basic transcription
cargo run --bin transcribe -- --input sermon.mp3

# Choose a larger model for better accuracy
cargo run --bin transcribe -- --input sermon.mp3 --model small

# Translate audio to English
cargo run --bin transcribe -- --input sermon.mp3 --task translate

# Output as JSON or SRT subtitles
cargo run --bin transcribe -- --input sermon.mp3 --format json --output result.json
cargo run --bin transcribe -- --input sermon.mp3 --format srt --output result.srt

# Include timestamps, specify language, force CPU
cargo run --bin transcribe -- --input sermon.mp3 --language en --timestamps --cpu

# Check system compatibility without running
cargo run --bin transcribe -- --input sermon.mp3 --check-only --strict
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input <FILE>` | required | Input audio file (MP3, WAV, FLAC, OGG, M4A, AAC) |
| `--output <FILE>` | stdout | Output file path |
| `--model <MODEL>` | `tiny` | Whisper model: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`, `large-v2`, `large-v3` |
| `--task <TASK>` | `transcribe` | `transcribe` or `translate` (translate converts audio to English text) |
| `--language <CODE>` | auto-detect | ISO 639-1 language code (e.g. `en`, `es`, `fr`) |
| `--format <FORMAT>` | `text` | Output format: `text`, `json`, `srt` |
| `--timestamps` | off | Include timestamps in output |
| `--cpu` | off | Force CPU inference (disable GPU) |
| `--check-only` | off | Validate system compatibility without transcribing |
| `--strict` | off | Treat compatibility warnings as errors |
| `--force` | off | Run despite compatibility warnings/errors |

To build with CUDA:
```bash
cargo run --features cuda --bin transcribe -- --input sermon.mp3 --model large-v3
```

### translate

Translate text using neural machine translation (MarianMT and MADLAD-400 models).

```bash
cd src-tauri

# Translate inline text
cargo run --bin translate -- --text "Hello world" --source en --target es

# Translate a file
cargo run --bin translate -- --input-file document.txt --source en --target fr --output result.txt

# Use a specific model
cargo run --bin translate -- --text "Hello" --source en --target de --model madlad-3b

# List available models
cargo run --bin translate -- --list-models

# List only models already downloaded
cargo run --bin translate -- --list-models --cached-only
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--text <TEXT>` | | Inline text to translate (mutually exclusive with `--input-file`) |
| `--input-file <FILE>` | | Text or SRT file to translate |
| `--output <FILE>` | stdout | Output file path |
| `--source <CODE>` | `en` | Source language (ISO 639-1) |
| `--target <CODE>` | `es` | Target language (ISO 639-1) |
| `--model <NAME>` | auto | Model name (auto-selects best for language pair) |
| `--max-length <N>` | `512` | Maximum translation length in tokens |
| `--cpu` | off | Force CPU inference |
| `--no-progress` | off | Disable progress bar |
| `--list-models` | off | List available models from catalog |
| `--cached-only` | off | Only show cached models when listing |

Supports 38 languages and 40+ MarianMT language pair models plus MADLAD-400 multilingual models (3B, 7B, 10B).

### audio-trim

Trim audio files to a specific time range.

```bash
cd src-tauri

cargo run --bin audio-trim -- --input sermon.mp3 --output trimmed.wav --start 5.0 --end 60.0

# Verbose mode shows audio metadata and processing time
cargo run --bin audio-trim -- --input sermon.mp3 --output trimmed.wav --start 0 --end 30 --verbose
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input <FILE>` | required | Input audio file |
| `--output <FILE>` | required | Output WAV file |
| `--start <SECONDS>` | required | Start time in seconds |
| `--end <SECONDS>` | required | End time in seconds |
| `--verbose` | off | Show detailed info |

### generate-test-audio

Generate synthetic test audio files for benchmarking. Requires `ffmpeg` in PATH for MP3/FLAC output.

```bash
cd src-tauri

# Generate a 2-minute stereo WAV test tone
cargo run --bin generate-test-audio -- --output test.wav --duration 120

# Generate an MP3 at 192 kbps
cargo run --bin generate-test-audio -- --output test.mp3 --duration 60 --mp3-bitrate 192

# Custom sample rate and mono
cargo run --bin generate-test-audio -- --output test.wav --sample-rate 48000 --channels 1
```

### Profiling Tools

For benchmarking audio trimming performance:

```bash
cd src-tauri

# Benchmark trim over multiple iterations
cargo run --bin profile-trim -- --input large.mp3 --output out.wav --iterations 10

# Detailed step-by-step timing for WAV trimming
cargo run --bin profile-trim-detailed -- --input test.wav --output out.wav

# Detailed timing for compressed audio (MP3/FLAC) trimming
cargo run --bin profile-compressed-detailed -- --input song.mp3 --output out.wav
```

---

## GPU Support

### Automatic Detection

The app detects your GPU at startup and applies optimizations automatically:

- **Linux**: Detects NVIDIA GPUs via `lspci`, applies PRIME offload for hybrid laptops
- **macOS**: Metal GPU acceleration via Candle's `metal` feature
- **Windows**: GPU acceleration works automatically

### Cargo Feature Flags

```bash
# CPU only (default)
cargo build

# NVIDIA CUDA
cargo build --features cuda

# Apple Metal
cargo build --features metal
```

### Troubleshooting

**Black screen or rendering issues on Linux:**
- The app automatically sets `WEBKIT_DISABLE_DMABUF_RENDERER=1` for NVIDIA GPUs
- If issues persist, try `npm run dev:nvidia`

**GPU not detected on Linux:**
- Ensure `lspci` is available: `sudo apt install pciutils`
- Check NVIDIA driver status: `nvidia-smi`

---

## Project Structure

```
hermeneia/
├── src/                           # Frontend (SolidJS + TypeScript)
│   ├── pages/
│   │   ├── Home.tsx               # Main menu
│   │   ├── AudioEditor.tsx        # Waveform editor & trimming
│   │   ├── Transcription.tsx      # Whisper transcription UI
│   │   └── Translation.tsx        # Translation UI
│   ├── components/
│   │   ├── FileUploader.tsx       # Audio file selection
│   │   ├── WaveformEditor.tsx     # Waveform visualization
│   │   ├── GreekScrollLoader.tsx  # Loading animation
│   │   └── ...
│   ├── types/                     # TypeScript type definitions
│   ├── utils/                     # Theme toggle, time formatting
│   └── styles/                    # CSS variables (parchment theme)
│
├── src-tauri/                     # Backend (Rust + Tauri 2)
│   ├── src/
│   │   ├── lib.rs                 # Tauri commands & shared state
│   │   ├── main.rs                # Binary entry point
│   │   ├── gpu.rs                 # GPU detection & optimization
│   │   ├── error.rs               # Centralized error types
│   │   ├── audio/                 # Audio pipeline (decode, encode, trim, playback, waveform)
│   │   ├── transcribe/            # Whisper transcription (model, inference, preprocessing)
│   │   ├── translate/             # Translation (MarianMT, MADLAD, tokenization, generation)
│   │   └── bin/                   # CLI tools (transcribe, translate, audio-trim, profilers)
│   ├── Cargo.toml                 # Rust dependencies & feature flags
│   ├── docker-compose.cuda.yml    # CUDA build services
│   ├── Dockerfile.cuda-static     # Multi-stage CUDA build image
│   ├── build-static-cuda.sh       # Standalone CLI distribution packaging
│   └── tauri.cuda.conf.json       # Bundle config overlay for CUDA builds
│
├── scripts/
│   ├── dev-cuda.sh                # CUDA development runner
│   └── build-cuda-bundle.sh       # CUDA bundle build orchestrator
├── package.json                   # NPM scripts & frontend deps
└── vite.config.ts                 # Vite config (port 1420)
```

---

## NPM Scripts Reference

| Script | Description |
|--------|-------------|
| `npm run dev` | Frontend-only Vite dev server |
| `npm run dev:tauri` | Full app (Vite + Tauri backend) |
| `npm run dev:tauri:debug` | Full app with `RUST_LOG=debug` |
| `npm run dev:nvidia` | Dev with forced NVIDIA PRIME settings (Linux) |
| `npm run dev:cuda` | Run pre-built CUDA binary with Vite frontend |
| `npm run dev:cpu` | Run optimized CPU release build |
| `npm run build:tauri` | Production build (creates platform installers) |
| `npm run build:cuda` | Build Rust backend with CUDA inside Docker (debug) |
| `npm run build:cuda:release` | Build Rust backend with CUDA (release, all optimizations) |
| `npm run build:cuda:bundle` | Build .deb/.rpm/.AppImage with bundled CUDA libs |
| `npm run build:cpu` | Optimized CPU-only release build |
