## Download

Pick the right file for your system:

| You have | Download |
|----------|----------|
| **Linux** + NVIDIA GPU | `hermeneia_*-cuda.deb`, `*-cuda.AppImage`, and `*-cuda.rpm` or `*-cuda.rpm.xz` |
| **Linux**, no NVIDIA | `hermeneia_*-cpu.deb`, `*-cpu.rpm`, or `*-cpu.AppImage` |
| **Windows** + NVIDIA GPU | `hermeneia_*_windows_x64_cuda_installer.exe` (recommended), fallback: `hermeneia_*_windows_x64_cuda_portable.7z` or split `...7z.001`, `...7z.002`, ... |
| **Windows**, no NVIDIA | `hermeneia_*-cpu.exe` |
| **Mac** (Apple Silicon M1-M4) | `hermeneia_*_aarch64.dmg` |
| **Mac** (Intel) | `hermeneia_*_x64.dmg` |

> **CUDA variants** bundle CUDA, ONNX Runtime, and cuDNN libraries, so they are much larger. You still need compatible NVIDIA drivers installed on your system.

## NVIDIA GPU Support

The **CUDA** variants accelerate transcription, translation, and speaker diarization on NVIDIA GPUs. GPU acceleration is split across two backends with different minimum hardware requirements:

| Feature | Minimum GPU |
|---------|-------------|
| Full GPU acceleration (transcription, translation, **and diarization**) | **Turing** — compute capability 7.5 (e.g. RTX 20-series, GTX 16-series, Quadro RTX/T-series, Tesla T4) |
| GPU transcription/translation only | **Pascal** — compute capability 6.1 (e.g. GTX 10-series, Quadro P-series) |

- On Pascal and Volta GPUs, diarization must run on the CPU; Tesla P100 / Quadro GP100 (compute capability 6.0) are not supported.
- **Newest supported:** Blackwell (RTX 50-series, B100/B200). Newer architectures run via forward-compatible PTX JIT.
- **Driver requirement:** NVIDIA driver >= 525 (Linux) / ~527 (Windows) for CUDA 12.x.
- The **CPU** variants do not require a GPU.

## Installation Notes

**macOS:** The app is not code-signed. macOS will block it on first launch.
To fix, run in Terminal:
```
xattr -cr /Applications/Hermeneia.app
```
Or: right-click the app, then click **Open**.

**Windows:** You may see a SmartScreen warning ("Windows protected your PC").
Click **"More info"** then **"Run anyway"**.

**Windows CUDA (recommended for pre-releases and releases):**
- Download and run `hermeneia_*_windows_x64_cuda_installer.exe`.
- The installer downloads and verifies CUDA package parts automatically, installs per-user, creates shortcuts, and launches the app.

**Windows CUDA (draft releases):**
- Draft release assets are not publicly downloadable by URL, so the bootstrap installer may fail to fetch package parts.
- For draft testing, download the portable archive assets directly from the release page and use the manual fallback steps below.

**Windows CUDA (manual fallback):**
- If the release includes multiple files like `.7z.001`, `.7z.002`, download all parts into the same folder.
- Open only the first part with 7-Zip and extract it.
- Run `hermeneia.exe` from the extracted folder.
- Microsoft Edge WebView2 Runtime must be installed on the target machine.

**Linux CUDA RPM:**
- If you see `*.rpm.xz`, decompress it first with `xz -d file.rpm.xz`.
- If you see split files like `*.rpm.xz.part-000`, combine them first:
```
cat file.rpm.xz.part-* > file.rpm.xz
xz -d file.rpm.xz
```

**Linux .AppImage:** Make it executable first:
```
chmod +x hermeneia_*.AppImage
./hermeneia_*.AppImage
```
