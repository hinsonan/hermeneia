## Download

Pick the right file for your system:

| You have | Download |
|----------|----------|
| **Linux** + NVIDIA GPU | `hermeneia_*-cuda.deb`, `*-cuda.AppImage`, and `*-cuda.rpm` or `*-cuda.rpm.xz` |
| **Linux**, no NVIDIA | `hermeneia_*-cpu.deb`, `*-cpu.rpm`, or `*-cpu.AppImage` |
| **Windows** + NVIDIA GPU | `hermeneia_*_x64-cuda-portable.7z` or split `...7z.001`, `...7z.002`, ... |
| **Windows**, no NVIDIA | `hermeneia_*-cpu.exe` |
| **Mac** (Apple Silicon M1-M4) | `hermeneia_*_aarch64.dmg` |
| **Mac** (Intel) | `hermeneia_*_x64.dmg` |

> **CUDA variants** bundle CUDA, ONNX Runtime, and cuDNN libraries, so they are much larger. You still need compatible NVIDIA drivers installed on your system.

## Installation Notes

**macOS:** The app is not code-signed. macOS will block it on first launch.
To fix, run in Terminal:
```
xattr -cr /Applications/Hermeneia.app
```
Or: right-click the app, then click **Open**.

**Windows:** You may see a SmartScreen warning ("Windows protected your PC").
Click **"More info"** then **"Run anyway"**.

**Windows CUDA portable build:**
- If the release includes multiple files like `.7z.001`, `.7z.002`, download all parts into the same folder.
- Open the first part with 7-Zip and extract it.
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
