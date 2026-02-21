## Download

Pick the right file for your system:

| You have | Download |
|----------|----------|
| **Linux** + NVIDIA GPU | `hermeneia_*-cuda.deb`, `*-cuda.rpm`, or `*-cuda.AppImage` |
| **Linux**, no NVIDIA | `hermeneia_*-cpu.deb`, `*-cpu.rpm`, or `*-cpu.AppImage` |
| **Windows** + NVIDIA GPU | `hermeneia_*-cuda.exe` |
| **Windows**, no NVIDIA | `hermeneia_*-cpu.exe` |
| **Mac** (Apple Silicon M1-M4) | `hermeneia_*_aarch64.dmg` |
| **Mac** (Intel) | `hermeneia_*_x64.dmg` |

> **CUDA variants** bundle the CUDA runtime (~500 MB larger). You still need NVIDIA drivers installed on your system.

## Installation Notes

**macOS:** The app is not code-signed. macOS will block it on first launch.
To fix, run in Terminal:
```
xattr -cr /Applications/Hermeneia.app
```
Or: right-click the app, then click **Open**.

**Windows:** You may see a SmartScreen warning ("Windows protected your PC").
Click **"More info"** then **"Run anyway"**.

**Linux .AppImage:** Make it executable first:
```
chmod +x hermeneia_*.AppImage
./hermeneia_*.AppImage
```
