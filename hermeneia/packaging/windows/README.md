# Windows CUDA Bootstrap Installer

This directory contains the NSIS bootstrap installer used for large Windows CUDA releases.

- `bootstrap-installer.nsi` builds a small per-user installer EXE.
- `bootstrap-install.ps1` performs runtime download, checksum verification, extraction, and app install.
- `third_party/7z/` is populated in CI before NSIS compilation and should contain `7z.exe` + `7z.dll`.

The bootstrap installer expects release assets named like:

- `hermeneia_<version>_windows_x64_cuda_portable.7z` or multipart `.7z.001`, `.7z.002`, ...
- `hermeneia_<version>_windows_x64_cuda_portable.sha256`

Install target is per-user:

- `%LocalAppData%\Programs\Hermeneia-CUDA`
