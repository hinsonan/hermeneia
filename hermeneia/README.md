# Hermeneia - Divine Word Transcription & Translation

## Installation & Running

### Quick Start
```bash
# Install dependencies
npm install

# Run in development mode
npm run dev:tauri
```

### GPU Support

#### Automatic Detection (Recommended)

The app automatically detects your GPU and applies optimizations:

- **Windows**: GPU acceleration works automatically
- **macOS**: Metal GPU acceleration enabled by default
- **Linux**: 
  - Detects NVIDIA GPUs automatically
  - Applies PRIME offload for hybrid GPU laptops
  - No configuration needed!

#### Manual NVIDIA Mode (Linux)

If you experience rendering issues on Linux with NVIDIA:
```bash
# Development with forced NVIDIA settings
npm run dev:nvidia
```

Or run the built app with:
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./hermeneia
```

#### Troubleshooting

**Black screen or rendering issues on Linux:**
- The app automatically applies `WEBKIT_DISABLE_DMABUF_RENDERER=1` for NVIDIA GPUs
- If issues persist, try running with `npm run dev:nvidia`

**GPU not detected:**
- Ensure `lspci` is installed: `sudo apt install pciutils`
- Check if your NVIDIA drivers are installed: `nvidia-smi`

## Development
```bash
# Frontend only (no Tauri)
npm run dev

# Full app with Tauri
npm run dev:tauri

# Build for production
npm run build:tauri
```

## Building for Distribution
```bash
# Build optimized binary
npm run build:tauri

# Output location
src-tauri/target/release/bundle/
```

The built app includes automatic GPU detection