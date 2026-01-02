# Hermeneia

**Private, Offline Audio Transcription & Translation**

Transcribe and translate sermons, teachings, and documents entirely on your computer. No cloud services, no proprietary APIs, no large corporation scraping your data, you own your data.

---

## 📁 Project Structure

```
Hermeneia/
├── LICENSE              # GNU Affero General Public License v3.0
└── hermeneia/          # Main application folder
    ├── src/            # Frontend (SolidJS + TypeScript)
    ├── src-tauri/      # Backend (Rust + Tauri 2)
    ├── package.json    # Node dependencies & scripts
    └── README.md       # Installation & usage guide
```

### What is the `hermeneia/` folder?

The `hermeneia/` directory contains the complete application:
- **Frontend**: SolidJS-based UI for audio editing, transcription, and translation
- **Backend**: Rust-powered audio and AI processing with GPU acceleration
- **Privacy-first**: All processing happens locally—Windows and Linux builds available

See `hermeneia/README.md` for installation and usage instructions.

---

## 🗺️ Development Roadmap (MVP)

### ✅ Phase 1: Audio Processing (Complete)
Complete high-performance audio processing engine with streaming playback
- Multi-format audio support (MP3, FLAC, WAV, OGG, AAC)
- Real-time waveform visualization
- Audio playback with full controls (play, pause, seek)
- fast audio trimming
- GPU optimization for Linux NVIDIA
- Complete audio editor UI with dark mode

### ☐ Phase 2: Transcription
Convert sermon audio to text using local AI models
- Local speech-to-text engine with GPU acceleration
- Interactive text editor synced with audio timeline
- Speaker identification and labeling
- Word-level timestamps for precise editing
- Export to multiple formats (TXT, SRT, VTT, JSON, DOCX, PDF)

### ☐ Phase 3: Translation
Translate transcriptions to multiple languages offline
- Local translation model integration
- Side-by-side editor for source and translation
- English ↔ Greek as primary language pair
- Custom theological terminology glossary
- Translation memory for consistency
- Review tools with revision history

### ☐ Phase 4: Deployment & Distribution
Package and distribute application for Windows and Linux
- Windows installer (.exe, .msi)
- Linux packages (.deb, .AppImage, .rpm)
- Code signing for both platforms
- Automatic update system
- CI/CD pipeline for releases
- Complete user documentation   

---


## 🚀 Getting Started

See `hermeneia/README.md` for:
- Installation instructions
- Development setup
- Building for production
- GPU configuration
- Troubleshooting

Quick start:
```bash
cd hermeneia
npm install
npm run dev:tauri
```

---

## 📄 License

GNU Affero General Public License v3.0
