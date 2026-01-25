# Translation CLI Guide

This guide explains how to use the `translate` binary tool for command-line text translation.

## Build

### Standard Build (CPU)
To build the CLI tool for CPU usage:
```bash
cd src-tauri
cargo build --bin translate
```

### CUDA Build (NVIDIA GPU)
To build with CUDA support, you must use the project's Docker-based build process:
```bash
# From the project root
npm run build:cuda
```
This will compile the binary in a CUDA-enabled container and extract the necessary libraries to `src-tauri/cuda-libs/`.

## Usage

### Basic Translation
Translate text from English to Spanish (default):
```bash
# CPU Version
./target/debug/translate --text "Hello, my name is Andrew"

# CUDA Version (Requires library path)
LD_LIBRARY_PATH=./cuda-libs ./target-cuda/debug/translate --text "Hello, my name is Andrew" --source en --target es
```

### Specifying Languages
Translate from French to English:
```bash
./target/debug/translate --text "Bonjour tout le monde" --source fr --target en
```

### Using a Specific Model
List available models first:
```bash
./target/debug/translate --list-models
```

Translate using a specific model (e.g., MADLAD-400 3B):
```bash
./target/debug/translate --text "Hello" --model madlad-3b --target de
```

### File Input/Output
Translate a text file and save the result:
```bash
./target/debug/translate --input-file input.txt --output result.txt --source en --target fr
```

## Commands & Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--text` | `-t` | Input text to translate |
| `--input-file` | `-i` | Path to a text file to translate |
| `--output` | `-o` | Save translation to this file |
| `--source` | | Source language code (e.g., `en`, `es`, `fr`) |
| `--target` | | Target language code (e.g., `en`, `es`, `fr`) |
| `--model` | `-m` | Model name (e.g., `madlad-3b`, `marian-en-es`) |
| `--cpu` | | Force CPU usage even if GPU is available |
| `--max-length`| | Max tokens to generate (default: 512) |
| `--list-models`| | Show all available and cached models |
| `--no-progress`| | Hide the progress bar |

## Supported Languages
Commonly supported codes include:
- `en` (English)
- `es` (Spanish)
- `fr` (French)
- `de` (German)
- `it` (Italian)
- `pt` (Portuguese)
- `zh` (Chinese)
- `ja` (Japanese)
- `ko` (Korean)
- `ru` (Russian)
- `ar` (Arabic)
- `hi` (Hindi)
- `tr` (Turkish)
- `vi` (Vietnamese)

For a full list of models and their pairs, run `translate --list-models`.
