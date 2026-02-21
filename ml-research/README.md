# Hermeneia ML Research Tools

Companion tooling for [Hermeneia](../), a Tauri 2 desktop application that performs audio transcription and translation using on-device machine learning. This package provides Python utilities for curating and maintaining the Marian-MT model catalog that Hermeneia's Rust backend uses for translation.

## What it does

Hermeneia ships with a TOML catalog of Helsinki-NLP Marian machine-translation models. Before a model can be downloaded at runtime, we need to know whether a SafeTensors version is available on Hugging Face (SafeTensors loads faster and is safer than the legacy PyTorch format). This package automates that check:

1. Reads the model catalog (`models.toml`) containing model IDs and revisions.
2. Queries the Hugging Face API for each model to determine if a `model.safetensors` file exists.
3. Writes an updated catalog with a `has_safetensors` field on every entry.

The updated catalog is then consumed by the Rust backend to decide which weight format to fetch.

## Setup

1. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate hermeneia-ml-research
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

Update Marian SafeTensors metadata:

```bash
update-marian-safetensors \
  --input ../src-tauri/src/translate/models.toml \
  --output ../src-tauri/src/translate/models_updated.toml
```

## Tools

| Command | Description |
|---------|-------------|
| `update-marian-safetensors` | Checks every Marian model in the catalog for SafeTensors availability via the Hugging Face API and writes an updated TOML file. |

---

## Background Removal (`remove_bg.py`)

Removes the background from a PNG image using [rembg](https://github.com/danielgatis/rembg) (U2Net neural network). Unlike color-based approaches, this works correctly on subjects with complex or irregular edges — such as the Hermeneia stone tablet logo.

### Setup

Install `rembg` and its dependencies (already included in `pyproject.toml`):

```bash
pip install -e .
```

The U2Net model (~176 MB) is downloaded automatically on first run to `~/.u2net/u2net.onnx`.

### Usage

```bash
python src/hermeneia_ml_research/remove_bg.py <input.png> [output.png]
```

Examples:

```bash
# Output saved as logo_nobg.png alongside the input
python src/hermeneia_ml_research/remove_bg.py ~/Downloads/logo.png

# Explicit output path
python src/hermeneia_ml_research/remove_bg.py ~/Downloads/logo.png logo_transparent.png
```

### Notes

- If you have `rembg[gpu]` installed but CUDA libraries are missing, onnxruntime will log a warning and fall back to CPU automatically — this is fine.
- For CPU-only systems, `rembg` (without `[gpu]`) is sufficient.