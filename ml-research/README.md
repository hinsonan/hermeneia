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