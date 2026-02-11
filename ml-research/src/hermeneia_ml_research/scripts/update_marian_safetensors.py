#!/usr/bin/env python3
"""
Update Marian models TOML with SafeTensors availability metadata.

Checks each Marian model for SafeTensors availability via HTTP HEAD requests
and adds has_safetensors field to the model catalog.
"""

import argparse
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote

import requests
import toml


def check_safetensors_available(model_id: str, revision: str) -> bool:
    """Check if SafeTensors version exists for a model."""
    encoded_revision = quote(revision, safe="")
    url = f"https://huggingface.co/api/models/{model_id}/tree/{encoded_revision}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return False
        entries = response.json()
    except Exception:
        return False

    for entry in entries:
        if entry.get("path") == "model.safetensors":
            return True
    return False


def update_models_with_safetensors(models_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update models data with has_safetensors metadata."""
    if 'marian' not in models_data:
        return models_data
    
    print(f"Checking {len(models_data['marian'])} Marian models for SafeTensors availability...")
    
    updated_models = models_data.copy()
    available_count = 0
    
    for model in updated_models['marian']:
        model_id = model['model_id']
        revision = model['revision']
        
        has_safetensors = check_safetensors_available(model_id, revision)
        model['has_safetensors'] = has_safetensors
        
        if has_safetensors:
            available_count += 1
            print(f"✓ {model_id}: SafeTensors available")
        else:
            print(f"✗ {model_id}: SafeTensors not available")
    
    print(f"\nSummary: {available_count}/{len(updated_models['marian'])} models have SafeTensors")
    return updated_models


def main():
    parser = argparse.ArgumentParser(description="Update Marian models with SafeTensors metadata")
    parser.add_argument("--input", required=True, help="Input TOML file path")
    parser.add_argument("--output", required=True, help="Output TOML file path")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Read input TOML
    with open(input_path, 'r') as f:
        models_data = toml.load(f)
    
    # Update with SafeTensors metadata
    updated_data = update_models_with_safetensors(models_data)
    
    # Write output TOML
    with open(output_path, 'w') as f:
        toml.dump(updated_data, f)
    
    print(f"Updated model catalog written to {output_path}")


if __name__ == "__main__":
    main()
