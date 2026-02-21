#!/usr/bin/env python3
"""
Remove background from a PNG using rembg (U2Net neural network).

Usage:
    python remove_bg.py input.png [output.png]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Remove PNG background via rembg (U2Net)")
    parser.add_argument("input", help="Input PNG file")
    parser.add_argument("output", nargs="?", help="Output PNG file (default: input_nobg.png)")
    args = parser.parse_args()

    try:
        from rembg import remove
        from PIL import Image
    except ImportError:
        print("Error: install dependencies with:  pip install rembg pillow", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output) if args.output
        else input_path.with_stem(input_path.stem + "_nobg")
    ).with_suffix(".png")

    print(f"Loading '{input_path}'...")
    img = Image.open(input_path)

    print("Removing background (downloads U2Net model on first run)...")
    result = remove(img)

    result.save(output_path)
    print(f"Saved to '{output_path}'")


if __name__ == "__main__":
    main()
