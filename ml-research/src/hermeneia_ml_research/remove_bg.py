#!/usr/bin/env python3
"""
Remove background from a PNG using rembg (U2Net neural network),
with high-quality post-processing: alpha edge smoothing, foreground
decontamination, unsharp-mask sharpening, and optional Lanczos upscaling.

Usage:
    python remove_bg.py input.png [output.png] [--upscale N] [--no-postprocess]
"""

import argparse
import sys
from pathlib import Path


def postprocess(
    result,
    *,
    alpha_smooth_radius: float = 1.2,
    decontaminate: bool = True,
    sharpen: bool = True,
    upscale: int = 1,
):
    """
    High-quality post-processing for a background-removed RGBA image.

    Steps
    -----
    1. Alpha edge smoothing  – Gaussian blur applied only in the transition
       zone (5 < alpha < 250), leaving solid interior/exterior untouched.
    2. Foreground decontamination – removes background-color bleed on
       semi-transparent edge pixels using premultiplied-alpha extrapolation.
    3. Unsharp mask – sharpens the foreground subject's RGB channels.
    4. Lanczos upscaling – high-quality resize when upscale > 1.

    Parameters
    ----------
    result : PIL.Image
        RGBA image output from rembg.
    alpha_smooth_radius : float
        Gaussian sigma for edge smoothing / decontamination extrapolation.
    decontaminate : bool
        Apply foreground colour decontamination.
    sharpen : bool
        Apply unsharp-mask sharpening to RGB.
    upscale : int
        Integer scale factor for Lanczos upscaling (1 = no upscale).
    """
    import numpy as np
    from PIL import Image, ImageFilter
    from scipy.ndimage import gaussian_filter

    rgba = np.array(result.convert("RGBA"), dtype=np.float32)
    rgb = rgba[..., :3]          # shape (H, W, 3)
    alpha = rgba[..., 3]         # shape (H, W),  0-255

    # ------------------------------------------------------------------ #
    # 1. Alpha edge smoothing                                              #
    # Blur the alpha channel, then restrict the result to the transition  #
    # zone so solid opaque/transparent regions remain crisp.              #
    # ------------------------------------------------------------------ #
    smoothed_alpha = gaussian_filter(alpha, sigma=alpha_smooth_radius)

    solid   = alpha > 240   # fully opaque  → keep original
    empty   = alpha < 5     # fully transparent → keep original
    blended = ~solid & ~empty

    alpha[blended] = smoothed_alpha[blended]

    # ------------------------------------------------------------------ #
    # 2. Foreground decontamination                                        #
    # Semi-transparent edge pixels contain a mix of foreground and        #
    # background colour.  Extrapolate the "true" foreground colour using  #
    # premultiplied-alpha Gaussian extrapolation, then replace the edge   #
    # pixel colours with that estimate.                                   #
    #                                                                     #
    # Derivation:  for a pixel with colour C and alpha α,                 #
    #   C_premult = α * C                                                 #
    # Blurring premultiplied values and the alpha map, then dividing,    #
    # extrapolates interior foreground colour outward into the fringe.    #
    # ------------------------------------------------------------------ #
    if decontaminate:
        alpha_norm  = alpha / 255.0                          # 0-1

        # Premultiply
        premult = rgb * alpha_norm[..., np.newaxis]          # (H, W, 3)

        # Blur both premultiplied colour and alpha
        blurred_premult = np.stack(
            [gaussian_filter(premult[..., c], sigma=alpha_smooth_radius) for c in range(3)],
            axis=-1,
        )
        blurred_alpha = gaussian_filter(alpha_norm, sigma=alpha_smooth_radius)

        # Un-premultiply to get extrapolated foreground colour
        safe_alpha = np.maximum(blurred_alpha, 1e-6)[..., np.newaxis]
        extrapolated = np.clip(blurred_premult / safe_alpha, 0.0, 255.0)

        # Apply only within the transition zone
        rgb[blended] = extrapolated[blended]

    # ------------------------------------------------------------------ #
    # 3. Reconstruct RGBA and apply unsharp-mask sharpening              #
    # ------------------------------------------------------------------ #
    out_arr = np.dstack([np.clip(rgb, 0, 255), alpha[..., np.newaxis]]).astype(np.uint8)
    out = Image.fromarray(out_arr, "RGBA")

    if sharpen:
        r, g, b, a_ch = out.split()
        rgb_img    = Image.merge("RGB", (r, g, b))
        sharpened  = rgb_img.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=130, threshold=2)
        )
        r2, g2, b2 = sharpened.split()
        out = Image.merge("RGBA", (r2, g2, b2, a_ch))

    # ------------------------------------------------------------------ #
    # 4. Lanczos upscaling                                                #
    # ------------------------------------------------------------------ #
    if upscale > 1:
        w, h = out.size
        out = out.resize((w * upscale, h * upscale), Image.LANCZOS)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Remove PNG background via rembg (U2Net) with HQ post-processing"
    )
    parser.add_argument("input",  help="Input PNG file")
    parser.add_argument("output", nargs="?", help="Output PNG file (default: input_nobg.png)")
    parser.add_argument(
        "--upscale", type=int, default=1, metavar="N",
        help="Lanczos upscale factor after processing (default: 1 = no upscale)",
    )
    parser.add_argument(
        "--alpha-smooth-radius", type=float, default=1.2, metavar="R",
        help="Gaussian sigma for alpha edge smoothing / decontamination (default: 1.2)",
    )
    parser.add_argument(
        "--no-postprocess", action="store_true",
        help="Skip all post-processing and save the raw rembg output",
    )
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

    if args.no_postprocess:
        result.save(output_path)
    else:
        print("Post-processing: alpha smoothing, decontamination, sharpening"
              + (f", {args.upscale}x Lanczos upscale" if args.upscale > 1 else "") + "...")
        try:
            import numpy  # noqa: F401
            from scipy.ndimage import gaussian_filter  # noqa: F401
        except ImportError:
            print("Warning: numpy/scipy not found — skipping post-processing.", file=sys.stderr)
            result.save(output_path)
            print(f"Saved to '{output_path}'")
            return

        result = postprocess(
            result,
            alpha_smooth_radius=args.alpha_smooth_radius,
            upscale=args.upscale,
        )
        result.save(output_path, optimize=True)

    print(f"Saved to '{output_path}'")


if __name__ == "__main__":
    main()
