"""
Command-line interface for ILDA converter.
"""

import argparse
import sys
from pathlib import Path

from .pipeline import convert_image
from .galvo_conditioner import GalvoConfig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert raster images to optimized ILDA laser projector files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  ilda-convert logo.png output.ilda

  # High-contrast black & white with Potrace
  ilda-convert --vectorizer potrace --preprocess logo.png output.ilda

  # Thin lines / text with centerline extraction
  ilda-convert --vectorizer centerline --preprocess text.png output.ilda

  # Custom galvo settings for 40k PPS scanner
  ilda-convert --pps 40000 --fps 30 logo.png output.ilda

  # Blue color output
  ilda-convert --color 0,0,255 logo.png output.ilda
        """,
    )

    parser.add_argument(
        "input",
        help="Input image file (PNG, JPG, BMP, etc.)",
    )

    parser.add_argument(
        "output",
        help="Output ILDA file (.ilda)",
    )

    parser.add_argument(
        "-v",
        "--vectorizer",
        choices=["vtracer", "potrace", "centerline"],
        default="centerline",
        help="Vectorization method (default: centerline - works on all platforms)",
    )

    parser.add_argument(
        "-p",
        "--preprocess",
        action="store_true",
        help="Apply preprocessing (thresholding, cleanup)",
    )

    parser.add_argument(
        "--threshold",
        choices=["otsu", "adaptive", "manual"],
        default="otsu",
        help="Thresholding method for preprocessing (default: otsu)",
    )

    parser.add_argument(
        "--pps",
        type=int,
        default=30000,
        help="Galvo points per second (default: 30000)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate (default: 30)",
    )

    parser.add_argument(
        "--color",
        type=str,
        default="255,255,255",
        help="RGB color as R,G,B (default: 255,255,255 white)",
    )

    parser.add_argument(
        "--corner-threshold",
        type=float,
        default=45.0,
        help="Corner dwell angle threshold in degrees (default: 45.0)",
    )

    parser.add_argument(
        "--simplify",
        type=float,
        default=1.0,
        help="Path simplification tolerance (default: 1.0)",
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Parse color
    try:
        color = tuple(map(int, args.color.split(",")))
        if len(color) != 3 or not all(0 <= c <= 255 for c in color):
            raise ValueError
    except ValueError:
        print(
            f"Error: Invalid color format. Use R,G,B with values 0-255",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build galvo config
    galvo_config = GalvoConfig(
        pps=args.pps,
        target_fps=args.fps,
        corner_threshold_deg=args.corner_threshold,
    )

    # Run conversion
    try:
        from .pipeline import ILDAConverter

        converter = ILDAConverter(
            vectorizer_method=args.vectorizer,
            galvo_config=galvo_config,
            simplify_tolerance=args.simplify,
        )

        stats = converter.convert(
            str(input_path),
            args.output,
            color=color,
            preprocess=args.preprocess,
            threshold_method=args.threshold,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        print(f"Input:              {stats['input_image']}")
        print(f"Output:             {stats['output_file']}")
        print(f"Vectorizer:         {stats['vectorizer']}")
        print(f"Paths:              {stats['path_count']}")
        print(f"Total Points:       {stats['total_points']}")
        print(f"  Beam On:          {stats['beam_on_points']}")
        print(f"  Beam Off:         {stats['beam_off_points']}")
        print(f"Blanking Distance:  {stats['blanking_distance']:.1f} units")
        print(f"Point Budget:       {stats['point_budget']}")
        print(f"Budget Usage:       {stats['budget_utilization']:.1%}")
        print("=" * 60)

        if stats["budget_utilization"] > 1.0:
            print(
                "\n⚠ WARNING: Point budget exceeded! Frame rate will be reduced.",
                file=sys.stderr,
            )
            print(
                "Consider: reducing PPS, increasing FPS, or simplifying the image.",
                file=sys.stderr,
            )

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
