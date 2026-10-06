"""
Command-line interface for ILDA converter.
"""

import argparse
import sys
from pathlib import Path

from .pipeline import ILDAConverter
from .galvo_conditioner import GalvoConfig


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert raster images and SVG files to optimized ILDA laser projector files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (SVG or Raster)
  ilda-convert logo.png output.ilda
  ilda-convert vector.svg output.ilda

  # Modern vision-based color/binary vectorization with VTracer
  ilda-convert --vectorizer vtracer logo.png output.ilda

  # High-contrast black & white with Potrace
  ilda-convert --vectorizer potrace --preprocess logo.png output.ilda

  # Thin lines / text with centerline skeleton extraction
  ilda-convert --vectorizer centerline --preprocess text.png output.ilda

  # Custom galvo settings for 40k PPS scanner at 30 FPS
  ilda-convert --pps 40000 --fps 30 logo.png output.ilda

  # Cyan color output (R,G,B)
  ilda-convert --color 0,255,255 logo.png output.ilda
        """,
    )

    parser.add_argument(
        "input",
        help="Input image or SVG file (SVG, PNG, JPG, BMP, etc.)",
    )

    parser.add_argument(
        "output",
        help="Output ILDA file (.ilda)",
    )

    parser.add_argument(
        "-v",
        "--vectorizer",
        choices=["vtracer", "centerline", "potrace"],
        default="vtracer",
        help="Vectorization method (default: vtracer)",
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

    parser.add_argument(
        "--line-thickness",
        type=float,
        default=25.0,
        help="Max line thickness in pixels to collapse ribbons into single centerlines (default: 25.0, 0 to disable)",
    )

    parser.add_argument(
        "--merge-close-distance",
        type=float,
        default=0.0,
        help="Max distance in units to merge separate nearby parallel lines into a single centerline (default: 0, 0 to disable)",
    )

    parser.add_argument(
        "--text-roi",
        type=str,
        default=None,
        help="Region of interest for text: 'auto', 'ymin,ymax', or 'xmin,ymin,xmax,ymax' in 0.0-1.0 coords (default: None)",
    )

    parser.add_argument(
        "--text-simplify",
        type=float,
        default=0.5,
        help="Simplification tolerance for text paths in mm (default: 0.5)",
    )

    parser.add_argument(
        "--bg-simplify",
        type=float,
        default=3.5,
        help="Simplification tolerance for background paths in mm (default: 3.5)",
    )

    parser.add_argument(
        "--text-thickness",
        type=float,
        default=0.0,
        help="Ribbon collapse thickness for text (default: 0.0 to preserve font outlines)",
    )

    parser.add_argument(
        "--bg-thickness",
        type=float,
        default=None,
        help="Ribbon collapse thickness for background (default: matches --line-thickness)",
    )

    parser.add_argument(
        "--frame-name",
        type=str,
        default="FRAME",
        help="ILDA frame name (up to 8 chars)",
    )

    parser.add_argument(
        "--company-name",
        type=str,
        default="CONVERTER",
        help="ILDA company name (up to 8 chars)",
    )

    parser.add_argument(
        "-b",
        "--both",
        action="store_true",
        help="Write both .ild (Pangolin Beyond) and .ilda files",
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
            "Error: Invalid color format. Use R,G,B with values 0-255",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse text ROI
    text_roi = None
    if args.text_roi:
        if args.text_roi.lower() == "auto":
            text_roi = "auto"
        else:
            try:
                parts = tuple(map(float, args.text_roi.split(",")))
                if len(parts) in (2, 4):
                    text_roi = parts
                else:
                    raise ValueError
            except ValueError:
                print(
                    "Error: Invalid text-roi format. Use 'auto', 'ymin,ymax', or 'xmin,ymin,xmax,ymax'",
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
        converter = ILDAConverter(
            vectorizer_method=args.vectorizer,
            galvo_config=galvo_config,
            simplify_tolerance=args.simplify,
            line_thickness=args.line_thickness,
            merge_close_distance=args.merge_close_distance,
            text_roi=text_roi,
            text_tolerance=args.text_simplify,
            bg_tolerance=args.bg_simplify,
            text_line_thickness=args.text_thickness,
            bg_line_thickness=args.bg_thickness,
        )

        stats = converter.convert(
            str(input_path),
            args.output,
            color=color,
            preprocess=args.preprocess,
            threshold_method=args.threshold,
            frame_name=args.frame_name,
            company_name=args.company_name,
            write_both=args.both,
            line_thickness=args.line_thickness,
            merge_close_distance=args.merge_close_distance,
            text_roi=text_roi,
            text_tolerance=args.text_simplify,
            bg_tolerance=args.bg_simplify,
            text_line_thickness=args.text_thickness,
            bg_line_thickness=args.bg_thickness,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        input_file = stats.get("input_file", stats.get("input_image", str(input_path)))
        print(f"Input:              {input_file}")
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

        if stats["budget_utilization"] > 1.25:
            print(
                "\n[WARNING] Point budget exceeded! Frame rate will be reduced.",
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
