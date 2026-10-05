"""
Example: Basic ILDA conversion usage.
"""

from pathlib import Path
import sys

# Add parent to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ilda_converter import convert_image, ILDAConverter, GalvoConfig


def example_basic():
    """Basic conversion with defaults."""
    print("=" * 60)
    print("Example 1: Basic Conversion")
    print("=" * 60)

    # You would replace this with an actual image path
    input_image = "logo.png"
    output_file = "output_basic.ilda"

    if not Path(input_image).exists():
        print(f"⚠ Skipping: {input_image} not found")
        print("  Create a sample image first")
        return

    stats = convert_image(
        input_image,
        output_file,
        vectorizer="vtracer",
        pps=30000,
        fps=30,
        color=(255, 255, 255),
    )

    print(f"\n✓ Created: {output_file}")
    print(f"  Points: {stats['total_points']}")
    print(f"  Budget: {stats['budget_utilization']:.1%}")


def example_high_contrast():
    """High-contrast B&W logo with Potrace."""
    print("\n" + "=" * 60)
    print("Example 2: High-Contrast B&W (Potrace)")
    print("=" * 60)

    input_image = "logo_bw.png"
    output_file = "output_potrace.ilda"

    if not Path(input_image).exists():
        print(f"⚠ Skipping: {input_image} not found")
        return

    stats = convert_image(
        input_image,
        output_file,
        vectorizer="potrace",
        preprocess=True,  # Apply thresholding
        color=(0, 255, 0),  # Green
    )

    print(f"\n✓ Created: {output_file}")


def example_text():
    """Text or thin lines with centerline extraction."""
    print("\n" + "=" * 60)
    print("Example 3: Text / Thin Lines (Centerline)")
    print("=" * 60)

    input_image = "text.png"
    output_file = "output_text.ilda"

    if not Path(input_image).exists():
        print(f"⚠ Skipping: {input_image} not found")
        return

    stats = convert_image(
        input_image,
        output_file,
        vectorizer="centerline",
        preprocess=True,
        color=(255, 0, 0),  # Red
    )

    print(f"\n✓ Created: {output_file}")


def example_custom_galvo():
    """Custom galvo configuration for high-end scanner."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Galvo Config (40k PPS)")
    print("=" * 60)

    input_image = "logo.png"
    output_file = "output_40k.ilda"

    if not Path(input_image).exists():
        print(f"⚠ Skipping: {input_image} not found")
        return

    # High-performance scanner config
    galvo_config = GalvoConfig(
        pps=40000,  # 40k PPS scanner
        target_fps=30,
        corner_threshold_deg=30.0,  # Tighter corners
        corner_dwell_points=4,  # More dwell for precision
        jump_settle_points=4,
    )

    converter = ILDAConverter(
        vectorizer_method="vtracer",
        galvo_config=galvo_config,
        simplify_tolerance=0.5,  # Less simplification
    )

    stats = converter.convert(
        input_image,
        output_file,
        color=(255, 128, 0),  # Orange
    )

    print(f"\n✓ Created: {output_file}")
    print(f"  Frame budget: {stats['point_budget']} points")
    print(f"  Actual usage: {stats['total_points']} points")


def example_pipeline_details():
    """Show detailed pipeline execution."""
    print("\n" + "=" * 60)
    print("Example 5: Detailed Pipeline")
    print("=" * 60)

    input_image = "logo.png"

    if not Path(input_image).exists():
        print(f"⚠ Skipping: {input_image} not found")
        print("\nTo run examples, create sample images:")
        print("  logo.png        - Full color logo")
        print("  logo_bw.png     - Black & white logo")
        print("  text.png        - Text or thin lines")
        return

    # This runs with verbose output from the pipeline
    converter = ILDAConverter(
        vectorizer_method="vtracer",
        simplify_tolerance=1.0,
    )

    stats = converter.convert(
        input_image,
        "output_detailed.ilda",
        color=(100, 200, 255),
        preprocess=False,
    )

    # Show all statistics
    print("\n" + "=" * 60)
    print("Pipeline Statistics")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")


if __name__ == "__main__":
    print("ILDA Converter Examples\n")

    # Run all examples
    example_basic()
    example_high_contrast()
    example_text()
    example_custom_galvo()
    example_pipeline_details()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
