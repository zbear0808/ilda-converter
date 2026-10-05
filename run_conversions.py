"""
Run ILDA conversions on both test files:
1. speedsvg.svg  (SVG vector input) -> speedsvg.ilda
2. speedstars.jpg (JPEG raster input) -> speedstars.ilda
"""

import os
from pathlib import Path
from ilda_converter import ILDAConverter, GalvoConfig, read_ilda_file

svg_input = r"C:\Users\zubair\Downloads\speedsvg.svg"
svg_output_downloads = r"C:\Users\zubair\Downloads\speedsvg.ilda"
svg_output_workspace = r"speedsvg.ilda"

jpg_input = r"C:\Users\zubair\Downloads\speedstars.jpg"
jpg_output_downloads = r"C:\Users\zubair\Downloads\speedstars.ilda"
jpg_output_workspace = r"speedstars.ilda"

galvo_config = GalvoConfig(
    pps=30000,
    target_fps=30,
    corner_threshold_deg=45.0,
    corner_dwell_points=3,
    jump_settle_points=3,
    tail_dwell_points=2,
    jump_step_size=2000.0,
    invert_y=True,
    loop_to_start=True,
)

print("=" * 70)
print("RUNNING ILDA CONVERSIONS")
print("=" * 70)

# --- 1. SVG CONVERSION ---
print("\n[1/2] Converting SVG input: speedsvg.svg")
print("-" * 50)
converter_svg = ILDAConverter(galvo_config=galvo_config)
stats_svg = converter_svg.convert_svg(
    svg_input,
    svg_output_downloads,
    color=(0, 255, 255),  # Cyan
    frame_name="SPEEDSVG",
    company_name="ILDA_CVT",
)

# Also save a copy in the workspace
import shutil
shutil.copy2(svg_output_downloads, svg_output_workspace)

# Verify SVG ILDA output
data_svg = read_ilda_file(svg_output_downloads)
print("\nSVG Conversion Verification:")
print(f"  Output path:        {svg_output_downloads}")
print(f"  Workspace copy:     {os.path.abspath(svg_output_workspace)}")
print(f"  File size:          {os.path.getsize(svg_output_downloads):,} bytes")
print(f"  ILDA Format Code:   {data_svg['header']['format_code']}")
print(f"  Frame Name:         {data_svg['header']['frame_name']}")
print(f"  Company:            {data_svg['header']['company_name']}")
print(f"  Total Points:       {data_svg['total_points']}")
print(f"    Beam On:          {data_svg['beam_on_count']}")
print(f"    Blanked:          {data_svg['blanked_count']}")
print(f"  Budget Utilization: {data_svg['total_points'] / galvo_config.pps * galvo_config.target_fps:.1%}")

# --- 2. JPEG RASTER CONVERSION ---
print("\n[2/2] Converting Raster JPEG input: speedstars.jpg")
print("-" * 50)
converter_jpg = ILDAConverter(
    vectorizer_method="vtracer",
    galvo_config=galvo_config,
    simplify_tolerance=1.0,
    path_merge_threshold=0.5,
)
stats_jpg = converter_jpg.convert_raster(
    jpg_input,
    jpg_output_downloads,
    color=(255, 255, 0),  # Yellow
    frame_name="SPDSTARS",
    company_name="ILDA_CVT",
)

# Also save a copy in the workspace
shutil.copy2(jpg_output_downloads, jpg_output_workspace)

# Verify JPEG ILDA output
data_jpg = read_ilda_file(jpg_output_downloads)
print("\nJPEG Conversion Verification:")
print(f"  Output path:        {jpg_output_downloads}")
print(f"  Workspace copy:     {os.path.abspath(jpg_output_workspace)}")
print(f"  File size:          {os.path.getsize(jpg_output_downloads):,} bytes")
print(f"  ILDA Format Code:   {data_jpg['header']['format_code']}")
print(f"  Frame Name:         {data_jpg['header']['frame_name']}")
print(f"  Company:            {data_jpg['header']['company_name']}")
print(f"  Total Points:       {data_jpg['total_points']}")
print(f"    Beam On:          {data_jpg['beam_on_count']}")
print(f"    Blanked:          {data_jpg['blanked_count']}")
print(f"  Budget Utilization: {data_jpg['total_points'] / galvo_config.pps * galvo_config.target_fps:.1%}")

print("\n" + "=" * 70)
print("[OK] ALL CONVERSIONS COMPLETED AND VERIFIED SUCCESSFULLY")
print("=" * 70)
