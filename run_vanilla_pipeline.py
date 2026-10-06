"""
Vanilla ILDA Conversion Pipeline for a Single Raster Image:
Input: C:\\Users\\zubair\\Downloads\\logo_clean_lines.png
No multi-image merging. Standard vanilla conversion:
1. Raster vectorization via VTracer
2. Standard path optimization (vpype TSP shortest jumps, collinear simplify)
3. Coordinate normalization to ILDA space (-32768 to 32767)
4. Galvo dynamics conditioning (30k PPS, corner dwells, blanked jumps & settles)
5. ILDA Format 5 binary export (.ild for Beyond, .ilda)
6. Laser preview rendering & zoom detail inspection
"""

import os
import shutil
from pathlib import Path
from PIL import Image

from ilda_converter import (
    ILDAConverter,
    GalvoConfig,
    read_ilda_file,
    Vectorizer,
)
from generate_thickness_comparison import render_ilda_preview
from run_close_line_merge_pipeline import crop_detail_region

# Configuration Paths
INPUT_IMAGE = Path(r"C:\Users\zubair\Downloads\logo_clean_lines.png")
DOWNLOADS_DIR = Path(r"C:\Users\zubair\Downloads")

OUTPUT_DIR = Path("c:/Users/zubair/Documents/GitHub/ilda-converter/output_vanilla")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_vanilla_pipeline():
    print("=" * 75)
    print("RUNNING VANILLA ILDA PIPELINE (SINGLE FILE, NO MERGING)")
    print("=" * 75)
    print(f"Input image:      {INPUT_IMAGE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_IMAGE}")
        
    # Standard galvo configuration (30k PPS, 30 FPS target)
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
    
    # Standard vanilla converter
    converter = ILDAConverter(
        vectorizer_method="vtracer",
        galvo_config=galvo_config,
        simplify_tolerance=1.0,
        path_merge_threshold=0.5,
        line_thickness=25.0,  # Standard ribbon collapse for single centerlines
        merge_close_distance=0.0,  # No close line merging
    )
    
    # 1. Also export raw vectorized SVG for the user
    svg_out_workspace = OUTPUT_DIR / "logo_clean_lines.svg"
    svg_out_downloads = DOWNLOADS_DIR / "logo_clean_lines.svg"
    print(f"\n[1/3] Vectorizing to SVG...")
    paths, _ = converter.vectorizer.vectorize(str(INPUT_IMAGE), str(svg_out_workspace))
    shutil.copy2(svg_out_workspace, svg_out_downloads)
    print(f"      -> Extracted {len(paths)} vector paths")
    print(f"      -> Saved SVG: {svg_out_downloads} ({svg_out_downloads.stat().st_size:,} bytes)")
    
    # 2. Run full vanilla conversion to .ild and .ilda
    ild_out_workspace = OUTPUT_DIR / "logo_clean_lines.ild"
    print(f"\n[2/3] Running vanilla raster-to-ILDA conversion...")
    stats = converter.convert_raster(
        str(INPUT_IMAGE),
        str(ild_out_workspace),
        frame_name="CLEANLOGO",
        company_name="ILDA_CVT",
        write_both=True,
    )
    
    # Copy ILDA files to Downloads
    ild_out_downloads = DOWNLOADS_DIR / "logo_clean_lines.ild"
    ilda_out_downloads = DOWNLOADS_DIR / "logo_clean_lines.ilda"
    shutil.copy2(ild_out_workspace, ild_out_downloads)
    shutil.copy2(ild_out_workspace.with_suffix(".ilda"), ilda_out_downloads)
    
    # 3. Read back and verify ILDA binary structure
    data = read_ilda_file(str(ild_out_downloads))
    print(f"\n[3/3] Verifying ILDA binary and rendering preview...")
    print(f"      Format code:        {data['header']['format_code']}")
    print(f"      Frame name:         {data['header']['frame_name']}")
    print(f"      Total points:       {data['total_points']}")
    print(f"      Beam-on points:     {data['beam_on_count']}")
    print(f"      Blanked points:     {data['blanked_count']}")
    print(f"      Blanking distance:  {stats['blanking_distance']:.1f} units")
    print(f"      Frame duration:     {data['total_points'] / galvo_config.pps * 1000:.1f} ms ({galvo_config.pps / data['total_points']:.1f} FPS @ 30k PPS)")
    
    # Render authentic laser simulation preview
    sub_title = (
        f"Paths: {stats['path_count']} | Points: {data['total_points']} "
        f"(Beam: {data['beam_on_count']}, Blanked: {data['blanked_count']}) | "
        f"Blanking Jump: {stats['blanking_distance']:.1f}"
    )
    img_preview = render_ilda_preview(
        data,
        size=900,
        title_text="LOGO CLEAN LINES - VANILLA PIPELINE",
        subtitle_text=sub_title,
    )
    preview_workspace = OUTPUT_DIR / "logo_clean_lines_preview.png"
    preview_downloads = DOWNLOADS_DIR / "logo_clean_lines_preview.png"
    img_preview.save(preview_workspace)
    shutil.copy2(preview_workspace, preview_downloads)
    
    # Render detail zoom
    img_zoom = crop_detail_region(img_preview)
    zoom_workspace = OUTPUT_DIR / "logo_clean_lines_detail_zoom.png"
    img_zoom.save(zoom_workspace)
    
    print("\n" + "=" * 75)
    print("CONVERSION COMPLETE AND VERIFIED (.ild AND .ilda)")
    print("=" * 75)
    print("Deliverables saved to Downloads:")
    print(f"  Vector SVG:     {svg_out_downloads} ({svg_out_downloads.stat().st_size:,} bytes)")
    print(f"  Beyond (.ild):  {ild_out_downloads} ({ild_out_downloads.stat().st_size:,} bytes)")
    print(f"  Format 5 (.ilda): {ilda_out_downloads} ({ilda_out_downloads.stat().st_size:,} bytes)")
    print(f"  Preview Image:  {preview_downloads}")
    print(f"\nAll intermediate files and zoom detail saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_vanilla_pipeline()

