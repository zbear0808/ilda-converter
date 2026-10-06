"""
Rerun the complete ILDA conversion pipeline with close-line merging post-processing logic
on all test files: speedsvg.svg, speedstars.jpg, and another_speed.jpg.
Saves all binary .ild / .ilda outputs, preview renders, and comparison sheets into output_close_line_merge/.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ilda_converter import (
    ILDAConverter,
    GalvoConfig,
    read_ilda_file,
)
from generate_thickness_comparison import render_ilda_preview


OUTPUT_DIR = Path("c:/Users/zubair/Documents/GitHub/ilda-converter/output_close_line_merge")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = [
    {
        "id": "speedsvg",
        "name": "Speedstars (SVG Direct)",
        "path": r"C:\Users\zubair\Downloads\speedsvg.svg",
        "merge_distance": 12.0,
    },
    {
        "id": "speedstars_jpg",
        "name": "Speedstars (Raster JPG)",
        "path": r"C:\Users\zubair\Downloads\speedstars.jpg",
        "merge_distance": 18.0,
    },
    {
        "id": "another_speed_jpg",
        "name": "Another Speed (Raster JPG)",
        "path": r"C:\Users\zubair\Downloads\another_speed.jpg",
        "merge_distance": 18.0,
    },
]


def crop_detail_region(img: Image.Image) -> Image.Image:
    """Crop the central graphic/stroke region of a 900x900 preview render."""
    w, h = img.size
    crop_box = (int(w * 0.10), int(h * 0.35), int(w * 0.90), int(h * 0.78))
    cropped = img.crop(crop_box)
    return cropped.resize((int(w * 0.78), int(h * 0.44)), Image.Resampling.LANCZOS)


def run_pipeline():
    print(f"================================================================")
    print(f"Executing ILDA Pipeline with Close-Line Merging Post-Processing")
    print(f"Target Output Directory: {OUTPUT_DIR}")
    print(f"================================================================\n")

    galvo_config = GalvoConfig(pps=30000, target_fps=30)
    summary_results = []

    for item in INPUTS:
        src_path = item["path"]
        item_id = item["id"]
        title = item["name"]
        merge_dist = item["merge_distance"]

        if not Path(src_path).exists():
            print(f"[SKIP] Input not found: {src_path}")
            continue

        print(f"\n>>> Processing: {title} ({src_path})")
        print(f"    Proximity Merge Distance: {merge_dist:.1f} units")

        # -------------------------------------------------------------
        # 1. BASELINE: Without close-line merging
        # -------------------------------------------------------------
        base_ild_path = OUTPUT_DIR / f"{item_id}_baseline_no_merge.ild"
        conv_base = ILDAConverter(
            vectorizer_method="vtracer",
            galvo_config=galvo_config,
            simplify_tolerance=1.0,
            line_thickness=25.0,
            merge_close_distance=0.0,
            text_roi="auto",
            text_tolerance=0.5,
            bg_tolerance=3.5,
            text_line_thickness=0.0,
            bg_line_thickness=25.0,
        )
        stats_base = conv_base.convert(
            src_path,
            str(base_ild_path),
            write_both=True,
            line_thickness=25.0,
            merge_close_distance=0.0,
            text_roi="auto",
            text_tolerance=0.5,
            bg_tolerance=3.5,
            text_line_thickness=0.0,
            bg_line_thickness=25.0,
        )
        data_base = read_ilda_file(str(base_ild_path))

        # -------------------------------------------------------------
        # 2. PROXIMITY MERGED: With close-line merging logic active
        # -------------------------------------------------------------
        merged_ild_path = OUTPUT_DIR / f"{item_id}_with_close_merge.ild"
        conv_merged = ILDAConverter(
            vectorizer_method="vtracer",
            galvo_config=galvo_config,
            simplify_tolerance=1.0,
            line_thickness=25.0,
            merge_close_distance=merge_dist,
            text_roi="auto",
            text_tolerance=0.5,
            bg_tolerance=3.5,
            text_line_thickness=0.0,
            bg_line_thickness=25.0,
        )
        stats_merged = conv_merged.convert(
            src_path,
            str(merged_ild_path),
            write_both=True,
            line_thickness=25.0,
            merge_close_distance=merge_dist,
            text_roi="auto",
            text_tolerance=0.5,
            bg_tolerance=3.5,
            text_line_thickness=0.0,
            bg_line_thickness=25.0,
        )
        data_merged = read_ilda_file(str(merged_ild_path))

        # -------------------------------------------------------------
        # 3. Render authentic laser projector preview images
        # -------------------------------------------------------------
        img_base = render_ilda_preview(
            data_base,
            size=900,
            title_text=f"{item_id.upper()} - WITHOUT CLOSE MERGE",
            subtitle_text=(
                f"Close-Merge: OFF | Paths: {stats_base['path_count']} | "
                f"Points: {stats_base['total_points']} | Blanking: {stats_base['blanking_distance']:.1f}"
            ),
        )
        img_merged = render_ilda_preview(
            data_merged,
            size=900,
            title_text=f"{item_id.upper()} - WITH CLOSE MERGE ({merge_dist:.1f}u)",
            subtitle_text=(
                f"Close-Merge: {merge_dist:.1f}u | Paths: {stats_merged['path_count']} | "
                f"Points: {stats_merged['total_points']} | Blanking: {stats_merged['blanking_distance']:.1f}"
            ),
        )

        preview_base_path = OUTPUT_DIR / f"{item_id}_baseline_preview.png"
        preview_merged_path = OUTPUT_DIR / f"{item_id}_with_close_merge_preview.png"
        img_base.save(preview_base_path)
        img_merged.save(preview_merged_path)

        # -------------------------------------------------------------
        # 4. Generate Side-by-Side Comparison Sheet
        # -------------------------------------------------------------
        W, H = img_base.size
        side_by_side = Image.new("RGB", (W * 2 + 30, H + 80), color=(15, 17, 23))
        draw = ImageDraw.Draw(side_by_side)
        draw.text(
            (30, 20),
            f"Close Vector Line Merge Comparison: {title}",
            fill=(255, 255, 255),
        )
        draw.text(
            (30, 48),
            f"Baseline: {stats_base['path_count']} paths, {stats_base['total_points']} pts | "
            f"Merged: {stats_merged['path_count']} paths, {stats_merged['total_points']} pts (Blanking: {stats_base['blanking_distance']:.0f} -> {stats_merged['blanking_distance']:.0f})",
            fill=(0, 230, 200),
        )
        side_by_side.paste(img_base, (10, 70))
        side_by_side.paste(img_merged, (W + 20, 70))

        sbs_path = OUTPUT_DIR / f"{item_id}_side_by_side_comparison.png"
        side_by_side.save(sbs_path)

        # -------------------------------------------------------------
        # 5. Generate Zoom Detail Sheet
        # -------------------------------------------------------------
        crop_b = crop_detail_region(img_base)
        crop_m = crop_detail_region(img_merged)
        cw, ch = crop_b.size
        zoom_sheet = Image.new("RGB", (cw * 2 + 30, ch + 80), color=(15, 17, 23))
        draw_z = ImageDraw.Draw(zoom_sheet)
        draw_z.text((30, 20), f"STROKE & DETAIL ZOOM: {title}", fill=(255, 255, 255))
        draw_z.text((30, 45), f"Left: Without Close Merge | Right: With Close Merge ({merge_dist:.1f}u)", fill=(0, 230, 200))
        zoom_sheet.paste(crop_b, (10, 70))
        zoom_sheet.paste(crop_m, (cw + 20, 70))

        zoom_path = OUTPUT_DIR / f"{item_id}_detail_zoom.png"
        zoom_sheet.save(zoom_path)

        summary_results.append({
            "id": item_id,
            "title": title,
            "merge_distance": merge_dist,
            "base_paths": stats_base["path_count"],
            "base_points": stats_base["total_points"],
            "base_blanking": stats_base["blanking_distance"],
            "merged_paths": stats_merged["path_count"],
            "merged_points": stats_merged["total_points"],
            "merged_blanking": stats_merged["blanking_distance"],
            "ild_output": str(merged_ild_path),
            "ilda_output": str(merged_ild_path.with_suffix(".ilda")),
            "preview_png": str(preview_merged_path),
            "comparison_png": str(sbs_path),
            "zoom_png": str(zoom_path),
        })

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Input Name':<28} | {'Base Pts':<10} | {'Merged Pts':<10} | {'Base Jump':<10} | {'Merged Jump':<10}")
    print("=" * 80)
    for res in summary_results:
        print(
            f"{res['id']:<28} | {res['base_points']:<10} | {res['merged_points']:<10} | "
            f"{res['base_blanking']:<10.1f} | {res['merged_blanking']:<10.1f}"
        )
    print("=" * 80)
    print(f"\nAll outputs successfully saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()

