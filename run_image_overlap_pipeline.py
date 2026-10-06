"""
Complete ILDA Conversion Pipeline for Overlapping Multi-Layer Images:
1. Vectorize raster background image (bgonly.jpg).
2. Vectorize raster foreground/text image (image.psd(1).png).
3. Overlap and merge into a single unified SVG with layered structure.
4. Execute full post-processing optimization steps:
   - Ribbon outline collapsing (medial axis centerlines)
   - Proximity / close-line merging
   - Collinear / segment simplification
   - Global TSP scanner route optimization (reloop + linesort)
   - Galvo conditioning (30k PPS, corner dwells, jump settles)
   - Format 5 binary ILDA export (.ild for Beyond, .ilda)
   - Laser projector preview rendering & comparison sheets
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw

from ilda_converter import (
    ILDAConverter,
    GalvoConfig,
    Vectorizer,
    read_ilda_file,
)
from generate_thickness_comparison import render_ilda_preview
from run_close_line_merge_pipeline import crop_detail_region

# Configuration Paths
OUTPUT_DIR = Path("c:/Users/zubair/Documents/GitHub/ilda-converter/output_merged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS_DIR = Path(r"C:\Users\zubair\Downloads")

BG_IMAGE = DOWNLOADS_DIR / "bg_clean.png"
TEXT_IMAGE = DOWNLOADS_DIR / "image.psd(1).png"


def prepare_raster_for_vectorizer(img_path: Path, temp_dir: Path) -> Path:
    """Ensure raster image has solid white background if RGBA/transparent."""
    img = Image.open(str(img_path))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        print(f"      [NOTE] Compositing alpha transparency of {img_path.name} onto solid white background...")
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            white_bg.paste(img, mask=img.split()[3])
        else:
            rgba = img.convert("RGBA")
            white_bg.paste(rgba, mask=rgba.split()[3])
        out_path = temp_dir / f"solid_bg_{img_path.name}"
        white_bg.save(str(out_path))
        return out_path
    return img_path


def vectorize_image_layers(
    bg_path: Path,
    text_path: Path,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Vectorize background and foreground/text raster images to SVG."""
    vectorizer = Vectorizer(method="vtracer")
    
    bg_svg = out_dir / "bgonly.svg"
    text_svg = out_dir / "textonly.svg"
    
    # Pre-composite transparency if present
    prep_bg = prepare_raster_for_vectorizer(bg_path, out_dir)
    prep_text = prepare_raster_for_vectorizer(text_path, out_dir)
    
    print(f"\n[1/4] Vectorizing background layer: {bg_path.name}...")
    paths_bg, _ = vectorizer.vectorize(str(prep_bg), str(bg_svg))
    print(f"      -> Extracted {len(paths_bg)} background vector path(s)")
    
    print(f"[2/4] Vectorizing text/foreground layer: {text_path.name}...")
    paths_text, _ = vectorizer.vectorize(str(prep_text), str(text_svg))
    print(f"      -> Extracted {len(paths_text)} text vector path(s)")
    
    return bg_svg, text_svg


def merge_svg_layers(
    bg_svg: Path,
    text_svg: Path,
    output_svg: Path,
    width: int = 2752,
    height: int = 1536,
) -> Path:
    """Combine background and text SVGs into a single overlapped SVG."""
    print(f"\n[3/4] Merging vector layers into unified SVG: {output_svg.name}...")
    
    tree_bg = ET.parse(str(bg_svg))
    tree_text = ET.parse(str(text_svg))
    
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    merged_root = ET.Element(
        "{http://www.w3.org/2000/svg}svg",
        {
            "version": "1.1",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )
    
    # Layer 1: Background paths
    g_bg = ET.SubElement(
        merged_root,
        "{http://www.w3.org/2000/svg}g",
        {"id": "background_layer"},
    )
    for elem in tree_bg.getroot().iter():
        if elem.tag.endswith("path"):
            g_bg.append(elem)
            
    # Layer 2: Text / Foreground paths
    g_text = ET.SubElement(
        merged_root,
        "{http://www.w3.org/2000/svg}g",
        {"id": "text_layer"},
    )
    for elem in tree_text.getroot().iter():
        if elem.tag.endswith("path"):
            g_text.append(elem)
            
    merged_tree = ET.ElementTree(merged_root)
    merged_tree.write(str(output_svg), encoding="utf-8", xml_declaration=True)
    print(f"      -> Successfully saved merged SVG: {output_svg} ({output_svg.stat().st_size:,} bytes)")
    return output_svg


def run_pipeline():
    print("=" * 80)
    print("STARTING MULTI-LAYER IMAGE OVERLAP & ILDA OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"Input Background: {BG_IMAGE}")
    print(f"Input Text Image: {TEXT_IMAGE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    if not BG_IMAGE.exists():
        raise FileNotFoundError(f"Missing background image: {BG_IMAGE}")
    if not TEXT_IMAGE.exists():
        raise FileNotFoundError(f"Missing text image: {TEXT_IMAGE}")
        
    # 1. Vectorize layers
    bg_svg, text_svg = vectorize_image_layers(BG_IMAGE, TEXT_IMAGE, OUTPUT_DIR)
    
    # 2. Overlap & Merge into single SVG
    merged_svg = OUTPUT_DIR / "merged_speedstars.svg"
    merge_svg_layers(bg_svg, text_svg, merged_svg)
    
    # Also save merged SVG to Downloads
    downloads_svg = DOWNLOADS_DIR / "merged_speedstars.svg"
    shutil.copy2(merged_svg, downloads_svg)
    
    # 3. Post-Processing Optimization Steps
    print(f"\n[4/4] Executing ILDA post-processing and optimization steps...")
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
    
    # Stage 3A: Baseline (without line merging or ribbon collapse)
    base_ild = OUTPUT_DIR / "merged_speedstars_baseline.ild"
    conv_base = ILDAConverter(
        galvo_config=galvo_config,
        simplify_tolerance=0.8,
        line_thickness=0.0,
        merge_close_distance=0.0,
    )
    stats_base = conv_base.convert_svg(
        str(merged_svg),
        str(base_ild),
        frame_name="SPD_BASE",
        write_both=True,
    )
    data_base = read_ilda_file(str(base_ild))
    
    # Stage 3B: Fully Optimized (Ribbon collapse + close-line merge + text outline control + TSP)
    opt_ild = OUTPUT_DIR / "merged_speedstars.ild"
    conv_opt = ILDAConverter(
        galvo_config=galvo_config,
        simplify_tolerance=1.0,
        line_thickness=25.0,
        merge_close_distance=18.0,
        text_roi="auto",
        text_tolerance=0.5,
        bg_tolerance=3.5,
        text_line_thickness=0.0,
        bg_line_thickness=25.0,
    )
    stats_opt = conv_opt.convert_svg(
        str(merged_svg),
        str(opt_ild),
        frame_name="SPD_OPT",
        write_both=True,
    )
    data_opt = read_ilda_file(str(opt_ild))
    
    # Copy final ILDA files to Downloads
    downloads_ild = DOWNLOADS_DIR / "merged_speedstars.ild"
    downloads_ilda = DOWNLOADS_DIR / "merged_speedstars.ilda"
    shutil.copy2(opt_ild, downloads_ild)
    shutil.copy2(opt_ild.with_suffix(".ilda"), downloads_ilda)
    
    # 4. Render Laser Previews
    print("\n[5/5] Generating authentic laser simulation previews...")
    img_base = render_ilda_preview(
        data_base,
        size=900,
        title_text="MERGED SPEEDSTARS - BASELINE (NO MERGE)",
        subtitle_text=(
            f"Paths: {stats_base['path_count']} | Points: {stats_base['total_points']} | "
            f"Blanking Jump: {stats_base['blanking_distance']:.1f} | Util: {stats_base['budget_utilization']:.1%}"
        ),
    )
    img_opt = render_ilda_preview(
        data_opt,
        size=900,
        title_text="MERGED SPEEDSTARS - FULLY OPTIMIZED",
        subtitle_text=(
            f"Paths: {stats_opt['path_count']} | Points: {stats_opt['total_points']} | "
            f"Blanking Jump: {stats_opt['blanking_distance']:.1f} | Util: {stats_opt['budget_utilization']:.1%}"
        ),
    )
    
    preview_base_path = OUTPUT_DIR / "merged_speedstars_baseline_preview.png"
    preview_opt_path = OUTPUT_DIR / "merged_speedstars_preview.png"
    img_base.save(preview_base_path)
    img_opt.save(preview_opt_path)
    
    # Copy preview to Downloads
    shutil.copy2(preview_opt_path, DOWNLOADS_DIR / "merged_speedstars_preview.png")
    
    # Contact sheet: Side-by-Side comparison
    W, H = img_base.size
    sbs = Image.new("RGB", (W * 2 + 30, H + 80), color=(15, 17, 23))
    draw_s = ImageDraw.Draw(sbs)
    draw_s.text((30, 20), "MERGED SPEEDSTARS: Baseline vs Optimized Comparison", fill=(255, 255, 255))
    draw_s.text(
        (30, 48),
        f"Baseline: {stats_base['total_points']} pts, {stats_base['blanking_distance']:.0f} jump | "
        f"Optimized: {stats_opt['total_points']} pts, {stats_opt['blanking_distance']:.0f} jump (-{(stats_base['blanking_distance'] - stats_opt['blanking_distance']) / stats_base['blanking_distance']:.1%} blanking)",
        fill=(0, 230, 200),
    )
    sbs.paste(img_base, (10, 70))
    sbs.paste(img_opt, (W + 20, 70))
    sbs_path = OUTPUT_DIR / "merged_speedstars_side_by_side.png"
    sbs.save(sbs_path)
    
    # Detail zoom sheet
    crop_b = crop_detail_region(img_base)
    crop_o = crop_detail_region(img_opt)
    cw, ch = crop_b.size
    zoom = Image.new("RGB", (cw * 2 + 30, ch + 80), color=(15, 17, 23))
    draw_z = ImageDraw.Draw(zoom)
    draw_z.text((30, 20), "ZOOM DETAIL: Stroke & Letterform Inspection", fill=(255, 255, 255))
    draw_z.text((30, 48), "Left: Baseline (Direct trace) | Right: Fully Optimized (Ribbon collapse + close merge + TSP route)", fill=(0, 230, 200))
    zoom.paste(crop_b, (10, 70))
    zoom.paste(crop_o, (cw + 20, 70))
    zoom_path = OUTPUT_DIR / "merged_speedstars_detail_zoom.png"
    zoom.save(zoom_path)
    
    # Summary report
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Baseline':<15} | {'Fully Optimized':<15} | {'Improvement'}")
    print("-" * 80)
    print(f"{'Total Paths':<25} | {stats_base['path_count']:<15} | {stats_opt['path_count']:<15} | --")
    print(f"{'Total Points':<25} | {stats_base['total_points']:<15} | {stats_opt['total_points']:<15} | {stats_opt['total_points'] - stats_base['total_points']:+d} pts")
    print(f"{'Beam-On Points':<25} | {stats_base['beam_on_points']:<15} | {stats_opt['beam_on_points']:<15} | {stats_opt['beam_on_points'] - stats_base['beam_on_points']:+d} pts")
    print(f"{'Blanked Points':<25} | {stats_base['beam_off_points']:<15} | {stats_opt['beam_off_points']:<15} | {stats_opt['beam_off_points'] - stats_base['beam_off_points']:+d} pts")
    print(f"{'Blanking Jump Distance':<25} | {stats_base['blanking_distance']:<15.1f} | {stats_opt['blanking_distance']:<15.1f} | -{(stats_base['blanking_distance'] - stats_opt['blanking_distance']) / stats_base['blanking_distance']:.1%}")
    print(f"{'Budget Utilization':<25} | {stats_base['budget_utilization']:<15.1%} | {stats_opt['budget_utilization']:<15.1%} | Optimal for 30k PPS")
    print("=" * 80)
    print(f"\nDeliverable Files Saved to Downloads:")
    print(f"  Merged SVG:     {downloads_svg}")
    print(f"  ILDA (.ild):    {downloads_ild} ({downloads_ild.stat().st_size:,} bytes)")
    print(f"  ILDA (.ilda):   {downloads_ilda} ({downloads_ilda.stat().st_size:,} bytes)")
    print(f"  Preview Image:  {DOWNLOADS_DIR / 'merged_speedstars_preview.png'}")
    print(f"\nAll comparison and intermediate files preserved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()

