"""
Generate ILDA files across a range of line thickness thresholds
with visual laser preview renders and comparison grids.
"""

import os
import shutil
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
from ilda_converter.path_optimizer import PathOptimizer


def render_ilda_preview(
    data: Dict,
    size: int = 800,
    show_blanked: bool = True,
    glow: bool = True,
    title_text: str = "",
    subtitle_text: str = "",
    default_color: Tuple[int, int, int] = (0, 255, 255),
) -> Image.Image:
    """
    Render an ILDA frame into an authentic laser projector preview image.
    """
    pts = data["points"]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    glow_canvas = np.zeros_like(canvas)

    # Margins and coordinate mapping
    pad = 70
    usable = size - (pad * 2)

    def to_screen(x: int, y: int) -> Tuple[int, int]:
        sx = int((x + 32768) / 65535 * usable + pad)
        sy = int((32767 - y) / 65535 * usable + pad)
        return sx, sy

    if len(pts) > 1:
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            s0 = to_screen(p0.x, p0.y)
            s1 = to_screen(p1.x, p1.y)

            if not p1.blanked:
                # RGB in ILDAPoint is 0..255. BGR for OpenCV
                r, g, b = (p1.r, p1.g, p1.b)
                if r == 0 and g == 0 and b == 0:
                    r, g, b = default_color
                bgr = (int(b), int(g), int(r))

                # Thick bloom layer
                cv2.line(glow_canvas, s0, s1, bgr, 5, cv2.LINE_AA)
                # Intense white-tinted core
                core_bgr = (min(255, b + 180), min(255, g + 180), min(255, r + 180))
                cv2.line(canvas, s0, s1, core_bgr, 1, cv2.LINE_AA)
            elif show_blanked:
                # Faint dotted/dashed blanking travel jump
                cv2.line(canvas, s0, s1, (45, 45, 60), 1, cv2.LINE_AA)

    # Blur glow layer and combine
    if glow:
        blurred = cv2.GaussianBlur(glow_canvas, (11, 11), 0)
        result = cv2.addWeighted(canvas, 1.0, blurred, 0.9, 0)
        result = cv2.addWeighted(result, 1.0, glow_canvas, 0.5, 0)
    else:
        result = cv2.addWeighted(canvas, 1.0, glow_canvas, 1.0, 0)

    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    # Draw header overlay if requested
    if title_text:
        # Background bar
        draw.rectangle([0, 0, size, 56], fill=(16, 20, 28, 220))
        draw.text((20, 10), title_text, fill=(255, 255, 255))
        if subtitle_text:
            draw.text((20, 32), subtitle_text, fill=(160, 180, 200))

    return pil_img


def build_comparison_grid(
    images_with_labels: List[Tuple[Image.Image, str, str]],
    cols: int = 4,
    tile_size: int = 500,
) -> Image.Image:
    """
    Assemble preview images into a labeled contact sheet grid.
    """
    n = len(images_with_labels)
    rows = (n + cols - 1) // cols
    card_header = 45
    card_w = tile_size
    card_h = tile_size + card_header

    grid_w = cols * card_w
    grid_h = rows * card_h

    grid_img = Image.new("RGB", (grid_w, grid_h), (12, 16, 22))
    draw = ImageDraw.Draw(grid_img)

    for idx, (img, title, subtitle) in enumerate(images_with_labels):
        r = idx // cols
        c = idx % cols
        x = c * card_w
        y = r * card_h

        # Resize preview tile
        resized = img.resize((card_w, tile_size), Image.Resampling.LANCZOS)
        grid_img.paste(resized, (x, y + card_header))

        # Tile card header background
        draw.rectangle([x, y, x + card_w, y + card_header], fill=(22, 28, 38))
        draw.rectangle([x, y, x + card_w, y + card_h], outline=(35, 45, 60), width=1)

        # Tile labels
        draw.text((x + 12, y + 6), title, fill=(0, 255, 240))
        draw.text((x + 12, y + 24), subtitle, fill=(170, 190, 210))

    return grid_img


def run_thickness_sweep(
    input_file: str,
    output_dir: Path,
    thresholds: List[float],
    color: Tuple[int, int, int],
    is_svg: bool,
    dataset_name: str,
) -> List[Dict]:
    """Run conversion and generate previews for all thickness values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    grid_items = []

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

    print(f"\nProcessing {dataset_name} across {len(thresholds)} thickness values...")
    print("=" * 65)

    for t in thresholds:
        t_str = f"{int(t):03d}" if t.is_integer() else f"{t:04.1f}".replace(".", "_")
        base_name = f"{dataset_name}_thick_{t_str}"
        ild_path = output_dir / f"{base_name}.ild"
        preview_path = output_dir / f"{base_name}_preview.png"
        clean_path = output_dir / f"{base_name}_clean.png"

        converter = ILDAConverter(
            vectorizer_method="vtracer",
            galvo_config=galvo_config,
            line_thickness=t,
        )

        if is_svg:
            stats = converter.convert_svg(
                input_file,
                str(ild_path),
                color=color,
                frame_name=f"THK_{int(t):03d}"[:8],
                company_name="ILDA_CVT",
                write_both=True,
                line_thickness=t,
            )
        else:
            stats = converter.convert_raster(
                input_file,
                str(ild_path),
                color=color,
                preprocess=False,
                frame_name=f"THK_{int(t):03d}"[:8],
                company_name="ILDA_CVT",
                write_both=True,
                line_thickness=t,
            )

        # Read back ILDA data for verification & preview rendering
        ilda_data = read_ilda_file(str(ild_path))
        pts_count = ilda_data["total_points"]
        beam_on = ilda_data["beam_on_count"]
        blanked = ilda_data["blanked_count"]
        paths_count = stats["path_count"]
        budget_pct = (pts_count / galvo_config.pps * galvo_config.target_fps) * 100

        # Titles for preview overlay
        if t == 0:
            status_desc = "Baseline (No Collapse / All Outlines)"
        elif t <= 15:
            status_desc = "Subtle Collapse (Thin Streaks Only)"
        elif t <= 35:
            status_desc = "Balanced (Streaks Centerline / Stars Outline)"
        else:
            status_desc = "Aggressive (Wide Elements Collapsed)"

        title = f"Thickness: {t:.1f} px  |  {status_desc}"
        subtitle = f"Paths: {paths_count}  |  Points: {pts_count} (On: {beam_on}, Blank: {blanked})  |  Budget: {budget_pct:.1f}%"

        # 1. Full laser preview with blanked jumps
        preview_img = render_ilda_preview(
            ilda_data,
            size=800,
            show_blanked=True,
            glow=True,
            title_text=title,
            subtitle_text=subtitle,
            default_color=color,
        )
        preview_img.save(preview_path)

        # 2. Clean laser beam preview (pure projection)
        clean_img = render_ilda_preview(
            ilda_data,
            size=800,
            show_blanked=False,
            glow=True,
            title_text=title,
            subtitle_text=subtitle,
            default_color=color,
        )
        clean_img.save(clean_path)

        # Add to grid items
        grid_items.append((
            clean_img,
            f"Thickness = {t:.1f} px",
            f"{pts_count} pts  |  {paths_count} paths  |  {status_desc}",
        ))

        rec = {
            "thickness": t,
            "status": status_desc,
            "paths": paths_count,
            "points": pts_count,
            "beam_on": beam_on,
            "blanked": blanked,
            "budget_pct": budget_pct,
            "ild_file": f"{base_name}.ild",
            "ilda_file": f"{base_name}.ilda",
            "preview_png": f"{base_name}_preview.png",
            "clean_png": f"{base_name}_clean.png",
        }
        results.append(rec)
        print(f"  t={t:5.1f} px: {paths_count:2d} paths | {pts_count:4d} pts | Budget: {budget_pct:5.1f}% | {status_desc}")

    # Build and save comparison contact sheet
    grid_img = build_comparison_grid(grid_items, cols=4, tile_size=480)
    grid_path = output_dir / "comparison_grid.png"
    grid_img.save(grid_path)
    print(f"\n[OK] Contact Sheet saved: {grid_path}")

    return results


def generate_html_dashboard(
    output_dir: Path,
    svg_results: List[Dict],
    jpg_results: List[Dict],
):
    """Generate an interactive HTML comparison dashboard."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ILDA Line Thickness Comparison Dashboard</title>
<style>
  :root {{
    --bg-dark: #0a0e17;
    --card-bg: #141a29;
    --card-border: #232c40;
    --accent-cyan: #00f0ff;
    --accent-green: #00ff88;
    --text-main: #f0f4fc;
    --text-muted: #8c9ba5;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-dark);
    color: var(--text-main);
    margin: 0;
    padding: 24px;
  }}
  header {{
    text-align: center;
    margin-bottom: 30px;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 20px;
  }}
  h1 {{
    margin: 0 0 10px 0;
    color: var(--accent-cyan);
    font-size: 28px;
    letter-spacing: -0.5px;
  }}
  p.desc {{
    color: var(--text-muted);
    font-size: 15px;
    margin: 0;
  }}
  .tabs {{
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-bottom: 24px;
  }}
  .tab-btn {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    color: var(--text-main);
    padding: 10px 24px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 15px;
    font-weight: 600;
    transition: all 0.2s ease;
  }}
  .tab-btn.active {{
    background: var(--accent-cyan);
    color: #000;
    border-color: var(--accent-cyan);
  }}
  .tab-content {{
    display: none;
  }}
  .tab-content.active {{
    display: block;
  }}
  .controls {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    padding: 14px 20px;
    border-radius: 8px;
    margin-bottom: 24px;
  }}
  .toggle-btn {{
    background: #252e42;
    border: 1px solid var(--card-border);
    color: var(--text-main);
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
  }}
  .toggle-btn.active {{
    background: var(--accent-green);
    color: #000;
    font-weight: 600;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
  }}
  .card:hover {{
    transform: translateY(-2px);
    border-color: var(--accent-cyan);
  }}
  .card-header {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--card-border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .card-title {{
    font-weight: 700;
    font-size: 16px;
    color: var(--accent-cyan);
  }}
  .badge {{
    font-size: 12px;
    padding: 3px 8px;
    border-radius: 4px;
    background: #1e283d;
    color: var(--text-muted);
  }}
  .badge.recommended {{
    background: #004d33;
    color: #00ff88;
    border: 1px solid #00aa66;
  }}
  .card-img {{
    width: 100%;
    aspect-ratio: 1;
    object-fit: contain;
    background: #060910;
    display: block;
    cursor: zoom-in;
  }}
  .card-body {{
    padding: 14px 16px;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 6px;
  }}
  .stat-label {{
    color: var(--text-muted);
  }}
  .stat-val {{
    font-weight: 600;
  }}
  .downloads {{
    display: flex;
    gap: 8px;
    margin-top: 12px;
  }}
  .dl-btn {{
    flex: 1;
    text-align: center;
    background: #1d2538;
    color: var(--text-main);
    text-decoration: none;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 0;
    border-radius: 4px;
    border: 1px solid var(--card-border);
    transition: background 0.15s ease;
  }}
  .dl-btn:hover {{
    background: #2b3652;
    color: var(--accent-cyan);
  }}
  .overview-banner {{
    margin-bottom: 24px;
    text-align: center;
  }}
  .overview-banner img {{
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid var(--card-border);
  }}
</style>
</head>
<body>

<header>
  <h1>ILDA Line Thickness Comparison Dashboard</h1>
  <p class="desc">Interactive evaluation of ribbon outline collapsing into single medial centerlines for flicker-free laser projection</p>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('svg-tab', this)">speedsvg.svg (SVG Direct)</button>
  <button class="tab-btn" onclick="showTab('jpg-tab', this)">speedstars.jpg (Raster Vectorized)</button>
</div>

<!-- SVG TAB -->
<div id="svg-tab" class="tab-content active">
  <div class="controls">
    <div><strong>Input:</strong> speedsvg.svg &nbsp;|&nbsp; <strong>Default PPS:</strong> 30,000 &nbsp;|&nbsp; <strong>Budget:</strong> 1,000 pts</div>
    <div>
      <button class="toggle-btn active" id="svg-toggle-blanked" onclick="toggleBlanked('svg')">Toggle Transit Jumps (Off)</button>
    </div>
  </div>

  <div class="overview-banner">
    <h3>All-in-One Visual Contact Sheet</h3>
    <a href="speedsvg/comparison_grid.png" target="_blank">
      <img src="speedsvg/comparison_grid.png" alt="SVG Comparison Grid">
    </a>
  </div>

  <div class="grid">
"""

    def render_cards(results: List[Dict], folder: str) -> str:
        cards_html = ""
        for r in results:
            t = r["thickness"]
            is_rec = (20.0 <= t <= 30.0)
            rec_badge = '<span class="badge recommended">Recommended</span>' if is_rec else f'<span class="badge">{r["status"].split()[0]}</span>'
            cards_html += f"""
    <div class="card">
      <div class="card-header">
        <span class="card-title">Thickness: {t:.1f} px</span>
        {rec_badge}
      </div>
      <img class="card-img" data-clean="{folder}/{r['clean_png']}" data-full="{folder}/{r['preview_png']}" src="{folder}/{r['clean_png']}" onclick="window.open(this.src)" title="Click to view full size">
      <div class="card-body">
        <div class="stat-row"><span class="stat-label">Paths:</span><span class="stat-val">{r['paths']}</span></div>
        <div class="stat-row"><span class="stat-label">Total Points:</span><span class="stat-val">{r['points']}</span></div>
        <div class="stat-row"><span class="stat-label">Beam On / Blanked:</span><span class="stat-val">{r['beam_on']} / {r['blanked']}</span></div>
        <div class="stat-row"><span class="stat-label">Budget Usage:</span><span class="stat-val">{r['budget_pct']:.1f}%</span></div>
        <div class="stat-row"><span class="stat-label">Status:</span><span class="stat-val">{r['status']}</span></div>
        <div class="downloads">
          <a class="dl-btn" href="{folder}/{r['ild_file']}" download>Download .ild (Beyond)</a>
          <a class="dl-btn" href="{folder}/{r['ilda_file']}" download>Download .ilda</a>
        </div>
      </div>
    </div>
"""
        return cards_html

    html_content += render_cards(svg_results, "speedsvg")
    html_content += """
  </div>
</div>

<!-- JPG TAB -->
<div id="jpg-tab" class="tab-content">
  <div class="controls">
    <div><strong>Input:</strong> speedstars.jpg &nbsp;|&nbsp; <strong>Vectorizer:</strong> VTracer &nbsp;|&nbsp; <strong>Budget:</strong> 1,000 pts</div>
    <div>
      <button class="toggle-btn active" id="jpg-toggle-blanked" onclick="toggleBlanked('jpg')">Toggle Transit Jumps (Off)</button>
    </div>
  </div>

  <div class="overview-banner">
    <h3>All-in-One Visual Contact Sheet</h3>
    <a href="speedstars/comparison_grid.png" target="_blank">
      <img src="speedstars/comparison_grid.png" alt="JPG Comparison Grid">
    </a>
  </div>

  <div class="grid">
"""
    html_content += render_cards(jpg_results, "speedstars")
    html_content += """
  </div>
</div>

<script>
function showTab(tabId, btn) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');
}

let showBlankedMap = { 'svg': false, 'jpg': false };

function toggleBlanked(tabPrefix) {
  showBlankedMap[tabPrefix] = !showBlankedMap[tabPrefix];
  const isBlanked = showBlankedMap[tabPrefix];
  const btn = document.getElementById(tabPrefix + '-toggle-blanked');
  btn.textContent = isBlanked ? 'Toggle Transit Jumps (On)' : 'Toggle Transit Jumps (Off)';
  btn.classList.toggle('active', isBlanked);

  const container = document.getElementById(tabPrefix + '-tab');
  container.querySelectorAll('.card-img').forEach(img => {
    img.src = isBlanked ? img.dataset.full : img.dataset.clean;
  });
}
</script>

</body>
</html>
"""
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Dashboard generated: {output_dir / 'index.html'}")


def main():
    svg_input = r"C:\Users\zubair\Downloads\speedsvg.svg"
    jpg_input = r"C:\Users\zubair\Downloads\speedstars.jpg"

    downloads_base = Path(r"C:\Users\zubair\Downloads\ilda_thickness_comparison")
    workspace_base = Path(r"output_comparison")

    downloads_base.mkdir(parents=True, exist_ok=True)
    workspace_base.mkdir(parents=True, exist_ok=True)

    thresholds = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0]

    # 1. Sweep speedsvg.svg
    svg_dir_dl = downloads_base / "speedsvg"
    svg_results = run_thickness_sweep(
        input_file=svg_input,
        output_dir=svg_dir_dl,
        thresholds=thresholds,
        color=(0, 255, 255),  # Cyan
        is_svg=True,
        dataset_name="speedsvg",
    )

    # 2. Sweep speedstars.jpg
    jpg_dir_dl = downloads_base / "speedstars"
    jpg_results = run_thickness_sweep(
        input_file=jpg_input,
        output_dir=jpg_dir_dl,
        thresholds=thresholds,
        color=(255, 255, 0),  # Yellow
        is_svg=False,
        dataset_name="speedstars",
    )

    # 3. Generate HTML dashboard in Downloads folder
    generate_html_dashboard(downloads_base, svg_results, jpg_results)

    # 4. Mirror everything into workspace_base
    print("\nMirroring comparison files to workspace output_comparison/...")
    if workspace_base.exists():
        shutil.rmtree(workspace_base)
    shutil.copytree(downloads_base, workspace_base)
    print(f"[OK] Mirrored to {workspace_base.resolve()}")

    print("\n" + "=" * 65)
    print("ALL THICKNESS SWEEPS COMPLETE!")
    print(f"Comparison folder: {downloads_base.resolve()}")
    print(f"Dashboard:         {downloads_base / 'index.html'}")
    print("=" * 65)


if __name__ == "__main__":
    main()

