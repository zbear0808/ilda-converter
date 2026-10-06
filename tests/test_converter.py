"""
Tests for full ILDA converter pipeline, normalization, and galvo conditioning.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw

from ilda_converter import (
    ILDAConverter,
    convert_image,
    convert_svg,
    GalvoConfig,
    normalize_coordinates,
    read_ilda_file,
    write_ilda_file,
)
from ilda_converter.galvo_conditioner import GalvoConditioner


def test_normalize_coordinates():
    """Test coordinate normalization and Y-axis inversion."""
    # Simple rectangle from (0, 0) to (100, 50)
    path = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0], [0.0, 0.0]])
    norm_paths = normalize_coordinates([path], target_range=(-32768, 32767), margin=0.05, invert_y=True)

    assert len(norm_paths) == 1
    p = norm_paths[0]

    # Verify all coordinates within ILDA range
    assert np.all(p[:, 0] >= -32768) and np.all(p[:, 0] <= 32767)
    assert np.all(p[:, 1] >= -32768) and np.all(p[:, 1] <= 32767)

    # Inverted Y: initial (0, 0) top-left should have higher Y in ILDA than (0, 50) bottom-left
    # p[0] is (0, 0), p[2] is (100, 50)
    assert p[0, 1] > p[2, 1]


def test_galvo_conditioner_budget():
    """Test that galvo conditioner generates points close to budget."""
    cfg = GalvoConfig(pps=30000, target_fps=30)  # budget = 1000 points
    conditioner = GalvoConditioner(cfg)

    # A square path in ILDA coordinates
    path = np.array([
        [-20000.0, -20000.0],
        [20000.0, -20000.0],
        [20000.0, 20000.0],
        [-20000.0, 20000.0],
        [-20000.0, -20000.0],
    ])

    points = conditioner.condition_paths([path], color=(0, 255, 0))
    # Points should be reasonably close to budget (within 15%)
    assert 850 <= len(points) <= 1150

    # Verify beam is active during drawing
    beam_on = [pt for pt in points if not pt["blanked"]]
    assert len(beam_on) > 500


def test_full_pipeline_roundtrip():
    """Test full conversion of a synthetic raster image and read back."""
    # Create test image
    img = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 80, 80], fill="black")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_img:
        img_path = f_img.name
    with tempfile.NamedTemporaryFile(suffix=".ilda", delete=False) as f_ilda:
        ilda_path = f_ilda.name

    try:
        img.save(img_path)
        stats = convert_image(
            img_path,
            ilda_path,
            vectorizer="centerline",
            pps=30000,
            fps=30,
            preprocess=True,
        )

        assert stats["total_points"] > 0
        assert stats["output_file"] == ilda_path

        # Read back and verify ILDA structure
        data = read_ilda_file(ilda_path)
        assert data["header"]["format_code"] == 5
        assert data["header"]["point_count"] == stats["total_points"]
        assert len(data["points"]) == stats["total_points"]
        assert data["beam_on_count"] == stats["beam_on_points"]

    finally:
        Path(img_path).unlink(missing_ok=True)
        Path(ilda_path).unlink(missing_ok=True)


def test_partition_paths_auto_and_custom():
    """Test auto-detection and explicit partitioning of text vs background paths."""
    from ilda_converter.path_optimizer import PathOptimizer

    bg_path = np.array([
        [0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0], [0.0, 0.0]
    ])

    letter1 = np.array([[200.0, 450.0], [250.0, 450.0], [250.0, 550.0], [200.0, 550.0], [200.0, 450.0]])
    letter2 = np.array([[300.0, 450.0], [350.0, 450.0], [350.0, 550.0], [300.0, 550.0], [300.0, 450.0]])
    letter3 = np.array([[400.0, 450.0], [450.0, 450.0], [450.0, 550.0], [400.0, 550.0], [400.0, 450.0]])

    all_paths = [bg_path, letter1, letter2, letter3]

    text_paths, bg_paths, roi = PathOptimizer.partition_paths(all_paths, text_roi="auto")
    assert len(text_paths) == 3
    assert len(bg_paths) == 1
    assert roi is not None
    assert roi[0] < 0.5 < roi[1]

    text_paths_exp, bg_paths_exp, _ = PathOptimizer.partition_paths(all_paths, text_roi=(0.4, 0.6))
    assert len(text_paths_exp) == 3
    assert len(bg_paths_exp) == 1

    text_paths_none, bg_paths_none, _ = PathOptimizer.partition_paths(all_paths, text_roi=None)
    assert len(text_paths_none) == 0
    assert len(bg_paths_none) == 4


def test_segmented_optimization():
    """Test that segmented optimization simplifies background more aggressively than text."""
    from ilda_converter.path_optimizer import PathOptimizer

    x_bg = np.linspace(0, 1000, 100)
    y_bg = 100.0 + 1.0 * np.sin(x_bg)
    bg_path = np.column_stack([x_bg, y_bg])

    t = np.linspace(0, 2 * np.pi, 50)
    letter = np.column_stack([500.0 + 30.0 * np.cos(t), 500.0 + 30.0 * np.sin(t)])
    letter2 = np.column_stack([600.0 + 30.0 * np.cos(t), 500.0 + 30.0 * np.sin(t)])

    optimizer = PathOptimizer(
        text_roi="auto",
        text_tolerance=0.2,
        bg_tolerance=5.0,
        text_line_thickness=0.0,
        bg_line_thickness=0.0,
    )

    opt_paths, _ = optimizer.optimize_paths([bg_path, letter, letter2])
    assert len(opt_paths) >= 2

    bg_out = [p for p in opt_paths if p[:, 1].max() < 200.0][0]
    assert len(bg_out) < 20


def test_merge_close_paths():
    """Test merging nearby vector lines into single centerlines."""
    from ilda_converter.path_optimizer import merge_close_paths

    # Two parallel lines 6 units apart
    x = np.linspace(0, 100, 40)
    l1 = np.column_stack([x, np.zeros_like(x)])
    l2 = np.column_stack([x, np.ones_like(x) * 6.0])

    # A third line far away (50 units apart)
    l3 = np.column_stack([x, np.ones_like(x) * 50.0])

    merged = merge_close_paths([l1, l2, l3], max_distance=10.0)
    assert len(merged) == 2  # l1 and l2 merged into 1, l3 remains separate

    # The merged line should have mean Y ~ 3.0
    merged_y = [p[:, 1].mean() for p in merged]
    assert any(abs(y - 3.0) < 0.2 for y in merged_y)
    assert any(abs(y - 50.0) < 0.2 for y in merged_y)


def test_optimizer_with_close_line_merging():
    """Test that PathOptimizer integrates close line merging into its pipeline."""
    from ilda_converter.path_optimizer import PathOptimizer

    x = np.linspace(0, 100, 30)
    l1 = np.column_stack([x, np.zeros_like(x)])
    l2 = np.column_stack([x, np.ones_like(x) * 4.0])

    # Without merge_close_distance: 2 paths
    opt_no_merge = PathOptimizer(tolerance=0.1, merge_close_distance=0.0, max_line_thickness=0.0)
    paths_no_merge, _ = opt_no_merge.optimize_paths([l1, l2])
    assert len(paths_no_merge) == 2

    # With merge_close_distance: 1 path
    opt_merge = PathOptimizer(tolerance=0.1, merge_close_distance=8.0, max_line_thickness=0.0)
    paths_merge, _ = opt_merge.optimize_paths([l1, l2])
    assert len(paths_merge) == 1

