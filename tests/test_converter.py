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
