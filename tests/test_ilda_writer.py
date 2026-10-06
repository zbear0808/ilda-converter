"""
Tests for ILDA Format 5 writer.
"""

import struct
import tempfile
from pathlib import Path
import pytest

from ilda_converter.ilda_writer import ILDAWriter, ILDAPoint, write_ilda_file


class TestILDAPoint:
    """Test ILDAPoint dataclass validation."""

    def test_valid_point(self):
        """Valid point should construct successfully."""
        point = ILDAPoint(x=0, y=0, blanked=False, r=255, g=128, b=64)
        assert point.x == 0
        assert point.y == 0
        assert point.r == 255
        assert point.g == 128
        assert point.b == 64

    def test_x_out_of_range(self):
        """X coordinate out of range should raise ValueError."""
        with pytest.raises(ValueError, match="X coordinate.*out of range"):
            ILDAPoint(x=40000, y=0, blanked=False, r=0, g=0, b=0)

    def test_color_out_of_range(self):
        """Color value out of range should raise ValueError."""
        with pytest.raises(ValueError, match="Red value.*out of range"):
            ILDAPoint(x=0, y=0, blanked=False, r=300, g=0, b=0)


class TestILDAWriter:
    """Test ILDA writer functionality."""

    def test_write_simple_frame(self):
        """Write a simple single-frame ILDA file."""
        writer = ILDAWriter(frame_name="TEST", company_name="PYTEST")

        points = [
            ILDAPoint(x=0, y=0, blanked=False, r=255, g=0, b=0),
            ILDAPoint(x=1000, y=1000, blanked=False, r=0, g=255, b=0),
            ILDAPoint(x=2000, y=0, blanked=False, r=0, g=0, b=255),
        ]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ilda") as f:
            output_path = f.name

        try:
            writer.write_frame(output_path, points)

            # Verify file exists and has correct size
            # Header (32) + 3 points (8 each) + EOF header (32) = 88 bytes
            assert Path(output_path).stat().st_size == 88

            # Read and verify header
            with open(output_path, "rb") as f:
                header = f.read(32)
                (
                    ilda_marker,
                    reserved,
                    format_code,
                    frame_name,
                    company_name,
                    num_points,
                    frame_num,
                    total_frames,
                    scanner_head,
                    reserved2,
                ) = struct.unpack(">4s3sB8s8sHHHBB", header)

                assert ilda_marker == b"ILDA"
                assert format_code == 5
                assert num_points == 3
                assert frame_num == 0
                assert total_frames == 1

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_point_limit(self):
        """Attempting to write >65535 points should raise ValueError."""
        writer = ILDAWriter()

        # Create too many points
        points = [
            ILDAPoint(x=0, y=0, blanked=False, r=255, g=255, b=255)
            for _ in range(65536)
        ]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ilda") as f:
            output_path = f.name

        try:
            with pytest.raises(ValueError, match="point count.*exceeds 65535"):
                writer.write_frame(output_path, points)
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_write_ilda_file_convenience(self):
        """Test convenience function with dict-based points."""
        points = [
            {"x": 0, "y": 0, "blanked": False, "r": 255, "g": 0, "b": 0},
            {"x": 5000, "y": 5000, "blanked": False, "r": 0, "g": 255, "b": 0},
        ]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ilda") as f:
            output_path = f.name

        try:
            write_ilda_file(output_path, points)
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_blanked_status_bit(self):
        """Verify blanking flag is correctly encoded."""
        writer = ILDAWriter()

        points = [
            ILDAPoint(x=0, y=0, blanked=True, r=0, g=0, b=0),  # Blanked
            ILDAPoint(x=100, y=100, blanked=False, r=255, g=255, b=255),  # On
        ]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ilda") as f:
            output_path = f.name

        try:
            writer.write_frame(output_path, points)

            with open(output_path, "rb") as f:
                f.seek(32)  # Skip header

                # Read first point
                point1_data = struct.unpack(">hhBBBB", f.read(8))
                assert point1_data[2] == 0x40  # Blanked flag

                # Read second point
                point2_data = struct.unpack(">hhBBBB", f.read(8))
                assert point2_data[2] == 0x00  # Beam on

        finally:
            Path(output_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
