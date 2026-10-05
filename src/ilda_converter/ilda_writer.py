"""
ILDA Format 5 (2D True Color) binary file writer.

The ILDA format is the industry standard for laser show projectors.
Format 5 supports full RGB color in 2D space.
"""

import struct
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ILDAPoint:
    """A single ILDA point with position, color, and blanking state."""
    
    x: int  # -32768 to 32767
    y: int  # -32768 to 32767
    blanked: bool
    r: int  # 0 to 255
    g: int  # 0 to 255
    b: int  # 0 to 255

    def __post_init__(self):
        """Validate point data."""
        if not (-32768 <= self.x <= 32767):
            raise ValueError(f"X coordinate {self.x} out of range [-32768, 32767]")
        if not (-32768 <= self.y <= 32767):
            raise ValueError(f"Y coordinate {self.y} out of range [-32768, 32767]")
        if not (0 <= self.r <= 255):
            raise ValueError(f"Red value {self.r} out of range [0, 255]")
        if not (0 <= self.g <= 255):
            raise ValueError(f"Green value {self.g} out of range [0, 255]")
        if not (0 <= self.b <= 255):
            raise ValueError(f"Blue value {self.b} out of range [0, 255]")


class ILDAWriter:
    """Writer for ILDA Format 5 files."""

    FORMAT_CODE = 5
    HEADER_SIZE = 32
    POINT_SIZE = 8
    BLANKED_FLAG = 0x40
    BEAM_ON_FLAG = 0x00

    def __init__(
        self,
        frame_name: str = "FRAME",
        company_name: str = "CONVERTER",
    ):
        """
        Initialize ILDA writer.

        Args:
            frame_name: 8-character frame name (padded/truncated)
            company_name: 8-character company name (padded/truncated)
        """
        self.frame_name = self._pad_name(frame_name, 8)
        self.company_name = self._pad_name(company_name, 8)

    @staticmethod
    def _pad_name(name: str, length: int) -> bytes:
        """Pad or truncate name to exact length."""
        return name[:length].ljust(length).encode("ascii")

    def write_frame(
        self,
        filename: str,
        points: List[ILDAPoint],
        frame_number: int = 0,
        total_frames: int = 1,
    ):
        """
        Write a single ILDA frame to file.

        Args:
            filename: Output file path
            points: List of ILDA points
            frame_number: Index of this frame (0-based)
            total_frames: Total number of frames in file
        """
        if len(points) > 65535:
            raise ValueError(
                f"ILDA frame point count {len(points)} exceeds 65535 limit"
            )

        with open(filename, "wb") as f:
            # Write frame header
            self._write_header(
                f, len(points), frame_number, total_frames
            )

            # Write point data
            for point in points:
                self._write_point(f, point)

            # Write end-of-file header
            self._write_eof_header(f, total_frames)

    def append_frame(
        self,
        filename: str,
        points: List[ILDAPoint],
        frame_number: int,
        total_frames: int,
    ):
        """
        Append a frame to an existing ILDA file.
        
        Warning: This overwrites the EOF header. Use carefully.
        """
        if len(points) > 65535:
            raise ValueError(
                f"ILDA frame point count {len(points)} exceeds 65535 limit"
            )

        with open(filename, "r+b") as f:
            # Seek to EOF header and overwrite it
            f.seek(-self.HEADER_SIZE, 2)
            
            # Write new frame header
            self._write_header(
                f, len(points), frame_number, total_frames
            )

            # Write point data
            for point in points:
                self._write_point(f, point)

            # Write new EOF header
            self._write_eof_header(f, total_frames)

    def _write_header(
        self,
        f,
        num_points: int,
        frame_number: int,
        total_frames: int,
    ):
        """Write 32-byte ILDA frame header."""
        header = struct.pack(
            ">4s3sB8s8sHHHBB",
            b"ILDA",                  # Protocol identifier
            b"\x00\x00\x00",          # Reserved (3 bytes)
            self.FORMAT_CODE,          # Format 5: 2D True Color
            self.frame_name,           # Frame Name (8 bytes)
            self.company_name,         # Company Name (8 bytes)
            num_points,                # Point count in frame (uint16)
            frame_number,              # Frame number (uint16)
            total_frames,              # Total frames (uint16)
            0,                         # Scanner head (uint8)
            0,                         # Reserved (uint8)
        )
        f.write(header)

    def _write_point(self, f, point: ILDAPoint):
        """Write 8-byte ILDA point record."""
        status = self.BLANKED_FLAG if point.blanked else self.BEAM_ON_FLAG
        point_data = struct.pack(
            ">hhBBBB",
            point.x,
            point.y,
            status,
            point.r,
            point.g,
            point.b,
        )
        f.write(point_data)

    def _write_eof_header(self, f, total_frames: int):
        """Write end-of-file header with 0 point count."""
        eof_header = struct.pack(
            ">4s3sB8s8sHHHBB",
            b"ILDA",
            b"\x00\x00\x00",
            self.FORMAT_CODE,
            b"        ",  # Empty frame name
            b"        ",  # Empty company name
            0,            # 0 points = EOF marker
            0,
            total_frames,
            0,
            0,
        )
        f.write(eof_header)


def write_ilda_file(
    filename: str,
    points: List[Dict],
    frame_name: str = "FRAME",
    company_name: str = "CONVERTER",
):
    """
    Convenience function to write ILDA file from dict-based points.

    Args:
        filename: Output file path
        points: List of dicts with keys: x, y, blanked, r, g, b
        frame_name: 8-character frame name
        company_name: 8-character company name
    """
    ilda_points = [
        ILDAPoint(
            x=int(p["x"]),
            y=int(p["y"]),
            blanked=bool(p["blanked"]),
            r=int(p["r"]),
            g=int(p["g"]),
            b=int(p["b"]),
        )
        for p in points
    ]

    writer = ILDAWriter(frame_name, company_name)
    writer.write_frame(filename, ilda_points)


def read_ilda_file(filename: str) -> dict:
    """
    Read an ILDA Format 5 file and return its header and points.

    Args:
        filename: Path to the .ilda file

    Returns:
        Dict with keys:
            - 'header': dict of header fields
            - 'points': list of ILDAPoint instances
            - 'total_points': total count of points
            - 'beam_on_count': points with laser beam active
            - 'blanked_count': points with laser beam off
    """
    with open(filename, "rb") as f:
        data = f.read()

    if len(data) < 32:
        raise ValueError("File is too small to be a valid ILDA file")

    header_data = data[:32]
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
    ) = struct.unpack(">4s3sB8s8sHHHBB", header_data)

    if ilda_marker != b"ILDA":
        raise ValueError(f"Invalid ILDA marker: {ilda_marker}")

    points = []
    offset = 32
    for _ in range(num_points):
        if offset + 8 > len(data):
            break
        x, y, status, r, g, b = struct.unpack(">hhBBBB", data[offset : offset + 8])
        blanked = bool(status & 0x40)
        points.append(ILDAPoint(x=x, y=y, blanked=blanked, r=r, g=g, b=b))
        offset += 8

    beam_on = sum(1 for p in points if not p.blanked)
    blanked = len(points) - beam_on

    return {
        "header": {
            "format_code": format_code,
            "frame_name": frame_name.decode("ascii", errors="replace").strip(),
            "company_name": company_name.decode("ascii", errors="replace").strip(),
            "point_count": num_points,
            "frame_number": frame_num,
            "total_frames": total_frames,
        },
        "points": points,
        "total_points": len(points),
        "beam_on_count": beam_on,
        "blanked_count": blanked,
    }

