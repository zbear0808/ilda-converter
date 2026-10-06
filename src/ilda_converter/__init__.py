"""
ILDA Converter: Convert raster images to optimized ILDA laser projector files.

Main pipeline:
    1. Vectorization (vtracer / potrace / centerline)
    2. Path optimization (TSP via vpype)
    3. Galvo conditioning (point budgeting, resampling, corner dwells, blanking)
    4. ILDA Format 5 binary generation
"""

from .pipeline import ILDAConverter, convert_image, convert_svg
from .ilda_writer import ILDAWriter, ILDAPoint, write_ilda_file, read_ilda_file
from .vectorizer import Vectorizer, preprocess_image
from .path_optimizer import PathOptimizer, merge_close_paths
from .galvo_conditioner import GalvoConditioner, GalvoConfig, normalize_coordinates

__version__ = "0.1.0"

__all__ = [
    "ILDAConverter",
    "convert_image",
    "convert_svg",
    "ILDAWriter",
    "ILDAPoint",
    "write_ilda_file",
    "read_ilda_file",
    "Vectorizer",
    "preprocess_image",
    "PathOptimizer",
    "merge_close_paths",
    "GalvoConditioner",
    "GalvoConfig",
    "normalize_coordinates",
]

