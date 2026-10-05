"""
Main ILDA conversion pipeline.

Orchestrates the complete conversion flow:
1. Vectorization
2. Path optimization (TSP)
3. Galvo conditioning
4. ILDA file generation
"""

from pathlib import Path
from typing import Optional, Tuple, Literal
import numpy as np

from .vectorizer import Vectorizer, VectorizerType, preprocess_image
from .path_optimizer import PathOptimizer
from .galvo_conditioner import GalvoConditioner, GalvoConfig, normalize_coordinates
from .ilda_writer import ILDAWriter, ILDAPoint


class ILDAConverter:
    """Complete raster-to-ILDA conversion pipeline."""

    def __init__(
        self,
        vectorizer_method: VectorizerType = "centerline",
        galvo_config: Optional[GalvoConfig] = None,
        simplify_tolerance: float = 1.0,
        path_merge_threshold: float = 0.5,
    ):
        """
        Initialize ILDA converter.

        Args:
            vectorizer_method: Vectorization strategy (default: centerline - works on all platforms)
            galvo_config: Galvo scanner configuration
            simplify_tolerance: Path simplification tolerance
            path_merge_threshold: Distance for merging adjacent segments
        """
        self.vectorizer = Vectorizer(
            method=vectorizer_method,
            simplify_tolerance=simplify_tolerance,
        )
        self.optimizer = PathOptimizer(
            tolerance=simplify_tolerance,
            merge_threshold=path_merge_threshold,
        )
        self.conditioner = GalvoConditioner(galvo_config)

    def convert(
        self,
        image_path: str,
        output_ilda_path: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        preprocess: bool = False,
        threshold_method: Literal["otsu", "adaptive", "manual"] = "otsu",
        frame_name: str = "IMAGE",
        company_name: str = "CONVERTER",
    ) -> dict:
        """
        Convert raster image to ILDA file.

        Args:
            image_path: Input image path
            output_ilda_path: Output .ilda file path
            color: RGB color tuple (0-255)
            preprocess: Apply preprocessing (threshold, cleanup)
            threshold_method: Thresholding method if preprocessing
            frame_name: ILDA frame name (8 chars)
            company_name: ILDA company name (8 chars)

        Returns:
            Dictionary with conversion statistics
        """
        print(f"Converting {image_path} to {output_ilda_path}")

        # Stage 1: Preprocess (optional)
        if preprocess:
            print("Stage 1: Preprocessing image...")
            preprocessed = Path(image_path).parent / f"_preprocessed_{Path(image_path).name}"
            preprocess_image(
                image_path,
                str(preprocessed),
                threshold_method=threshold_method,
            )
            image_path = str(preprocessed)

        # Stage 2: Vectorization
        print(f"Stage 2: Vectorizing ({self.vectorizer.method})...")
        paths, svg_path = self.vectorizer.vectorize(image_path)
        print(f"  → Extracted {len(paths)} paths")

        if not paths:
            raise ValueError("No paths extracted from image. Try adjusting preprocessing.")

        # Stage 3: Path Optimization (TSP)
        print("Stage 3: Optimizing path order (TSP)...")
        optimized_paths, blanking_distance = self.optimizer.optimize_paths(paths)
        print(f"  → Optimized to {len(optimized_paths)} paths")
        print(f"  → Total blanking distance: {blanking_distance:.1f} units")

        # Stage 4: Normalize to ILDA coordinate range
        print("Stage 4: Normalizing coordinates...")
        normalized_paths = normalize_coordinates(optimized_paths)

        # Stage 5: Galvo Conditioning
        print("Stage 5: Conditioning for galvo dynamics...")
        conditioned_points = self.conditioner.condition_paths(
            normalized_paths,
            color=color,
        )
        print(f"  → Generated {len(conditioned_points)} points")
        print(f"  → Point budget: {self.conditioner.max_points_per_frame} max")

        if len(conditioned_points) > self.conditioner.max_points_per_frame:
            print(
                f"  ⚠ WARNING: Point count exceeds budget! "
                f"({len(conditioned_points)} > {self.conditioner.max_points_per_frame})"
            )
            print("  Frame rate will be reduced or flicker may occur.")

        # Stage 6: Write ILDA File
        print("Stage 6: Writing ILDA file...")
        writer = ILDAWriter(frame_name=frame_name, company_name=company_name)
        
        ilda_points = [
            ILDAPoint(
                x=p["x"],
                y=p["y"],
                blanked=p["blanked"],
                r=p["r"],
                g=p["g"],
                b=p["b"],
            )
            for p in conditioned_points
        ]

        writer.write_frame(output_ilda_path, ilda_points)
        print(f"✓ Conversion complete: {output_ilda_path}")

        # Calculate statistics
        beam_on_points = sum(1 for p in conditioned_points if not p["blanked"])
        beam_off_points = len(conditioned_points) - beam_on_points

        stats = {
            "input_image": image_path,
            "output_file": output_ilda_path,
            "vectorizer": self.vectorizer.method,
            "path_count": len(optimized_paths),
            "total_points": len(conditioned_points),
            "beam_on_points": beam_on_points,
            "beam_off_points": beam_off_points,
            "blanking_distance": blanking_distance,
            "point_budget": self.conditioner.max_points_per_frame,
            "budget_utilization": len(conditioned_points) / self.conditioner.max_points_per_frame,
        }

        return stats


def convert_image(
    image_path: str,
    output_path: str,
    vectorizer: VectorizerType = "centerline",
    pps: int = 30000,
    fps: int = 30,
    color: Tuple[int, int, int] = (255, 255, 255),
    preprocess: bool = False,
) -> dict:
    """
    Convenience function for quick conversion.

    Args:
        image_path: Input image
        output_path: Output .ilda file
        vectorizer: Vectorization method
        pps: Galvo points per second
        fps: Target frame rate
        color: RGB color
        preprocess: Apply preprocessing

    Returns:
        Conversion statistics dictionary
    """
    galvo_config = GalvoConfig(pps=pps, target_fps=fps)

    converter = ILDAConverter(
        vectorizer_method=vectorizer,
        galvo_config=galvo_config,
    )

    return converter.convert(
        image_path,
        output_path,
        color=color,
        preprocess=preprocess,
    )
