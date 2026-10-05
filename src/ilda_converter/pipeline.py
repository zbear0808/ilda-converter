"""
Main ILDA conversion pipeline.

Orchestrates the complete 4-stage conversion flow:
1. Vectorization (Raster to SVG / Polylines)
2. Path Optimization (TSP shortest jumps, segment merge, collinear simplify)
3. Galvo Trajectory Conditioning (Point budgeting, uniform resampling, corner dwells, blanked jumps & settles)
4. ILDA Format 5 Binary Packaging (Big-endian header, 8-byte point records, EOF header)
"""

from pathlib import Path
from typing import Optional, Tuple, Literal, Dict
import numpy as np

from .vectorizer import Vectorizer, VectorizerType, preprocess_image
from .path_optimizer import PathOptimizer
from .galvo_conditioner import GalvoConditioner, GalvoConfig, normalize_coordinates
from .ilda_writer import ILDAWriter, ILDAPoint


class ILDAConverter:
    """Complete vector / raster to ILDA conversion pipeline."""

    def __init__(
        self,
        vectorizer_method: VectorizerType = "vtracer",
        galvo_config: Optional[GalvoConfig] = None,
        simplify_tolerance: float = 1.0,
        path_merge_threshold: float = 0.5,
    ):
        """
        Initialize ILDA converter.

        Args:
            vectorizer_method: Vectorization strategy ('vtracer', 'centerline', 'potrace')
            galvo_config: Galvo scanner configuration
            simplify_tolerance: Path simplification tolerance in mm (Ramer-Douglas-Peucker)
            path_merge_threshold: Distance for merging adjacent segments in mm
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
        input_path: str,
        output_ilda_path: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        preprocess: bool = False,
        threshold_method: Literal["otsu", "adaptive", "manual"] = "otsu",
        frame_name: str = "FRAME",
        company_name: str = "CONVERTER",
    ) -> Dict:
        """
        Convert raster image or SVG file to ILDA format.
        Automatically detects SVG files to skip rasterization.

        Args:
            input_path: Path to input file (JPG, PNG, BMP, or SVG)
            output_ilda_path: Path to write the .ilda file
            color: Default RGB color tuple (0-255)
            preprocess: Apply raster preprocessing (threshold, inversion)
            threshold_method: Thresholding method if preprocessing
            frame_name: ILDA frame name (up to 8 chars)
            company_name: ILDA company name (up to 8 chars)

        Returns:
            Dictionary with conversion statistics
        """
        path_obj = Path(input_path)
        if path_obj.suffix.lower() == ".svg":
            return self.convert_svg(
                input_path,
                output_ilda_path,
                color=color,
                frame_name=frame_name,
                company_name=company_name,
            )

        return self.convert_raster(
            input_path,
            output_ilda_path,
            color=color,
            preprocess=preprocess,
            threshold_method=threshold_method,
            frame_name=frame_name,
            company_name=company_name,
        )

    def convert_svg(
        self,
        svg_path: str,
        output_ilda_path: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        frame_name: str = "SVG_FRM",
        company_name: str = "CONVERTER",
    ) -> Dict:
        """
        Directly convert an SVG vector file to ILDA format.
        Bypasses rasterization and runs path optimization and galvo conditioning.
        """
        print(f"Converting SVG: {svg_path} -> {output_ilda_path}")

        # Stage 1: Path Optimization & TSP reordering
        print("Stage 1: Optimizing paths & TSP sorting (vpype)...")
        optimized_paths, blanking_distance = self.optimizer.optimize_svg(svg_path)
        print(f"  -> Extracted & optimized {len(optimized_paths)} paths")
        print(f"  -> Total blanking jump distance: {blanking_distance:.1f} units")

        if not optimized_paths:
            raise ValueError(f"No valid paths found in SVG: {svg_path}")

        # Stage 2: Normalize to ILDA coordinate space (-32768 to 32767)
        print("Stage 2: Normalizing coordinates to ILDA range (with Y-inversion)...")
        normalized_paths = normalize_coordinates(
            optimized_paths,
            invert_y=self.conditioner.config.invert_y,
            margin=self.conditioner.config.margin,
        )

        # Stage 3: Galvo Dynamics conditioning
        print("Stage 3: Conditioning for galvo dynamics...")
        conditioned_points = self.conditioner.condition_paths(
            normalized_paths,
            color=color,
        )
        print(f"  -> Generated {len(conditioned_points)} points")
        print(f"  -> Point budget: {self.conditioner.max_points_per_frame} max")

        # Stage 4: Write ILDA binary file
        print("Stage 4: Writing ILDA Format 5 binary file...")
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
        print(f"[OK] Conversion complete: {output_ilda_path}")

        beam_on_points = sum(1 for p in conditioned_points if not p["blanked"])
        beam_off_points = len(conditioned_points) - beam_on_points

        return {
            "input_file": svg_path,
            "output_file": output_ilda_path,
            "vectorizer": "svg_direct",
            "path_count": len(optimized_paths),
            "total_points": len(conditioned_points),
            "beam_on_points": beam_on_points,
            "beam_off_points": beam_off_points,
            "blanking_distance": blanking_distance,
            "point_budget": self.conditioner.max_points_per_frame,
            "budget_utilization": len(conditioned_points) / self.conditioner.max_points_per_frame,
        }

    def convert_raster(
        self,
        image_path: str,
        output_ilda_path: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        preprocess: bool = False,
        threshold_method: Literal["otsu", "adaptive", "manual"] = "otsu",
        frame_name: str = "IMAGE",
        company_name: str = "CONVERTER",
    ) -> Dict:
        """
        Convert raster image to ILDA format through full 4-stage pipeline.
        """
        print(f"Converting raster: {image_path} -> {output_ilda_path}")

        # Stage 1: Preprocess (optional thresholding / cleanup)
        active_image_path = image_path
        if preprocess:
            print("Stage 1a: Preprocessing raster image...")
            preprocessed = Path(image_path).parent / f"_preprocessed_{Path(image_path).name}"
            preprocess_image(
                image_path,
                str(preprocessed),
                threshold_method=threshold_method,
            )
            active_image_path = str(preprocessed)

        # Stage 1b: Vectorization (Raster to vector paths)
        print(f"Stage 1: Vectorizing image using {self.vectorizer.method}...")
        paths, svg_path = self.vectorizer.vectorize(active_image_path)
        print(f"  -> Extracted {len(paths)} raw paths")

        if not paths:
            raise ValueError("No paths extracted from image. Try adjusting preprocessing.")

        # Stage 2: Path Optimization (TSP reordering, segment merging, collinear simplify)
        print("Stage 2: Optimizing path order (TSP via vpype)...")
        optimized_paths, blanking_distance = self.optimizer.optimize_paths(paths)
        print(f"  -> Optimized to {len(optimized_paths)} paths")
        print(f"  -> Total blanking jump distance: {blanking_distance:.1f} units")

        # Stage 3: Normalize to ILDA coordinate range
        print("Stage 3: Normalizing coordinates to ILDA range (with Y-inversion)...")
        normalized_paths = normalize_coordinates(
            optimized_paths,
            invert_y=self.conditioner.config.invert_y,
            margin=self.conditioner.config.margin,
        )

        # Stage 4: Galvo Dynamics conditioning
        print("Stage 4: Conditioning for galvo dynamics...")
        conditioned_points = self.conditioner.condition_paths(
            normalized_paths,
            color=color,
        )
        print(f"  -> Generated {len(conditioned_points)} points")
        print(f"  -> Point budget: {self.conditioner.max_points_per_frame} max")

        if len(conditioned_points) > self.conditioner.max_points_per_frame * 1.25:
            print(
                f"  [WARNING] Point count exceeds budget! "
                f"({len(conditioned_points)} > {self.conditioner.max_points_per_frame})"
            )

        # Stage 5: Binary ILDA packaging
        print("Stage 5: Writing ILDA Format 5 binary file...")
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
        print(f"[OK] Conversion complete: {output_ilda_path}")

        beam_on_points = sum(1 for p in conditioned_points if not p["blanked"])
        beam_off_points = len(conditioned_points) - beam_on_points

        return {
            "input_file": image_path,
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


def convert_image(
    image_path: str,
    output_path: str,
    vectorizer: VectorizerType = "vtracer",
    pps: int = 30000,
    fps: int = 30,
    color: Tuple[int, int, int] = (255, 255, 255),
    preprocess: bool = False,
) -> Dict:
    """
    Convenience function to convert a raster image or SVG to ILDA.
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


def convert_svg(
    svg_path: str,
    output_path: str,
    pps: int = 30000,
    fps: int = 30,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> Dict:
    """
    Convenience function to directly convert an SVG to ILDA.
    """
    galvo_config = GalvoConfig(pps=pps, target_fps=fps)
    converter = ILDAConverter(galvo_config=galvo_config)
    return converter.convert_svg(
        svg_path,
        output_path,
        color=color,
    )
