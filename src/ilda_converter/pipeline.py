"""
Main ILDA conversion pipeline.

Orchestrates the complete 4-stage conversion flow:
1. Vectorization (Raster to SVG / Polylines)
2. Path Optimization (TSP shortest jumps, segment merge, collinear simplify)
3. Galvo Trajectory Conditioning (Point budgeting, uniform resampling, corner dwells, blanked jumps & settles)
4. ILDA Format 5 Binary Packaging (Big-endian header, 8-byte point records, EOF header)
"""

from pathlib import Path
from typing import Optional, Tuple, Literal, Dict, Union
import numpy as np

from .vectorizer import Vectorizer, VectorizerType, preprocess_image
from .path_optimizer import PathOptimizer
from .galvo_conditioner import GalvoConditioner, GalvoConfig, normalize_coordinates
from .ilda_writer import ILDAWriter, ILDAPoint


class ILDAConverter:
    """Complete vector / raster to ILDA conversion pipeline."""

    def __init__(
        self,
        vectorizer_method: VectorizerType = "centerline",
        galvo_config: Optional[GalvoConfig] = None,
        simplify_tolerance: float = 1.0,
        path_merge_threshold: float = 0.5,
        line_thickness: float = 25.0,
        merge_close_distance: float = 0.0,
        text_roi: Optional[Union[str, Tuple[float, ...]]] = None,
        text_tolerance: Optional[float] = None,
        bg_tolerance: Optional[float] = None,
        text_line_thickness: Optional[float] = None,
        bg_line_thickness: Optional[float] = None,
    ):
        """
        Initialize ILDA converter.

        Args:
            vectorizer_method: Vectorization strategy ('vtracer', 'centerline', 'potrace')
            galvo_config: Galvo scanner configuration
            simplify_tolerance: Default path simplification tolerance in mm
            path_merge_threshold: Distance for merging adjacent segments in mm
            line_thickness: Max line thickness in pixels/units to collapse into single centerlines
                            (avoids double lines; 0 to disable)
            merge_close_distance: Max distance in units to merge separate nearby parallel lines
                                  into a single centerline (0 to disable)
            text_roi: Text region of interest: 'auto', (ymin, ymax), (xmin, ymin, xmax, ymax), or None
            text_tolerance: Simplification tolerance for text paths (default: 0.5mm)
            bg_tolerance: Simplification tolerance for background paths (default: 3.5mm)
            text_line_thickness: Ribbon collapse thickness for text (default: 0.0 to preserve font outlines)
            bg_line_thickness: Ribbon collapse thickness for background (default: line_thickness)
        """
        self.vectorizer = Vectorizer(
            method=vectorizer_method,
            simplify_tolerance=simplify_tolerance,
            line_thickness=line_thickness,
        )
        self.optimizer = PathOptimizer(
            tolerance=simplify_tolerance,
            merge_threshold=path_merge_threshold,
            max_line_thickness=line_thickness,
            merge_close_distance=merge_close_distance,
            text_roi=text_roi,
            text_tolerance=text_tolerance,
            bg_tolerance=bg_tolerance,
            text_line_thickness=text_line_thickness,
            bg_line_thickness=bg_line_thickness,
        )
        self.conditioner = GalvoConditioner(galvo_config)
        self.line_thickness = line_thickness
        self.merge_close_distance = merge_close_distance
        self.text_roi = text_roi
        self.text_tolerance = text_tolerance
        self.bg_tolerance = bg_tolerance
        self.text_line_thickness = text_line_thickness
        self.bg_line_thickness = bg_line_thickness

    def _apply_overrides(
        self,
        line_thickness: Optional[float] = None,
        merge_close_distance: Optional[float] = None,
        text_roi: Optional[Union[str, Tuple[float, ...]]] = None,
        text_tolerance: Optional[float] = None,
        bg_tolerance: Optional[float] = None,
        text_line_thickness: Optional[float] = None,
        bg_line_thickness: Optional[float] = None,
    ):
        """Apply parameter overrides to vectorizer and path optimizer."""
        if line_thickness is not None:
            self.optimizer.max_line_thickness = line_thickness
            self.vectorizer.line_thickness = line_thickness
        if merge_close_distance is not None:
            self.optimizer.merge_close_distance = merge_close_distance
        if text_roi is not None:
            self.optimizer.text_roi = text_roi
        if text_tolerance is not None:
            self.optimizer.text_tolerance = text_tolerance
        if bg_tolerance is not None:
            self.optimizer.bg_tolerance = bg_tolerance
        if text_line_thickness is not None:
            self.optimizer.text_line_thickness = text_line_thickness
        if bg_line_thickness is not None:
            self.optimizer.bg_line_thickness = bg_line_thickness

    def convert(
        self,
        input_path: str,
        output_ilda_path: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        preprocess: bool = False,
        threshold_method: Literal["otsu", "adaptive", "manual"] = "otsu",
        frame_name: str = "FRAME",
        company_name: str = "CONVERTER",
        write_both: bool = False,
        line_thickness: Optional[float] = None,
        merge_close_distance: Optional[float] = None,
        text_roi: Optional[Union[str, Tuple[float, ...]]] = None,
        text_tolerance: Optional[float] = None,
        bg_tolerance: Optional[float] = None,
        text_line_thickness: Optional[float] = None,
        bg_line_thickness: Optional[float] = None,
    ) -> Dict:
        """
        Convert raster image or SVG file to ILDA format (.ilda or .ild).
        Automatically detects SVG files to skip rasterization.
        """
        self._apply_overrides(
            line_thickness=line_thickness,
            merge_close_distance=merge_close_distance,
            text_roi=text_roi,
            text_tolerance=text_tolerance,
            bg_tolerance=bg_tolerance,
            text_line_thickness=text_line_thickness,
            bg_line_thickness=bg_line_thickness,
        )

        path_obj = Path(input_path)
        if path_obj.suffix.lower() == ".svg":
            return self.convert_svg(
                input_path,
                output_ilda_path,
                color=color,
                frame_name=frame_name,
                company_name=company_name,
                write_both=write_both,
                line_thickness=line_thickness,
                text_roi=text_roi,
                text_tolerance=text_tolerance,
                bg_tolerance=bg_tolerance,
                text_line_thickness=text_line_thickness,
                bg_line_thickness=bg_line_thickness,
            )

        return self.convert_raster(
            input_path,
            output_ilda_path,
            color=color,
            preprocess=preprocess,
            threshold_method=threshold_method,
            frame_name=frame_name,
            company_name=company_name,
            write_both=write_both,
            line_thickness=line_thickness,
            text_roi=text_roi,
            text_tolerance=text_tolerance,
            bg_tolerance=bg_tolerance,
            text_line_thickness=text_line_thickness,
            bg_line_thickness=bg_line_thickness,
        )

    @staticmethod
    def _write_outputs(
        writer: ILDAWriter,
        output_path: str,
        points: list,
        write_both: bool = False,
    ) -> list:
        """Write ILDA file and optionally its counterpart (.ild vs .ilda)."""
        p = Path(output_path)
        writer.write_frame(str(p), points)
        written = [str(p)]

        if write_both:
            ext = p.suffix.lower()
            if ext == ".ilda":
                alt = p.with_suffix(".ild")
            elif ext == ".ild":
                alt = p.with_suffix(".ilda")
            else:
                alt = p.with_name(p.name + ".ild")
            writer.write_frame(str(alt), points)
            written.append(str(alt))

        return written

    def convert_svg(
        self,
        svg_path: str,
        output_ilda_path: str,
        color: Tuple[int, int, int] = (255, 255, 255),
        frame_name: str = "SVG_FRM",
        company_name: str = "CONVERTER",
        write_both: bool = False,
        line_thickness: Optional[float] = None,
        merge_close_distance: Optional[float] = None,
        text_roi: Optional[Union[str, Tuple[float, ...]]] = None,
        text_tolerance: Optional[float] = None,
        bg_tolerance: Optional[float] = None,
        text_line_thickness: Optional[float] = None,
        bg_line_thickness: Optional[float] = None,
    ) -> Dict:
        """
        Directly convert an SVG vector file to ILDA format.
        Bypasses rasterization and runs path optimization and galvo conditioning.
        """
        self._apply_overrides(
            line_thickness=line_thickness,
            merge_close_distance=merge_close_distance,
            text_roi=text_roi,
            text_tolerance=text_tolerance,
            bg_tolerance=bg_tolerance,
            text_line_thickness=text_line_thickness,
            bg_line_thickness=bg_line_thickness,
        )

        print(f"Converting SVG: {svg_path} -> {output_ilda_path}")
        if self.optimizer.max_line_thickness > 0:
            print(f"  -> Ribbon collapse active (max thickness: {self.optimizer.max_line_thickness:.1f} px to prevent double lines)")
        if self.optimizer.merge_close_distance > 0:
            print(f"  -> Close line merge active (merge distance: {self.optimizer.merge_close_distance:.1f} units)")
        if self.optimizer.text_roi is not None:
            print(f"  -> Text-aware ROI active (ROI: {self.optimizer.text_roi}, text_tol: {self.optimizer.text_tolerance}, bg_tol: {self.optimizer.bg_tolerance})")

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

        # Stage 4: Write ILDA binary file(s)
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
        written_files = self._write_outputs(
            writer, output_ilda_path, ilda_points, write_both=write_both
        )
        for w in written_files:
            print(f"[OK] Conversion complete: {w}")

        beam_on_points = sum(1 for p in conditioned_points if not p["blanked"])
        beam_off_points = len(conditioned_points) - beam_on_points

        return {
            "input_file": svg_path,
            "output_file": written_files[0],
            "output_files": written_files,
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
        write_both: bool = False,
        line_thickness: Optional[float] = None,
        merge_close_distance: Optional[float] = None,
        text_roi: Optional[Union[str, Tuple[float, ...]]] = None,
        text_tolerance: Optional[float] = None,
        bg_tolerance: Optional[float] = None,
        text_line_thickness: Optional[float] = None,
        bg_line_thickness: Optional[float] = None,
    ) -> Dict:
        """
        Convert raster image to ILDA format through full 4-stage pipeline.
        """
        self._apply_overrides(
            line_thickness=line_thickness,
            merge_close_distance=merge_close_distance,
            text_roi=text_roi,
            text_tolerance=text_tolerance,
            bg_tolerance=bg_tolerance,
            text_line_thickness=text_line_thickness,
            bg_line_thickness=bg_line_thickness,
        )

        print(f"Converting raster: {image_path} -> {output_ilda_path}")
        if self.optimizer.max_line_thickness > 0:
            print(f"  -> Ribbon collapse active (max thickness: {self.optimizer.max_line_thickness:.1f} px to prevent double lines)")
        if self.optimizer.merge_close_distance > 0:
            print(f"  -> Close line merge active (merge distance: {self.optimizer.merge_close_distance:.1f} units)")
        if self.optimizer.text_roi is not None:
            print(f"  -> Text-aware ROI active (ROI: {self.optimizer.text_roi}, text_tol: {self.optimizer.text_tolerance}, bg_tol: {self.optimizer.bg_tolerance})")

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
        written_files = self._write_outputs(
            writer, output_ilda_path, ilda_points, write_both=write_both
        )
        for w in written_files:
            print(f"[OK] Conversion complete: {w}")

        beam_on_points = sum(1 for p in conditioned_points if not p["blanked"])
        beam_off_points = len(conditioned_points) - beam_on_points

        return {
            "input_file": image_path,
            "output_file": written_files[0],
            "output_files": written_files,
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
    vectorizer: VectorizerType = "centerline",
    pps: int = 30000,
    fps: int = 30,
    color: Tuple[int, int, int] = (255, 255, 255),
    preprocess: bool = False,
    write_both: bool = False,
    line_thickness: float = 25.0,
    merge_close_distance: float = 0.0,
    text_roi: Optional[Union[str, Tuple[float, ...]]] = None,
    text_tolerance: Optional[float] = None,
    bg_tolerance: Optional[float] = None,
    text_line_thickness: Optional[float] = None,
    bg_line_thickness: Optional[float] = None,
) -> Dict:
    """
    Convenience function to convert a raster image or SVG to ILDA (.ilda or .ild).
    """
    galvo_config = GalvoConfig(pps=pps, target_fps=fps)
    converter = ILDAConverter(
        vectorizer_method=vectorizer,
        galvo_config=galvo_config,
        line_thickness=line_thickness,
        merge_close_distance=merge_close_distance,
        text_roi=text_roi,
        text_tolerance=text_tolerance,
        bg_tolerance=bg_tolerance,
        text_line_thickness=text_line_thickness,
        bg_line_thickness=bg_line_thickness,
    )
    return converter.convert(
        image_path,
        output_path,
        color=color,
        preprocess=preprocess,
        write_both=write_both,
        line_thickness=line_thickness,
        merge_close_distance=merge_close_distance,
        text_roi=text_roi,
        text_tolerance=text_tolerance,
        bg_tolerance=bg_tolerance,
        text_line_thickness=text_line_thickness,
        bg_line_thickness=bg_line_thickness,
    )


def convert_svg(
    svg_path: str,
    output_path: str,
    pps: int = 30000,
    fps: int = 30,
    color: Tuple[int, int, int] = (255, 255, 255),
    write_both: bool = False,
    line_thickness: float = 25.0,
    merge_close_distance: float = 0.0,
    text_roi: Optional[Union[str, Tuple[float, ...]]] = None,
    text_tolerance: Optional[float] = None,
    bg_tolerance: Optional[float] = None,
    text_line_thickness: Optional[float] = None,
    bg_line_thickness: Optional[float] = None,
) -> Dict:
    """
    Convenience function to directly convert an SVG to ILDA (.ilda or .ild).
    """
    galvo_config = GalvoConfig(pps=pps, target_fps=fps)
    converter = ILDAConverter(
        galvo_config=galvo_config,
        line_thickness=line_thickness,
        merge_close_distance=merge_close_distance,
        text_roi=text_roi,
        text_tolerance=text_tolerance,
        bg_tolerance=bg_tolerance,
        text_line_thickness=text_line_thickness,
        bg_line_thickness=bg_line_thickness,
    )
    return converter.convert_svg(
        svg_path,
        output_path,
        color=color,
        write_both=write_both,
        line_thickness=line_thickness,
        merge_close_distance=merge_close_distance,
        text_roi=text_roi,
        text_tolerance=text_tolerance,
        bg_tolerance=bg_tolerance,
        text_line_thickness=text_line_thickness,
        bg_line_thickness=bg_line_thickness,
    )

