"""
Path optimization using vpype for TSP-based reordering and simplification.

Minimizes blanked travel distance and reduces redundant points.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np


class PathOptimizer:
    """Optimize vector paths for laser scanning."""

    def __init__(
        self,
        tolerance: float = 1.0,
        merge_threshold: float = 0.5,
    ):
        """
        Initialize path optimizer.

        Args:
            tolerance: Simplification tolerance in mm (Ramer-Douglas-Peucker)
            merge_threshold: Distance threshold for merging adjacent segments in mm
        """
        self.tolerance = tolerance
        self.merge_threshold = merge_threshold

    def optimize_paths(
        self,
        paths: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], float]:
        """
        Optimize paths using vpype in-memory pipeline.

        Args:
            paths: List of Nx2 coordinate arrays

        Returns:
            Tuple of (optimized_paths, total_blanking_distance)
        """
        valid_paths = [p for p in paths if len(p) >= 2]
        if not valid_paths:
            return [], 0.0

        try:
            import vpype
            import vpype_cli

            lc = vpype.LineCollection()
            for p in valid_paths:
                line = p[:, 0] + 1j * p[:, 1]
                lc.append(line)

            doc = vpype.Document()
            doc.add(lc, 1)

            pipeline = (
                f"linesimplify --tolerance {self.tolerance}mm "
                f"linemerge --tolerance {self.merge_threshold}mm "
                f"reloop "
                f"linesort"
            )

            opt_doc = vpype_cli.execute(pipeline, document=doc)
            opt_lc = opt_doc.layers.get(1, vpype.LineCollection())

            optimized_paths = [
                np.column_stack([line.real, line.imag])
                for line in opt_lc
                if len(line) >= 2
            ]

        except Exception as e:
            # Fallback if vpype pipeline fails
            print(f"[WARN] In-memory vpype optimization fallback: {e}")
            optimized_paths = valid_paths

        blanking_dist = self._calculate_blanking_distance(optimized_paths)
        return optimized_paths, blanking_dist

    def optimize_svg(
        self,
        svg_path: str,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Optimize paths directly from an SVG file using vpype.

        Args:
            svg_path: Path to SVG file

        Returns:
            Tuple of (optimized_paths, total_blanking_distance)
        """
        import vpype
        import vpype_cli

        posix_path = Path(svg_path).as_posix()
        pipeline = (
            f"read '{posix_path}' "
            f"linesimplify --tolerance {self.tolerance}mm "
            f"linemerge --tolerance {self.merge_threshold}mm "
            f"reloop "
            f"linesort"
        )

        opt_doc = vpype_cli.execute(pipeline)
        opt_lc = opt_doc.layers.get(1, vpype.LineCollection())

        optimized_paths = [
            np.column_stack([line.real, line.imag])
            for line in opt_lc
            if len(line) >= 2
        ]

        blanking_dist = self._calculate_blanking_distance(optimized_paths)
        return optimized_paths, blanking_dist

    @staticmethod
    def _calculate_blanking_distance(paths: List[np.ndarray]) -> float:
        """
        Calculate total blanked travel distance between paths.
        """
        if len(paths) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(len(paths) - 1):
            end_point = paths[i][-1]
            start_point = paths[i + 1][0]
            dist = float(np.linalg.norm(end_point - start_point))
            total_dist += dist

        return total_dist
