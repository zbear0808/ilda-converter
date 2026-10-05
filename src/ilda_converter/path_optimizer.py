"""
Path optimization using vpype for TSP-based reordering, simplification,
and ribbon-to-centerline collapse for eliminating double lines.

Minimizes blanked travel distance, reduces redundant points, and collapses
thick stroke ribbons into clean single-stroke centerlines.
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
        max_line_thickness: float = 25.0,
    ):
        """
        Initialize path optimizer.

        Args:
            tolerance: Simplification tolerance in mm (Ramer-Douglas-Peucker)
            merge_threshold: Distance threshold for merging adjacent segments in mm
            max_line_thickness: Max width in units to collapse closed ribbon outlines
                                into single centerlines (avoids double lines; 0 to disable)
        """
        self.tolerance = tolerance
        self.merge_threshold = merge_threshold
        self.max_line_thickness = max_line_thickness

    def optimize_paths(
        self,
        paths: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], float]:
        """
        Optimize paths using ribbon collapse and in-memory vpype pipeline.

        Args:
            paths: List of Nx2 coordinate arrays

        Returns:
            Tuple of (optimized_paths, total_blanking_distance)
        """
        valid_paths = [p for p in paths if len(p) >= 2]
        if not valid_paths:
            return [], 0.0

        # Step 1: Collapse ribbon outlines whose thickness <= max_line_thickness into single centerlines
        if self.max_line_thickness > 0:
            valid_paths = self.collapse_ribbons(valid_paths, self.max_line_thickness)

        try:
            import vpype
            import vpype_cli

            lc = vpype.LineCollection()
            for p in valid_paths:
                if len(p) >= 2:
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
            print(f"[WARN] In-memory vpype optimization fallback: {e}")
            optimized_paths = valid_paths

        blanking_dist = self._calculate_blanking_distance(optimized_paths)
        return optimized_paths, blanking_dist

    def optimize_svg(
        self,
        svg_path: str,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Optimize paths directly from an SVG file using ribbon collapse and vpype.

        Args:
            svg_path: Path to SVG file

        Returns:
            Tuple of (optimized_paths, total_blanking_distance)
        """
        import vpype
        import vpype_cli

        posix_path = Path(svg_path).as_posix()
        # Read raw SVG lines with vpype
        doc = vpype_cli.execute(f"read '{posix_path}'")
        raw_lc = doc.layers.get(1, vpype.LineCollection())
        raw_paths = [
            np.column_stack([line.real, line.imag])
            for line in raw_lc
            if len(line) >= 2
        ]

        # Optimize with ribbon collapsing and TSP
        return self.optimize_paths(raw_paths)

    @classmethod
    def collapse_ribbons(
        cls,
        paths: List[np.ndarray],
        max_thickness: float,
    ) -> List[np.ndarray]:
        """
        Collapse closed ribbon loop paths whose opposing side distance <= max_thickness
        into single centerlines, eliminating double lines.
        """
        result = []
        for p in paths:
            collapsed = cls._collapse_single_ribbon(p, max_thickness)
            result.extend(collapsed)
        return result

    @staticmethod
    def _collapse_single_ribbon(
        coords: np.ndarray,
        max_thickness: float,
    ) -> List[np.ndarray]:
        """
        Analyze a single closed loop. If it is an elongated ribbon with width <= max_thickness,
        return its medial centerline. Otherwise return original loop.
        """
        n = len(coords)
        if n < 6:
            return [coords]

        # Check if closed
        if np.linalg.norm(coords[0] - coords[-1]) > 2.0:
            return [coords]

        # Subsample to find the two turnaround tips (furthest points in Euclidean distance)
        sub_step = max(1, n // 120)
        indices = np.arange(0, n, sub_step)
        sub_coords = coords[indices]

        diff = sub_coords[:, np.newaxis, :] - sub_coords[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        i_sub, j_sub = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)

        idx1 = int(indices[i_sub])
        idx2 = int(indices[j_sub])
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        # Must divide the loop into two distinct segments
        if (idx2 - idx1) < 3 or (n - (idx2 - idx1)) < 3:
            return [coords]

        side_a = coords[idx1 : idx2 + 1]
        side_b = np.concatenate([coords[idx2:], coords[: idx1 + 1]])
        side_b_rev = side_b[::-1]

        len_a = float(np.sum(np.linalg.norm(np.diff(side_a, axis=0), axis=1)))
        len_b = float(np.sum(np.linalg.norm(np.diff(side_b, axis=0), axis=1)))
        long_dim = max(len_a, len_b)

        if long_dim < 1e-4:
            return [coords]

        # Resample both sides to equal arc-length points to measure thickness
        num_eval = max(20, min(100, n // 4))
        s_eval = np.linspace(0, 1, num_eval)

        def resample_arc(side, s):
            seg_lens = np.linalg.norm(np.diff(side, axis=0), axis=1)
            dists = np.concatenate([[0], np.cumsum(seg_lens)])
            if dists[-1] < 1e-6:
                return np.repeat(side[:1], len(s), axis=0)
            s_norm = dists / dists[-1]
            x_interp = np.interp(s, s_norm, side[:, 0])
            y_interp = np.interp(s, s_norm, side[:, 1])
            return np.column_stack([x_interp, y_interp])

        side_a_eval = resample_arc(side_a, s_eval)
        side_b_eval = resample_arc(side_b_rev, s_eval)

        widths = np.linalg.norm(side_a_eval - side_b_eval, axis=1)
        mean_w = float(np.mean(widths))

        # Condition for a stroke ribbon:
        # 1. Mean thickness <= max_thickness
        # 2. Elongated shape: length is at least 1.8x the mean thickness
        if mean_w <= max_thickness and long_dim >= 1.8 * mean_w:
            centerline = (side_a_eval + side_b_eval) / 2.0
            return [centerline]

        return [coords]

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

