"""
Path optimization using vpype for TSP-based reordering and simplification.

Minimizes blanked travel distance and reduces redundant points.
"""

import tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np
import subprocess


class PathOptimizer:
    """Optimize vector paths for laser scanning."""

    def __init__(
        self,
        tolerance: float = 0.1,
        merge_threshold: float = 0.5,
    ):
        """
        Initialize path optimizer.

        Args:
            tolerance: Simplification tolerance in mm
            merge_threshold: Distance threshold for merging adjacent segments in mm
        """
        self.tolerance = tolerance
        self.merge_threshold = merge_threshold

    def optimize_paths(
        self,
        paths: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], float]:
        """
        Optimize paths using vpype pipeline.

        Args:
            paths: List of Nx2 coordinate arrays

        Returns:
            Tuple of (optimized_paths, total_blanking_distance)
        """
        # Write paths to temporary SVG
        temp_svg_in = self._paths_to_svg(paths)
        temp_svg_out = tempfile.NamedTemporaryFile(
            mode="w", suffix=".svg", delete=False
        )
        temp_svg_out.close()

        try:
            # Run vpype optimization pipeline
            self._run_vpype_pipeline(temp_svg_in, temp_svg_out.name)

            # Parse optimized SVG back to paths
            optimized_paths = self._svg_to_paths(temp_svg_out.name)

            # Calculate total blanking distance
            blanking_dist = self._calculate_blanking_distance(optimized_paths)

            return optimized_paths, blanking_dist

        finally:
            Path(temp_svg_in).unlink(missing_ok=True)
            Path(temp_svg_out.name).unlink(missing_ok=True)

    def _run_vpype_pipeline(self, input_svg: str, output_svg: str):
        """
        Run vpype optimization pipeline.

        Pipeline stages:
        1. linesimplify: Remove redundant collinear points
        2. linemerge: Merge adjacent segments
        3. reloop: Optimize loop starting points
        4. linesort: TSP-based path reordering
        """
        try:
            cmd = [
                "vpype",
                "read", input_svg,
                "linesimplify", f"--tolerance={self.tolerance}mm",
                "linemerge", f"--tolerance={self.merge_threshold}mm",
                "reloop",
                "linesort",  # TSP optimization with forward/reverse consideration
                "write", output_svg,
            ]

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            # Optionally parse statistics from vpype output
            # (vpype prints useful info about path count, length, etc.)
            if result.stderr:
                print("vpype output:", result.stderr)

        except FileNotFoundError:
            raise ImportError(
                "vpype not found. Install with: pip install vpype"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"vpype optimization failed: {e.stderr}"
            )

    def _paths_to_svg(self, paths: List[np.ndarray]) -> str:
        """Convert path arrays to temporary SVG file."""
        temp_svg = tempfile.NamedTemporaryFile(
            mode="w", suffix=".svg", delete=False
        )

        # Find bounding box
        all_points = np.vstack(paths) if paths else np.array([[0, 0]])
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)

        width = max_x - min_x + 10
        height = max_y - min_y + 10

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="{min_x-5} {min_y-5} {width} {height}">',
        ]

        for path in paths:
            if len(path) < 2:
                continue

            # Build path string
            path_d = f"M {path[0][0]},{path[0][1]}"
            for pt in path[1:]:
                path_d += f" L {pt[0]},{pt[1]}"

            svg_lines.append(
                f'<path d="{path_d}" fill="none" stroke="black" stroke-width="1"/>'
            )

        svg_lines.append("</svg>")

        temp_svg.write("\n".join(svg_lines))
        temp_svg.close()

        return temp_svg.name

    def _svg_to_paths(self, svg_path: str) -> List[np.ndarray]:
        """Parse optimized SVG back to path arrays."""
        try:
            import svgpathtools
            paths, _ = svgpathtools.svg2paths(svg_path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse optimized SVG: {e}")

        path_arrays = []

        for path in paths:
            length = path.length()
            if length < 1e-6:
                continue

            # Sample path uniformly
            num_samples = max(int(length) + 1, 10)
            coords = []

            for i in range(num_samples):
                t = i / (num_samples - 1)
                point = path.point(t)
                coords.append([point.real, point.imag])

            path_arrays.append(np.array(coords))

        return path_arrays

    def _calculate_blanking_distance(self, paths: List[np.ndarray]) -> float:
        """
        Calculate total blanked travel distance between paths.

        This is the sum of Euclidean distances between the end of one path
        and the start of the next path.
        """
        if len(paths) < 2:
            return 0.0

        total_dist = 0.0

        for i in range(len(paths) - 1):
            end_point = paths[i][-1]
            start_point = paths[i + 1][0]
            dist = np.linalg.norm(end_point - start_point)
            total_dist += dist

        return total_dist


def optimize_with_vpype_api(
    paths: List[np.ndarray],
    tolerance: float = 0.1,
) -> List[np.ndarray]:
    """
    Alternative: Use vpype Python API directly (if available).

    This avoids subprocess calls but requires vpype to be importable.
    """
    try:
        import vpype
        from vpype import LineCollection, VectorData
    except ImportError:
        raise ImportError(
            "vpype API not available. Using CLI-based optimization instead."
        )

    # Create VectorData
    lc = LineCollection()

    for path in paths:
        if len(path) < 2:
            continue
        # Convert to complex array for vpype
        line = path[:, 0] + 1j * path[:, 1]
        lc.append(line)

    vd = VectorData()
    vd.add(lc, layer_id=1)

    # Apply optimizations
    vd = vpype.linesimplify(vd, tolerance=tolerance)
    vd = vpype.linemerge(vd, tolerance=tolerance * 5)
    vd = vpype.reloop(vd)
    vd = vpype.linesort(vd)

    # Extract back to numpy arrays
    optimized_paths = []
    for layer in vd.layers.values():
        for line in layer:
            coords = np.column_stack([line.real, line.imag])
            optimized_paths.append(coords)

    return optimized_paths
