"""
Galvo trajectory conditioning for physical scanner dynamics.

Handles:
- Point budgeting to stay within PPS limits
- Uniform spatial resampling for visible lines
- Corner dwell insertion for sharp angles
- Blanked jump interpolation and settling
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class GalvoConfig:
    """Configuration for galvo scanner dynamics."""
    
    pps: int = 30000  # Points per second
    target_fps: int = 30  # Target frame rate
    corner_threshold_deg: float = 45.0  # Angle threshold for dwell
    corner_dwell_points: int = 3  # Dwell repeats at corners
    jump_settle_points: int = 3  # Settle points after blanked jumps
    jump_interpolation_density: float = 100.0  # Points per unit distance for jumps
    min_segment_points: int = 2  # Minimum points per straight segment
    max_segment_spacing: float = 50.0  # Maximum spacing between points in units


class GalvoConditioner:
    """Condition vector paths for galvo scanner physics."""

    def __init__(self, config: GalvoConfig = None):
        """
        Initialize galvo conditioner.

        Args:
            config: Galvo configuration parameters
        """
        self.config = config or GalvoConfig()
        self.max_points_per_frame = self.config.pps // self.config.target_fps

    def condition_paths(
        self,
        paths: List[np.ndarray],
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> List[dict]:
        """
        Apply full conditioning pipeline to paths.

        Args:
            paths: List of Nx2 coordinate arrays (optimized order)
            color: RGB tuple for all paths

        Returns:
            List of point dictionaries ready for ILDA writer
        """
        # Stage 1: Budget points across all paths
        paths = self._apply_point_budget(paths)

        # Stage 2: Resample each path uniformly
        paths = [self._resample_path_uniform(p) for p in paths]

        # Stage 3: Insert corner dwells
        paths = [self._insert_corner_dwells(p) for p in paths]

        # Stage 4: Build final point list with blanked jumps
        points = self._build_point_list_with_jumps(paths, color)

        return points

    def _apply_point_budget(
        self,
        paths: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Decimate paths to fit within point budget.

        Strategy: Allocate points proportional to path length.
        """
        # Calculate total length
        total_length = sum(self._path_length(p) for p in paths)

        if total_length < 1e-6:
            return paths

        # Reserve budget for jumps (conservative estimate)
        jump_budget = len(paths) * (self.config.jump_settle_points + 10)
        drawable_budget = self.max_points_per_frame - jump_budget

        if drawable_budget <= 0:
            raise ValueError(
                f"Too many paths ({len(paths)}) for frame budget. "
                f"Reduce path count or increase PPS."
            )

        # Allocate points by length
        decimated = []

        for path in paths:
            path_length = self._path_length(path)
            allocated = int(
                (path_length / total_length) * drawable_budget
            )
            allocated = max(allocated, self.config.min_segment_points)

            # Resample to allocated count
            decimated.append(self._resample_path_count(path, allocated))

        return decimated

    def _resample_path_uniform(self, path: np.ndarray) -> np.ndarray:
        """
        Resample path to uniform spatial intervals.

        Ensures no segment exceeds max_segment_spacing.
        """
        if len(path) < 2:
            return path

        # Calculate cumulative distance along path
        segments = np.diff(path, axis=0)
        distances = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate([[0], np.cumsum(distances)])

        total_length = cumulative[-1]

        if total_length < 1e-6:
            return path

        # Determine target spacing
        target_spacing = min(
            self.config.max_segment_spacing,
            total_length / max(len(path), 2),
        )

        # Resample at uniform intervals
        num_points = max(int(total_length / target_spacing) + 1, 2)
        sample_distances = np.linspace(0, total_length, num_points)

        # Interpolate
        resampled = np.zeros((num_points, 2))
        resampled[:, 0] = np.interp(sample_distances, cumulative, path[:, 0])
        resampled[:, 1] = np.interp(sample_distances, cumulative, path[:, 1])

        return resampled

    def _resample_path_count(
        self,
        path: np.ndarray,
        target_count: int,
    ) -> np.ndarray:
        """Resample path to exactly target_count points."""
        if len(path) <= 2:
            return path

        # Calculate cumulative distance
        segments = np.diff(path, axis=0)
        distances = np.linalg.norm(segments, axis=1)
        cumulative = np.concatenate([[0], np.cumsum(distances)])

        total_length = cumulative[-1]

        if total_length < 1e-6:
            return path

        # Sample uniformly
        sample_distances = np.linspace(0, total_length, target_count)

        resampled = np.zeros((target_count, 2))
        resampled[:, 0] = np.interp(sample_distances, cumulative, path[:, 0])
        resampled[:, 1] = np.interp(sample_distances, cumulative, path[:, 1])

        return resampled

    def _insert_corner_dwells(self, path: np.ndarray) -> np.ndarray:
        """
        Insert dwell points at sharp corners.

        Allows galvos to decelerate before turning.
        """
        if len(path) < 3:
            return path

        result = [path[0]]

        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_pt = path[i + 1]

            # Calculate angle at this point
            angle = self._calculate_angle(prev, curr, next_pt)

            # If angle is sharp, repeat the point
            if angle > self.config.corner_threshold_deg:
                for _ in range(self.config.corner_dwell_points):
                    result.append(curr)
            else:
                result.append(curr)

        result.append(path[-1])

        return np.array(result)

    def _build_point_list_with_jumps(
        self,
        paths: List[np.ndarray],
        color: Tuple[int, int, int],
    ) -> List[dict]:
        """
        Build final ILDA point list with blanked jumps between paths.
        """
        r, g, b = color
        points = []

        for i, path in enumerate(paths):
            # Add path points (beam on)
            for pt in path:
                points.append({
                    "x": int(pt[0]),
                    "y": int(pt[1]),
                    "blanked": False,
                    "r": r,
                    "g": g,
                    "b": b,
                })

            # Add blanked jump to next path (if not last)
            if i < len(paths) - 1:
                # Dwell at end of current stroke
                end_pt = path[-1]
                for _ in range(2):
                    points.append({
                        "x": int(end_pt[0]),
                        "y": int(end_pt[1]),
                        "blanked": False,
                        "r": r,
                        "g": g,
                        "b": b,
                    })

                # Interpolate blanked jump
                start_pt = path[-1]
                end_pt = paths[i + 1][0]
                jump_points = self._interpolate_jump(start_pt, end_pt)

                for pt in jump_points:
                    points.append({
                        "x": int(pt[0]),
                        "y": int(pt[1]),
                        "blanked": True,
                        "r": 0,
                        "g": 0,
                        "b": 0,
                    })

                # Settle at start of next stroke
                start_next = paths[i + 1][0]
                for _ in range(self.config.jump_settle_points):
                    points.append({
                        "x": int(start_next[0]),
                        "y": int(start_next[1]),
                        "blanked": True,
                        "r": 0,
                        "g": 0,
                        "b": 0,
                    })

        return points

    def _interpolate_jump(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Interpolate points along a blanked jump.

        Controls mirror slew rate during beam-off transit.
        """
        distance = np.linalg.norm(end - start)

        if distance < 1e-6:
            return []

        # Number of interpolation points
        num_points = max(
            int(distance / self.config.jump_interpolation_density),
            1,
        )

        # Linear interpolation
        interp_points = []
        for i in range(1, num_points + 1):
            t = i / (num_points + 1)
            pt = start + t * (end - start)
            interp_points.append(pt)

        return interp_points

    @staticmethod
    def _path_length(path: np.ndarray) -> float:
        """Calculate total path length."""
        if len(path) < 2:
            return 0.0
        segments = np.diff(path, axis=0)
        distances = np.linalg.norm(segments, axis=1)
        return distances.sum()

    @staticmethod
    def _calculate_angle(
        prev: np.ndarray,
        curr: np.ndarray,
        next_pt: np.ndarray,
    ) -> float:
        """
        Calculate interior angle at curr point (in degrees).

        Returns angle deviation from straight line (0 = straight, 180 = reversal).
        """
        v1 = curr - prev
        v2 = next_pt - curr

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg


def normalize_coordinates(
    paths: List[np.ndarray],
    target_range: Tuple[int, int] = (-32768, 32767),
) -> List[np.ndarray]:
    """
    Normalize path coordinates to ILDA range.

    Args:
        paths: List of coordinate arrays
        target_range: Target coordinate range (ILDA is -32768 to 32767)

    Returns:
        Normalized paths
    """
    if not paths:
        return paths

    # Find global bounding box
    all_points = np.vstack(paths)
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)

    # Calculate scale to fit in target range with padding
    range_size = max_coords - min_coords
    target_size = target_range[1] - target_range[0]

    # Use 90% of available range for padding
    scale = 0.9 * target_size / range_size.max()

    # Center in coordinate space
    center_offset = (target_range[0] + target_range[1]) / 2

    normalized = []
    for path in paths:
        # Center around origin
        centered = path - (min_coords + range_size / 2)
        # Scale
        scaled = centered * scale
        # Shift to target center
        shifted = scaled + center_offset

        normalized.append(shifted)

    return normalized
