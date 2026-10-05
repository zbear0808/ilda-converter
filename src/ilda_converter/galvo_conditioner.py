"""
Galvo trajectory conditioning for physical scanner dynamics.

Handles:
- Coordinate normalization to ILDA standard space (-32768 to 32767) with Y-inversion
- Point budgeting to strictly stay within scanner PPS / target FPS limits
- Uniform spatial resampling along segments for consistent beam brightness
- Corner dwell insertion for sharp angles (>45 deg) to avoid corner rounding
- Blanked jump interpolation and destination settling to control slew rates and mirror ringing
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np


@dataclass
class GalvoConfig:
    """Configuration for galvo scanner dynamics."""

    pps: int = 30000  # Points per second (scanner speed, typically 20k - 30k)
    target_fps: int = 30  # Target frame rate (Hz) for flicker-free projection
    corner_threshold_deg: float = 45.0  # Deviation angle threshold for inserting corner dwells
    corner_dwell_points: int = 3  # Dwell point repetitions at sharp corners
    jump_settle_points: int = 3  # Blanked settle points at start of new stroke
    tail_dwell_points: int = 2  # Beam-on dwell points at end of stroke before blanking
    jump_step_size: float = 2000.0  # Max distance per point during blanked transit (ILDA units)
    min_step_size: float = 100.0  # Minimum distance between resampled points
    max_step_size: float = 1500.0  # Maximum distance between resampled points
    invert_y: bool = True  # Invert Y axis (graphics Y-down to ILDA Y-up)
    loop_to_start: bool = True  # Add blanked jump back to start of first stroke for seamless loop
    margin: float = 0.05  # Safety margin fraction from ILDA boundaries


class GalvoConditioner:
    """Condition vector paths for physical galvo scanner physics."""

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
    ) -> List[Dict]:
        """
        Apply complete conditioning pipeline to paths.

        Args:
            paths: List of Nx2 coordinate arrays (in normalized ILDA coordinate space)
            color: RGB color tuple (0-255)

        Returns:
            List of point dictionaries: {'x': int, 'y': int, 'blanked': bool, 'r': int, 'g': int, 'b': int}
        """
        valid_paths = [p for p in paths if len(p) >= 2]
        if not valid_paths:
            return []

        r, g, b = color
        num_paths = len(valid_paths)

        # 1. Analyze transitions between disconnected paths
        transitions = []
        for i in range(num_paths - 1):
            p_end = valid_paths[i][-1]
            p_next = valid_paths[i + 1][0]
            dist = float(np.linalg.norm(p_next - p_end))
            n_jump = max(1, int(dist / self.config.jump_step_size))
            transitions.append((i, p_end, p_next, n_jump))

        # Add return transition to close frame loop cleanly
        if self.config.loop_to_start and num_paths > 0:
            p_end = valid_paths[-1][-1]
            p_start = valid_paths[0][0]
            dist = float(np.linalg.norm(p_start - p_end))
            n_jump = max(1, int(dist / self.config.jump_step_size))
            transitions.append((num_paths - 1, p_end, p_start, n_jump))

        total_transition_points = sum(
            self.config.tail_dwell_points + n_jump + self.config.jump_settle_points
            for _, _, _, n_jump in transitions
        )

        # 2. Identify sharp corners and calculate path lengths
        path_corners = []
        total_corner_dwells = 0
        total_drawable_length = 0.0

        for path in valid_paths:
            corners = set()
            p_len = 0.0
            n_pts = len(path)
            for j in range(n_pts - 1):
                p_len += float(np.linalg.norm(path[j + 1] - path[j]))
            total_drawable_length += p_len

            for j in range(1, n_pts - 1):
                v1 = path[j] - path[j - 1]
                v2 = path[j + 1] - path[j]
                angle = self._calculate_angle(v1, v2)
                if angle >= self.config.corner_threshold_deg:
                    corners.add(j)
                    total_corner_dwells += self.config.corner_dwell_points
            path_corners.append(corners)

        # 3. Budget points for drawable strokes
        drawable_budget = max(
            self.max_points_per_frame - total_transition_points - total_corner_dwells,
            num_paths * 2,
        )

        if total_drawable_length < 1e-6:
            step_size = self.config.max_step_size
        else:
            step_size = total_drawable_length / drawable_budget
            step_size = float(
                np.clip(step_size, self.config.min_step_size, self.config.max_step_size)
            )

        # 4. Generate points with uniform spatial resampling, corner dwells, and jumps
        points: List[Dict] = []
        for i, path in enumerate(valid_paths):
            corners = path_corners[i]
            n_pts = len(path)

            for j in range(n_pts - 1):
                p0 = path[j]
                p1 = path[j + 1]
                seg_len = float(np.linalg.norm(p1 - p0))
                n_sub = max(1, int(round(seg_len / step_size)))

                for s in range(n_sub):
                    t = s / n_sub
                    pt = p0 + t * (p1 - p0)
                    points.append({
                        "x": int(np.clip(round(pt[0]), -32768, 32767)),
                        "y": int(np.clip(round(pt[1]), -32768, 32767)),
                        "blanked": False,
                        "r": r,
                        "g": g,
                        "b": b,
                    })

                # Insert corner dwell repeats at sharp vertex
                if (j + 1) in corners:
                    for _ in range(self.config.corner_dwell_points):
                        points.append({
                            "x": int(np.clip(round(p1[0]), -32768, 32767)),
                            "y": int(np.clip(round(p1[1]), -32768, 32767)),
                            "blanked": False,
                            "r": r,
                            "g": g,
                            "b": b,
                        })

            # Append the end point of the stroke
            p_last = path[-1]
            points.append({
                "x": int(np.clip(round(p_last[0]), -32768, 32767)),
                "y": int(np.clip(round(p_last[1]), -32768, 32767)),
                "blanked": False,
                "r": r,
                "g": g,
                "b": b,
            })

            # Append transition to next path (or loop closure)
            if i < len(transitions):
                _, p_end, p_next, n_jump = transitions[i]

                # Tail dwell (beam on) to prevent diode blanking delay cutting off stroke end
                for _ in range(self.config.tail_dwell_points):
                    points.append({
                        "x": int(np.clip(round(p_end[0]), -32768, 32767)),
                        "y": int(np.clip(round(p_end[1]), -32768, 32767)),
                        "blanked": False,
                        "r": r,
                        "g": g,
                        "b": b,
                    })

                # Blanked jump transit (beam off, controlled slew rate)
                for s in range(1, n_jump + 1):
                    t = s / (n_jump + 1)
                    pt = p_end + t * (p_next - p_end)
                    points.append({
                        "x": int(np.clip(round(pt[0]), -32768, 32767)),
                        "y": int(np.clip(round(pt[1]), -32768, 32767)),
                        "blanked": True,
                        "r": 0,
                        "g": 0,
                        "b": 0,
                    })

                # Destination settle points (beam off) to let mirror ringing settle
                for _ in range(self.config.jump_settle_points):
                    points.append({
                        "x": int(np.clip(round(p_next[0]), -32768, 32767)),
                        "y": int(np.clip(round(p_next[1]), -32768, 32767)),
                        "blanked": True,
                        "r": 0,
                        "g": 0,
                        "b": 0,
                    })

        return points

    @staticmethod
    def _calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate interior angle deviation in degrees.
        0 deg = straight continuation, 180 deg = full reversal.
        """
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))


def normalize_coordinates(
    paths: List[np.ndarray],
    target_range: Tuple[int, int] = (-32768, 32767),
    margin: float = 0.05,
    invert_y: bool = True,
) -> List[np.ndarray]:
    """
    Normalize path coordinates to ILDA range while preserving aspect ratio.

    Args:
        paths: List of coordinate arrays
        target_range: Target ILDA coordinate range (-32768 to 32767)
        margin: Safety margin fraction around borders (default 5%)
        invert_y: Invert Y axis so graphics appear right-side up

    Returns:
        List of normalized Nx2 coordinate arrays
    """
    valid_paths = [p for p in paths if len(p) >= 1]
    if not valid_paths:
        return paths

    all_points = np.vstack(valid_paths)
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    range_xy = max_xy - min_xy
    max_span = max(float(range_xy[0]), float(range_xy[1]), 1e-6)

    total_target_span = float(target_range[1] - target_range[0])
    usable_span = total_target_span * (1.0 - margin * 2)
    scale = usable_span / max_span
    center_src = (min_xy + max_xy) / 2.0

    target_center = (target_range[0] + target_range[1]) / 2.0

    normalized = []
    for path in valid_paths:
        centered = path - center_src
        x = centered[:, 0] * scale + target_center
        y = centered[:, 1] * scale
        if invert_y:
            y = -y + target_center
        else:
            y = y + target_center
        normalized.append(np.column_stack([x, y]))

    return normalized
