"""
Path optimization using vpype for TSP-based reordering, simplification,
and ribbon-to-centerline collapse for eliminating double lines.

Minimizes blanked travel distance, reduces redundant points, and collapses
thick stroke ribbons into clean single-stroke centerlines.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
from shapely.geometry import LineString


class PathOptimizer:
    """Optimize vector paths for laser scanning."""

    def __init__(
        self,
        tolerance: float = 1.0,
        merge_threshold: float = 0.5,
        max_line_thickness: float = 25.0,
        merge_close_distance: float = 0.0,
        text_roi: Optional[Union[str, Tuple[float, ...]]] = None,
        text_tolerance: Optional[float] = None,
        bg_tolerance: Optional[float] = None,
        text_line_thickness: Optional[float] = None,
        bg_line_thickness: Optional[float] = None,
        bg_merge_close_distance: Optional[float] = None,
    ):
        """
        Initialize path optimizer.

        Args:
            tolerance: Default simplification tolerance in mm (Ramer-Douglas-Peucker)
            merge_threshold: Distance threshold for merging adjacent segments in mm
            max_line_thickness: Max width in units to collapse closed ribbon outlines
                                into single centerlines (avoids double lines; 0 to disable)
            merge_close_distance: Max distance in units to merge separate nearby parallel lines
                                  into a single centerline (0 to disable)
            text_roi: Region of interest for text: 'auto', (ymin, ymax), (xmin, ymin, xmax, ymax), or None
            text_tolerance: Specific simplification tolerance for text (finer detail, e.g. 0.5mm)
            bg_tolerance: Specific simplification tolerance for background (coarse/simplified, e.g. 3.5mm)
            text_line_thickness: Ribbon collapse thickness for text (0 to preserve font outlines)
            bg_line_thickness: Ribbon collapse thickness for background ribbons/streaks
            bg_merge_close_distance: Merge distance specifically for background lines
        """
        self.tolerance = tolerance
        self.merge_threshold = merge_threshold
        self.max_line_thickness = max_line_thickness
        self.merge_close_distance = merge_close_distance
        self.text_roi = text_roi
        self.text_tolerance = text_tolerance
        self.bg_tolerance = bg_tolerance
        self.text_line_thickness = text_line_thickness
        self.bg_line_thickness = bg_line_thickness
        self.bg_merge_close_distance = bg_merge_close_distance

    @staticmethod
    def partition_paths(
        paths: List[np.ndarray],
        text_roi: Optional[Union[str, Tuple[float, ...]]] = "auto",
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[Tuple[float, float]]]:
        """
        Partition paths into (text_paths, bg_paths, roi_band).
        Identifies the text section using spatial clustering or specified ROI.
        """
        valid_paths = [p for p in paths if len(p) >= 2]
        if not valid_paths or not text_roi:
            return [], valid_paths, None

        all_coords = np.vstack(valid_paths)
        min_x, max_x = float(all_coords[:, 0].min()), float(all_coords[:, 0].max())
        min_y, max_y = float(all_coords[:, 1].min()), float(all_coords[:, 1].max())
        canvas_w = max(max_x - min_x, 1e-6)
        canvas_h = max(max_y - min_y, 1e-6)

        roi_ymin = 0.0
        roi_ymax = 1.0
        roi_xmin = 0.0
        roi_xmax = 1.0

        if text_roi == "auto":
            # Auto-detect horizontal text band from cluster of smaller features (letterforms, inner loops)
            path_stats = []
            for p in valid_paths:
                p_h = (p[:, 1].max() - p[:, 1].min()) / canvas_h
                p_mid_y = ((p[:, 1].min() + p[:, 1].max()) / 2.0 - min_y) / canvas_h
                path_stats.append((p_mid_y, p_h))

            # Candidate text features: individual height < 35% of overall canvas height
            small_paths = [s for s in path_stats if s[1] < 0.35]
            if len(small_paths) >= 2:
                y_centers = [s[0] for s in small_paths]
                med_y = float(np.median(y_centers))
                roi_ymin = max(0.0, med_y - 0.18)
                roi_ymax = min(1.0, med_y + 0.18)
            else:
                return [], valid_paths, None
        elif isinstance(text_roi, (list, tuple)):
            if len(text_roi) == 2:
                roi_ymin, roi_ymax = float(text_roi[0]), float(text_roi[1])
            elif len(text_roi) == 4:
                roi_xmin, roi_ymin, roi_xmax, roi_ymax = (float(v) for v in text_roi)

        text_paths = []
        bg_paths = []

        for p in valid_paths:
            p_mid_x = ((p[:, 0].min() + p[:, 0].max()) / 2.0 - min_x) / canvas_w
            p_mid_y = ((p[:, 1].min() + p[:, 1].max()) / 2.0 - min_y) / canvas_h
            p_h = (p[:, 1].max() - p[:, 1].min()) / canvas_h

            in_x = roi_xmin <= p_mid_x <= roi_xmax
            in_y = roi_ymin <= p_mid_y <= roi_ymax

            if in_x and in_y and p_h < 0.40:
                text_paths.append(p)
            else:
                bg_paths.append(p)

        return text_paths, bg_paths, (roi_ymin, roi_ymax)

    def optimize_paths(
        self,
        paths: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], float]:
        """
        Optimize paths using ribbon collapse and in-memory vpype pipeline.
        Supports region-aware segmented optimization (fine text, simplified background).
        """
        valid_paths = [p for p in paths if len(p) >= 2]
        if not valid_paths:
            return [], 0.0

        # Segmented region-aware processing (Text vs Background)
        if self.text_roi is not None:
            text_paths, bg_paths, roi = self.partition_paths(valid_paths, self.text_roi)
            if text_paths:
                text_tol = self.text_tolerance if self.text_tolerance is not None else 0.5
                bg_tol = self.bg_tolerance if self.bg_tolerance is not None else 3.5
                text_thk = self.text_line_thickness if self.text_line_thickness is not None else 0.0
                bg_thk = self.bg_line_thickness if self.bg_line_thickness is not None else self.max_line_thickness

                # 1. Background optimization: strong simplification + streak ribbon collapsing
                opt_bg = PathOptimizer(
                    tolerance=bg_tol,
                    merge_threshold=self.merge_threshold,
                    max_line_thickness=bg_thk,
                )
                bg_opt, _ = opt_bg.optimize_paths(bg_paths)

                # 2. Text optimization: fine detail preservation + outline preservation
                opt_text = PathOptimizer(
                    tolerance=text_tol,
                    merge_threshold=min(self.merge_threshold, 0.3),
                    max_line_thickness=text_thk,
                )
                text_opt, _ = opt_text.optimize_paths(text_paths)

                # 3. Combine and globally TSP-sort together
                try:
                    import vpype
                    import vpype_cli

                    lc = vpype.LineCollection()
                    for p in (bg_opt + text_opt):
                        if len(p) >= 2:
                            lc.append(p[:, 0] + 1j * p[:, 1])

                    doc = vpype.Document()
                    doc.add(lc, 1)
                    opt_doc = vpype_cli.execute("reloop linesort", document=doc)
                    opt_lc = opt_doc.layers.get(1, vpype.LineCollection())
                    final_paths = [
                        np.column_stack([line.real, line.imag])
                        for line in opt_lc
                        if len(line) >= 2
                    ]
                except Exception:
                    final_paths = bg_opt + text_opt

                blanking_dist = self._calculate_blanking_distance(final_paths)
                return final_paths, blanking_dist

        # Standard uniform pipeline
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

