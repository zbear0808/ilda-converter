"""
Image vectorization using multiple strategies for optimal path extraction.

Supports:
- VTracer for color images
- Potrace for high-contrast silhouettes  
- Centerline extraction for thin lines/text
"""

import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Literal
import numpy as np
import cv2
from skimage import morphology, img_as_ubyte
from PIL import Image
import svgpathtools


VectorizerType = Literal["vtracer", "potrace", "centerline"]


def trace_skeleton(skeleton_bool: np.ndarray) -> List[np.ndarray]:
    """
    Extract single-stroke paths from a 1-pixel skeleton image without doubling.
    Uses 8-connectivity topological graph traversal.
    """
    skel = skeleton_bool.astype(np.uint8)
    neighbors_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    y_coords, x_coords = np.where(skel > 0)
    pixel_set = set(zip(y_coords, x_coords))
    if not pixel_set:
        return []

    degree = {}
    adj = {}
    for y, x in pixel_set:
        nbrs = []
        for dy, dx in neighbors_offsets:
            ny, nx = y + dy, x + dx
            if (ny, nx) in pixel_set:
                nbrs.append((ny, nx))
        degree[(y, x)] = len(nbrs)
        adj[(y, x)] = nbrs

    visited_edges = set()
    paths = []

    # Priority 1: Endpoints (degree == 1)
    endpoints = [p for p in pixel_set if degree[p] == 1]
    # Priority 2: Branch points (degree >= 3)
    branchpoints = [p for p in pixel_set if degree[p] >= 3]
    start_nodes = endpoints + branchpoints

    for start in start_nodes:
        for nxt in adj[start]:
            edge = tuple(sorted([start, nxt]))
            if edge in visited_edges:
                continue

            path = [start, nxt]
            visited_edges.add(edge)
            prev = start
            curr = nxt

            while True:
                if degree[curr] != 2:
                    break
                next_candidates = [n for n in adj[curr] if n != prev]
                if not next_candidates:
                    break
                nxt_node = next_candidates[0]
                edge = tuple(sorted([curr, nxt_node]))
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                path.append(nxt_node)
                prev = curr
                curr = nxt_node

            if len(path) >= 2:
                coords = np.array([[x, y] for (y, x) in path], dtype=np.float32)
                paths.append(coords)

    # Priority 3: Remaining isolated closed loops
    remaining_pixels = set()
    for p in pixel_set:
        unvisited = [n for n in adj[p] if tuple(sorted([p, n])) not in visited_edges]
        if unvisited:
            remaining_pixels.add(p)

    while remaining_pixels:
        start = next(iter(remaining_pixels))
        unvisited_nbrs = [n for n in adj[start] if tuple(sorted([start, n])) not in visited_edges]
        if not unvisited_nbrs:
            remaining_pixels.remove(start)
            continue

        nxt = unvisited_nbrs[0]
        edge = tuple(sorted([start, nxt]))
        visited_edges.add(edge)
        path = [start, nxt]
        prev = start
        curr = nxt

        while curr != start:
            next_candidates = [n for n in adj[curr] if n != prev and tuple(sorted([curr, n])) not in visited_edges]
            if not next_candidates:
                break
            nxt_node = next_candidates[0]
            edge = tuple(sorted([curr, nxt_node]))
            visited_edges.add(edge)
            path.append(nxt_node)
            prev = curr
            curr = nxt_node

        coords = np.array([[x, y] for (y, x) in path], dtype=np.float32)
        paths.append(coords)
        for p in path:
            remaining_pixels.discard(p)

    return paths


class Vectorizer:
    """Convert raster images to vector paths."""

    def __init__(
        self,
        method: VectorizerType = "centerline",
        simplify_tolerance: float = 1.0,
        line_thickness: Optional[float] = None,
    ):
        """
        Initialize vectorizer.

        Args:
            method: Vectorization method to use ('vtracer', 'centerline', 'potrace')
            simplify_tolerance: Path simplification tolerance in pixels
            line_thickness: Max line thickness in pixels to collapse into single centerlines
        """
        self.method = method
        self.simplify_tolerance = simplify_tolerance
        self.line_thickness = line_thickness


    def vectorize(
        self,
        image_path: str,
        output_svg_path: Optional[str] = None,
    ) -> Tuple[List[np.ndarray], str]:
        """
        Vectorize an image to SVG paths.

        Args:
            image_path: Input image file path
            output_svg_path: Optional output SVG path (temporary if None)

        Returns:
            Tuple of (list of path arrays, svg_path)
            Each path array is Nx2 coordinates
        """
        if output_svg_path is None:
            svg_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".svg", delete=False
            )
            output_svg_path = svg_file.name
            svg_file.close()

        active_image_path = image_path
        temp_rgb_path = None
        try:
            with Image.open(image_path) as im:
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    white_bg = Image.new("RGB", im.size, (255, 255, 255))
                    if im.mode == "RGBA":
                        white_bg.paste(im, mask=im.split()[3])
                    else:
                        rgba = im.convert("RGBA")
                        white_bg.paste(rgba, mask=rgba.split()[3])
                    temp_rgb = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    temp_rgb_path = temp_rgb.name
                    temp_rgb.close()
                    white_bg.save(temp_rgb_path)
                    active_image_path = temp_rgb_path
        except Exception:
            pass

        try:
            if self.method == "vtracer":
                self._vectorize_vtracer(active_image_path, output_svg_path)
            elif self.method == "potrace":
                self._vectorize_potrace(active_image_path, output_svg_path)
            elif self.method == "centerline":
                self._vectorize_centerline(active_image_path, output_svg_path)
            else:
                raise ValueError(f"Unknown vectorization method: {self.method}")
        finally:
            if temp_rgb_path and Path(temp_rgb_path).exists():
                try:
                    Path(temp_rgb_path).unlink(missing_ok=True)
                except Exception:
                    pass

        # Parse SVG to coordinate arrays
        paths = self._parse_svg_paths(output_svg_path)
        return paths, output_svg_path

    def _vectorize_vtracer(
        self,
        image_path: str,
        output_path: str,
        colormode: str = "binary",
    ):
        """Vectorize using VTracer (modern vision-based vectorizer)."""
        try:
            import vtracer
        except ImportError:
            raise ImportError(
                "vtracer not installed.\n"
                "Install with: .\\.venv\\Scripts\\pip.exe install vtracer\n"
                "Or use alternative vectorizers:\n"
                "  --vectorizer centerline (for thin lines/text)\n"
                "  --vectorizer potrace (for B&W silhouettes)"
            )

        # VTracer with spline mode produces smooth laser-friendly curves
        vtracer.convert_image_to_svg_py(
            image_path,
            output_path,
            colormode=colormode,
            mode="spline",
            filter_speckle=4,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=8,
        )

    def _vectorize_potrace(self, image_path: str, output_path: str):
        """Vectorize using Potrace (best for high-contrast B&W)."""
        import subprocess

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        temp_bmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bmp", delete=False
        )
        temp_bmp_path = temp_bmp.name
        temp_bmp.close()

        cv2.imwrite(temp_bmp_path, binary)

        try:
            subprocess.run(
                [
                    "potrace",
                    "-s",
                    "-o",
                    output_path,
                    temp_bmp_path,
                ],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            raise ImportError(
                "potrace not found. Install potrace or use --vectorizer vtracer / centerline"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"potrace failed: {e.stderr.decode()}")
        finally:
            Path(temp_bmp_path).unlink(missing_ok=True)

    def _vectorize_centerline(self, image_path: str, output_path: str):
        """
        Extract centerlines using topological skeletonization (avoiding double lines).
        If line_thickness is set, strokes <= line_thickness become single centerlines,
        while regions > line_thickness keep their outer boundaries.
        """
        img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise ValueError(f"Could not load image: {image_path}")

        if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
            b, g, r, a = cv2.split(img_raw)
            alpha = a.astype(float) / 255.0
            white = np.ones_like(b, dtype=float) * 255.0
            b_comp = (b.astype(float) * alpha + white * (1.0 - alpha)).astype(np.uint8)
            g_comp = (g.astype(float) * alpha + white * (1.0 - alpha)).astype(np.uint8)
            r_comp = (r.astype(float) * alpha + white * (1.0 - alpha)).astype(np.uint8)
            comp = cv2.merge([b_comp, g_comp, r_comp])
            img = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
        elif len(img_raw.shape) == 3:
            img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            img = img_raw

        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        paths = []
        if self.line_thickness and self.line_thickness > 0:
            # Distance transform to differentiate lines from solid fills
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            # Strokes where thickness (2 * dist) <= line_thickness
            line_mask = (2 * dist <= self.line_thickness) & (binary > 0)
            fill_mask = (2 * dist > self.line_thickness) & (binary > 0)

            # 1. Centerlines for lines/strokes
            if np.any(line_mask):
                skel = morphology.skeletonize(line_mask)
                skel_paths = trace_skeleton(skel)
                paths.extend(skel_paths)

            # 2. Outlines for thick fills
            if np.any(fill_mask):
                fill_ubyte = img_as_ubyte(fill_mask)
                contours, _ = cv2.findContours(
                    fill_ubyte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in contours:
                    if len(c) >= 3:
                        paths.append(c.reshape(-1, 2).astype(np.float32))
        else:
            # Skeletonize all foreground strokes to single centerlines
            binary_bool = binary > 0
            skel = morphology.skeletonize(binary_bool)
            paths = trace_skeleton(skel)

        # Simplify extracted paths and prune micro-spurs (< 3px)
        simplified_paths = []
        for p in paths:
            if len(p) < 2:
                continue
            seg_lens = np.linalg.norm(np.diff(p, axis=0), axis=1)
            if np.sum(seg_lens) < 3.0:
                continue
            is_closed = len(p) >= 3 and np.linalg.norm(p[0] - p[-1]) < 2.0
            approx = cv2.approxPolyDP(p.astype(np.float32), self.simplify_tolerance, is_closed).reshape(-1, 2)
            if is_closed and len(approx) >= 3:
                # Ensure closed loop has identical start and end point
                approx = np.vstack([approx, approx[:1]])
            if len(approx) >= 2:
                simplified_paths.append(approx)

        self._paths_to_svg(simplified_paths, output_path, img.shape)

    def _paths_to_svg(
        self,
        paths: List[np.ndarray],
        output_path: str,
        image_shape: Tuple[int, int],
    ):
        """Convert coordinate path arrays to SVG file."""
        height, width = image_shape

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        ]

        for p in paths:
            if len(p) < 2:
                continue
            is_closed = len(p) >= 3 and np.linalg.norm(p[0] - p[-1]) < 1.0
            path_d = f"M {p[0][0]},{p[0][1]}"
            for pt in p[1:]:
                path_d += f" L {pt[0]},{pt[1]}"
            if is_closed:
                path_d += " Z"
            svg_lines.append(
                f'<path d="{path_d}" fill="none" stroke="black" stroke-width="1"/>'
            )

        svg_lines.append("</svg>")

        with open(output_path, "w") as f:
            f.write("\n".join(svg_lines))

    def load_svg(self, svg_path: str) -> List[np.ndarray]:
        """Load and parse paths directly from an existing SVG file."""
        return self._parse_svg_paths(svg_path)



    def _parse_svg_paths(self, svg_path: str) -> List[np.ndarray]:
        """
        Parse SVG file into coordinate arrays with full support for SVG transforms
        (translate, matrix, scale, rotate) to prevent component misalignments.

        Returns:
            List of Nx2 numpy arrays (each array is one continuous stroke)
        """
        # Primary parser: vpype handles full SVG standard including transforms and compound paths
        try:
            import vpype
            import vpype_cli

            posix_path = Path(svg_path).as_posix()
            doc = vpype_cli.execute(f"read '{posix_path}'")
            paths = [
                np.column_stack([line.real, line.imag])
                for lc in doc.layers.values()
                for line in lc
                if len(line) >= 2
            ]
            if paths:
                return paths
        except Exception:
            pass

        # Fallback parser using svgpathtools with explicit transform parsing
        try:
            paths, attributes = svgpathtools.svg2paths(svg_path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse SVG: {e}")

        path_arrays = []

        for path, attr in zip(paths, attributes):
            # Parse transform if present (e.g. translate(tx, ty))
            tx, ty = 0.0, 0.0
            transform_str = attr.get("transform", "")
            if transform_str:
                trans_match = re.search(r"translate\(\s*([-\d.]+)[,\s]+([-\d.]+)\s*\)", transform_str)
                if trans_match:
                    tx = float(trans_match.group(1))
                    ty = float(trans_match.group(2))

            subpaths = path.continuous_subpaths() if hasattr(path, "continuous_subpaths") else [path]
            for subpath in subpaths:
                length = subpath.length()
                if length < 1e-4:
                    continue

                num_samples = max(int(length) + 1, 10)
                coords = []

                for i in range(num_samples):
                    t = i / (num_samples - 1)
                    point = subpath.point(t)
                    coords.append([point.real + tx, point.imag + ty])

                path_arrays.append(np.array(coords))

        return path_arrays



def preprocess_image(
    image_path: str,
    output_path: str,
    threshold_method: Literal["otsu", "adaptive", "manual"] = "otsu",
    manual_threshold: int = 127,
    invert: bool = False,
) -> str:
    """
    Preprocess image before vectorization.

    Args:
        image_path: Input image path
        output_path: Output preprocessed image path
        threshold_method: Thresholding method
        manual_threshold: Threshold value for 'manual' method
        invert: Invert black/white after thresholding

    Returns:
        Path to preprocessed image
    """
    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise ValueError(f"Could not load image: {image_path}")

    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        # Alpha compositing onto white
        b, g, r, a = cv2.split(img_raw)
        alpha = a.astype(float) / 255.0
        white = np.ones_like(b, dtype=float) * 255.0
        b_comp = (b.astype(float) * alpha + white * (1.0 - alpha)).astype(np.uint8)
        g_comp = (g.astype(float) * alpha + white * (1.0 - alpha)).astype(np.uint8)
        r_comp = (r.astype(float) * alpha + white * (1.0 - alpha)).astype(np.uint8)
        comp = cv2.merge([b_comp, g_comp, r_comp])
        img = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
    elif len(img_raw.shape) == 3:
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    else:
        img = img_raw

    if threshold_method == "otsu":
        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
    elif threshold_method == "manual":
        _, binary = cv2.threshold(img, manual_threshold, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")

    if invert:
        binary = cv2.bitwise_not(binary)

    cv2.imwrite(output_path, binary)
    return output_path
