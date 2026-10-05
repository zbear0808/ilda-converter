"""
Image vectorization using multiple strategies for optimal path extraction.

Supports:
- VTracer for color images
- Potrace for high-contrast silhouettes  
- Centerline extraction for thin lines/text
"""

import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Literal
import numpy as np
import cv2
from skimage import morphology, img_as_ubyte
from PIL import Image
import svgpathtools


VectorizerType = Literal["vtracer", "potrace", "centerline"]


class Vectorizer:
    """Convert raster images to vector paths."""

    def __init__(
        self,
        method: VectorizerType = "centerline",
        simplify_tolerance: float = 1.0,
    ):
        """
        Initialize vectorizer.

        Args:
            method: Vectorization method to use (default: centerline - works on all platforms)
            simplify_tolerance: Path simplification tolerance in pixels
        """
        self.method = method
        self.simplify_tolerance = simplify_tolerance

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

        if self.method == "vtracer":
            self._vectorize_vtracer(image_path, output_svg_path)
        elif self.method == "potrace":
            self._vectorize_potrace(image_path, output_svg_path)
        elif self.method == "centerline":
            self._vectorize_centerline(image_path, output_svg_path)
        else:
            raise ValueError(f"Unknown vectorization method: {self.method}")

        # Parse SVG to coordinate arrays
        paths = self._parse_svg_paths(output_svg_path)
        return paths, output_svg_path

    def _vectorize_vtracer(self, image_path: str, output_path: str):
        """Vectorize using VTracer (best for color images)."""
        try:
            import vtracer
        except ImportError:
            raise ImportError(
                "vtracer not installed. VTracer requires Rust toolchain.\n"
                "Install with: pip install vtracer\n"
                "Or use alternative vectorizers:\n"
                "  --vectorizer potrace (for B&W images)\n"
                "  --vectorizer centerline (for thin lines/text)"
            )

        # VTracer works best with good contrast
        vtracer.convert_image_to_svg_py(
            image_path,
            output_path,
            colormode="color",  # or 'binary'
            hierarchical="stacked",
            mode="spline",
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=8,
        )

    def _vectorize_potrace(self, image_path: str, output_path: str):
        """Vectorize using Potrace (best for high-contrast B&W)."""
        # Preprocess: threshold to pure B&W
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Apply Otsu's thresholding
        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Save temporary BMP for potrace
        temp_bmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bmp", delete=False
        )
        temp_bmp_path = temp_bmp.name
        temp_bmp.close()

        cv2.imwrite(temp_bmp_path, binary)

        # Run potrace command line
        import subprocess

        try:
            subprocess.run(
                [
                    "potrace",
                    "-s",  # SVG output
                    "-o",
                    output_path,
                    temp_bmp_path,
                ],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            raise ImportError(
                "potrace not found. Install from: http://potrace.sourceforge.net/"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"potrace failed: {e.stderr.decode()}")
        finally:
            Path(temp_bmp_path).unlink(missing_ok=True)

    def _vectorize_centerline(self, image_path: str, output_path: str):
        """Extract centerlines using skeletonization (best for thin lines)."""
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Threshold
        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Skeletonize to 1-pixel-wide centerlines
        binary_bool = binary > 0
        skeleton = morphology.skeletonize(binary_bool)

        # Convert skeleton to contours
        skeleton_img = img_as_ubyte(skeleton)
        contours, _ = cv2.findContours(
            skeleton_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # Build SVG from contours
        self._contours_to_svg(contours, output_path, img.shape)

    def _contours_to_svg(
        self,
        contours: List[np.ndarray],
        output_path: str,
        image_shape: Tuple[int, int],
    ):
        """Convert OpenCV contours to SVG file."""
        height, width = image_shape

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        ]

        for contour in contours:
            if len(contour) < 2:
                continue

            # Simplify contour
            epsilon = self.simplify_tolerance
            approx = cv2.approxPolyDP(contour, epsilon, closed=False)

            # Build path
            points = approx.reshape(-1, 2)
            path_d = f"M {points[0][0]},{points[0][1]}"

            for pt in points[1:]:
                path_d += f" L {pt[0]},{pt[1]}"

            svg_lines.append(
                f'<path d="{path_d}" fill="none" stroke="black" stroke-width="1"/>'
            )

        svg_lines.append("</svg>")

        with open(output_path, "w") as f:
            f.write("\n".join(svg_lines))

    def _parse_svg_paths(self, svg_path: str) -> List[np.ndarray]:
        """
        Parse SVG file into coordinate arrays.

        Returns:
            List of Nx2 numpy arrays (each array is one path)
        """
        try:
            paths, _ = svgpathtools.svg2paths(svg_path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse SVG: {e}")

        path_arrays = []

        for path in paths:
            # Sample each SVG path into discrete points
            # Use length-based sampling for uniform spacing
            length = path.length()
            if length < 1e-6:
                continue

            # Sample approximately every 1 unit
            num_samples = max(int(length) + 1, 10)
            coords = []

            for i in range(num_samples):
                t = i / (num_samples - 1)
                point = path.point(t)
                coords.append([point.real, point.imag])

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
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

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
