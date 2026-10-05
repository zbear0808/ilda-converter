# ILDA Converter

Convert raster images into optimized `.ilda` laser projector files with proper galvo dynamics conditioning.

**✅ Full Windows Support** - All core features work on Windows without external binaries.

## Features

- **Multiple vectorization strategies**: Centerline (pure Python), Potrace (optional), VTracer (optional)
- **TSP path optimization**: Minimize blanked travel distance using vpype
- **Galvo dynamics conditioning**: Point budgeting, uniform resampling, corner dwells, blanked jump interpolation
- **ILDA Format 5**: Industry-standard 2D true color binary format
- **Cross-platform**: Windows, macOS, Linux

## Installation

## Installation

### Quick Install (Windows/Linux/macOS)

```bash
pip install -e .
```

**Windows Users**: See detailed setup guide: [`docs/windows-setup.md`](docs/windows-setup.md)

### Vectorization Methods

| Method | Availability | Best For |
|--------|-------------|----------|
| **Centerline** | ✅ Always (pure Python) | Text, thin lines, wireframes |
| **Potrace** | ⚠️ Optional (binary) | High-contrast B&W logos |
| **VTracer** | ⚠️ Optional (Rust) | Color images, gradients |

**Default**: Centerline (works on all platforms without additional setup)

### Optional Dependencies

**VTracer** (for color vectorization - requires Rust):
```bash
pip install -e ".[vtracer]"
```

**Potrace** (for B&W):
```bash
# Windows: Download from http://potrace.sourceforge.net/
# macOS
brew install potrace

# Ubuntu/Debian
sudo apt-get install potrace
```

## Quick Start

### Command Line

```bash
# Basic conversion (uses centerline - works everywhere)
ilda-convert logo.png output.ilda

# With preprocessing (recommended)
ilda-convert --preprocess logo.png output.ilda

# High-contrast B&W with Potrace (if installed)
ilda-convert --vectorizer potrace --preprocess logo.png output.ilda

# Color images with VTracer (if installed)
ilda-convert --vectorizer vtracer logo.png output.ilda

# Custom galvo settings for 40k PPS scanner
ilda-convert --pps 40000 --fps 30 logo.png output.ilda

# Colored output
ilda-convert --color 255,0,0 logo.png red_logo.ilda
```

### Python API

```python
from ilda_converter import convert_image, ILDAConverter, GalvoConfig

# Quick conversion
stats = convert_image(
    "logo.png",
    "output.ilda",
    vectorizer="centerline",  # or "potrace", "vtracer"
    pps=30000,
    fps=30,
    color=(255, 255, 255),
    preprocess=False,
)

# Advanced: Full pipeline control
galvo_config = GalvoConfig(
    pps=30000,
    target_fps=30,
    corner_threshold_deg=45.0,
    corner_dwell_points=3,
    jump_settle_points=3,
)

converter = ILDAConverter(
    vectorizer_method="centerline",
    galvo_config=galvo_config,
    simplify_tolerance=1.0,
)

stats = converter.convert(
    "input.png",
    "output.ilda",
    color=(0, 255, 0),  # Green
    preprocess=True,
    threshold_method="otsu",
)

print(f"Generated {stats['total_points']} points")
print(f"Budget utilization: {stats['budget_utilization']:.1%}")
```

## Pipeline Stages

### 1. Image Vectorization

Three strategies available (in order of platform compatibility):

- **Centerline** (`--vectorizer centerline`): ✅ Pure Python - works everywhere. Best for thin lines, text, wireframes
- **Potrace** (`--vectorizer potrace`): ⚠️ Requires binary. Best for high-contrast B&W silhouettes
- **VTracer** (`--vectorizer vtracer`): ⚠️ Requires Rust. Best for color images, logos with gradients

### 2. Path Optimization (TSP)

Uses `vpype` to:
- Sort paths to minimize blanked travel distance (Traveling Salesperson Problem)
- Merge adjacent segments within tolerance
- Simplify paths using Ramer-Douglas-Peucker
- Optimize loop starting points

### 3. Galvo Conditioning

Physical scanner dynamics require careful trajectory processing:

#### Point Budgeting
Projectors operate at fixed PPS (Points Per Second):
```
Max Points Per Frame = PPS / Target FPS
```
For 30k PPS at 30 FPS: ~1000 points/frame

#### Uniform Resampling
Long lines need intermediate points or they appear faint. Points are interpolated at uniform spatial intervals.

#### Corner Dwells
Sharp angles (>45°) repeat the coordinate 2-4 times to allow galvo deceleration.

#### Blanked Jumps
Transitions between disconnected paths:
1. Dwell 1-2 points at stroke end
2. Turn off beam (set blanking bit)
3. Interpolate points along transit vector
4. Add 2-4 blanked dwell points at destination for settling
5. Re-enable beam

### 4. ILDA Format 5 Generation

Standard laser projector format:
- 32-byte header (format code, frame name, company, point count)
- 8-byte point records (X, Y, status, R, G, B)
- Coordinates: int16 (-32768 to 32767)
- Colors: uint8 (0 to 255)

## Configuration

### Galvo Settings

```python
from ilda_converter import GalvoConfig

config = GalvoConfig(
    pps=30000,              # Points per second
    target_fps=30,          # Target frame rate
    corner_threshold_deg=45.0,  # Angle threshold for dwell insertion
    corner_dwell_points=3,  # Number of dwell repeats at corners
    jump_settle_points=3,   # Settle points after blanked jumps
    jump_interpolation_density=100.0,  # Points per unit for jumps
    min_segment_points=2,   # Minimum points per segment
    max_segment_spacing=50.0,  # Maximum spacing between points
)
```

### Preprocessing

```python
from ilda_converter import preprocess_image

# Otsu's automatic thresholding
preprocess_image(
    "input.png",
    "output.png",
    threshold_method="otsu",
    invert=False,
)

# Adaptive thresholding (better for varying illumination)
preprocess_image(
    "input.png",
    "output.png",
    threshold_method="adaptive",
)

# Manual threshold
preprocess_image(
    "input.png",
    "output.png",
    threshold_method="manual",
    manual_threshold=127,
)
```

## Examples

See `examples/` directory for sample conversions and advanced usage.

## Troubleshooting

### "Point budget exceeded" Warning

Your image is too complex for the frame rate. Solutions:
- Increase `--pps` (scanner hardware limit)
- Decrease `--fps` (reduces flicker-free threshold)
- Simplify image (fewer paths)
- Increase `--simplify` tolerance

### Flickering Output

- Increase corner dwell points
- Reduce frame rate
- Increase point budget (higher PPS)

### Distorted Lines

- Reduce `max_segment_spacing` for longer uniform resampling
- Increase `jump_settle_points` for better blanking transitions

## Technical References

- ILDA Format Specification: [ILDA Technical Committee](https://www.ilda.com/)
- Galvo Scanner Physics: [Pangolin Laser Systems](https://pangolin.com/)
- TSP Optimization: [vpype Documentation](https://vpype.readthedocs.io/)

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Areas of interest:
- Additional vectorization backends
- Multi-frame / animation support
- Real-time preview renderer
- Galvo calibration tools
- Additional ILDA format variants
