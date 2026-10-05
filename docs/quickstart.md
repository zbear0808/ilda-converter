# Quick Start Guide

## Installation

### Windows

```powershell
cd C:\Users\zubair\Documents\GitHub\ilda-converter
pip install -e .
```

**See detailed Windows setup**: [`windows-setup.md`](windows-setup.md)

### Linux/macOS

```bash
cd /path/to/ilda-converter
pip install -e .
```

This installs the package in editable mode with all dependencies.

## Platform Notes

- **Centerline vectorizer**: ✅ Works on all platforms (pure Python)
- **Potrace**: ⚠️ Requires external binary (see setup docs)
- **VTracer**: ⚠️ Requires Rust toolchain (optional)

## Your First Conversion

### 1. Prepare an Image

Create or use any PNG/JPG/BMP image:
- **Logo**: Color or B&W, clean edges work best
- **Text**: High contrast, clear fonts
- **Line art**: Simple shapes, not photographic

### 2. Basic Conversion

```bash
# Default: centerline vectorizer (works everywhere)
ilda-convert my_logo.png output.ilda
```

This uses default settings:
- Centerline vectorization (pure Python, no external dependencies)
- 30k PPS, 30 FPS
- White color
- Auto-optimization

### 3. View Output Stats

The CLI prints detailed statistics:
```
Converting my_logo.png to output.ilda
Stage 2: Vectorizing (centerline)...
  → Extracted 24 paths
Stage 3: Optimizing path order (TSP)...
  → Optimized to 24 paths
  → Total blanking distance: 1250.3 units
Stage 4: Normalizing coordinates...
Stage 5: Conditioning for galvo dynamics...
  → Generated 847 points
  → Point budget: 1000 max
Stage 6: Writing ILDA file...
✓ Conversion complete: output.ilda

============================================================
Conversion Summary
============================================================
Input:              my_logo.png
Output:             output.ilda
Vectorizer:         centerline
Paths:              24
Total Points:       847
  Beam On:          782
  Beam Off:         65
Blanking Distance:  1250.3 units
Point Budget:       1000
Budget Usage:       84.7%
============================================================
```

## Common Use Cases

### High-Contrast B&W Logo

Use Potrace for crisp silhouettes:

```bash
ilda-convert --vectorizer potrace --preprocess logo_bw.png output.ilda
```

### Text or Thin Lines

Use centerline extraction:

```bash
ilda-convert --vectorizer centerline --preprocess text.png output.ilda
```

### Colored Output

Specify RGB color (0-255):

```bash
# Red
ilda-convert --color 255,0,0 logo.png red_logo.ilda

# Green
ilda-convert --color 0,255,0 logo.png green_logo.ilda

# Blue
ilda-convert --color 0,0,255 logo.png blue_logo.ilda

# Orange
ilda-convert --color 255,128,0 logo.png orange_logo.ilda
```

### High-Performance Scanner

For 40k PPS scanners:

```bash
ilda-convert --pps 40000 --fps 30 logo.png output.ilda
```

### Lower Frame Rate (More Complex Images)

Reduce FPS to allow more points:

```bash
ilda-convert --pps 30000 --fps 20 complex_logo.png output.ilda
```

This gives you 1500 points/frame instead of 1000.

## Python API

### Quick Conversion

```python
from ilda_converter import convert_image

stats = convert_image(
    "logo.png",
    "output.ilda",
    vectorizer="centerline",  # Works on all platforms
    color=(255, 255, 255),
)

print(f"Generated {stats['total_points']} points")
```

### Advanced Pipeline Control

```python
from ilda_converter import ILDAConverter, GalvoConfig

# Configure galvo dynamics
config = GalvoConfig(
    pps=30000,
    target_fps=30,
    corner_threshold_deg=45.0,
    corner_dwell_points=3,
)

# Create converter
converter = ILDAConverter(
    vectorizer_method="centerline",  # or "potrace", "vtracer"
    galvo_config=config,
)

# Run conversion
stats = converter.convert(
    "input.png",
    "output.ilda",
    color=(0, 255, 0),  # Green
    preprocess=True,
)
```

## Troubleshooting

### "Point budget exceeded"

**Problem**: Image too complex for frame rate

**Solutions**:
```bash
# Option 1: Reduce frame rate
ilda-convert --fps 20 logo.png output.ilda

# Option 2: Simplify more aggressively
ilda-convert --simplify 2.0 logo.png output.ilda

# Option 3: Increase PPS (if hardware supports)
ilda-convert --pps 40000 logo.png output.ilda
```

### Output looks distorted

**Problem**: Not enough points for smooth lines

**Solutions**:
```bash
# Increase PPS
ilda-convert --pps 40000 logo.png output.ilda

# Lower simplification tolerance
ilda-convert --simplify 0.5 logo.png output.ilda
```

### No paths extracted

**Problem**: Image contrast too low or wrong format

**Solutions**:
```bash
# Enable preprocessing
ilda-convert --preprocess logo.png output.ilda

# Try different vectorizer
ilda-convert --vectorizer potrace --preprocess logo.png output.ilda

# Manual threshold
ilda-convert --preprocess --threshold manual logo.png output.ilda
```

### Lines appear too faint

**Problem**: Not enough intermediate points on long segments

**Fix**: This is handled automatically by `max_segment_spacing` in galvo conditioning. If still an issue, file a bug report with your image.

## Next Steps

1. **Read Architecture**: See `docs/architecture.md` for pipeline details
2. **Run Examples**: Check `examples/basic_usage.py`
3. **Test Output**: Load `.ilda` file into your laser control software
4. **Tune Settings**: Adjust PPS/FPS/colors for your specific hardware

## Getting Help

- Check the full README: `README.md`
- Review architecture docs: `docs/architecture.md`
- Run examples: `python examples/basic_usage.py`
- File issues on GitHub (if public repo)

## Hardware Testing

To test on actual hardware:
1. Convert test image
2. Load `.ilda` file into laser control software (LaserDock, LSX, etc.)
3. Verify:
   - Lines are smooth and visible
   - Corners are sharp (not rounded)
   - No excessive flicker
   - Frame rate is stable

If issues occur, adjust `--pps`, `--fps`, or `--corner-threshold` and reconvert.
