# ILDA Converter Architecture

## Overview

The ILDA converter implements a four-stage pipeline optimized for laser galvanometer scanner physics:

```
Raster Image
    ↓
[1] Vectorization (vtracer / potrace / centerline)
    ↓
Vector Paths (SVG-like)
    ↓
[2] Path Optimization (TSP via vpype)
    ↓
Optimized Paths (minimal blanking)
    ↓
[3] Galvo Conditioning
    ↓
Point List (budgeted, resampled, dwell-inserted)
    ↓
[4] ILDA Binary Serialization
    ↓
.ilda File
```

## Stage 1: Vectorization

**Module**: `vectorizer.py`

### Strategy Selection

| Method | Best For | Algorithm |
|--------|----------|-----------|
| `vtracer` | Color images, logos, gradients | Vision-based segmentation → SVG |
| `potrace` | High-contrast B&W silhouettes | Bitmap tracing → Bézier curves |
| `centerline` | Thin lines, text, wireframes | Morphological skeletonization → contours |

### Implementation

1. **Preprocessing** (optional):
   - Thresholding (Otsu / adaptive / manual)
   - Binarization
   - Morphological cleanup

2. **Vectorization**:
   - **VTracer**: Native color → spline paths
   - **Potrace**: Binary → boundary tracing
   - **Centerline**: Skeletonize → extract 1px-wide paths

3. **Path Extraction**:
   - Parse SVG using `svgpathtools`
   - Sample Bézier curves into discrete points
   - Return list of Nx2 coordinate arrays

## Stage 2: Path Optimization

**Module**: `path_optimizer.py`

### TSP Problem

Laser scanners draw **outlines**, not rasters. Disconnected paths require blanked jumps (beam off, mirrors slew to next start point). Unoptimized order causes:
- Excessive blanking distance → flicker
- Long jumps → scanner stress
- Wasted frame time on transits

### vpype Pipeline

```bash
vpype read <svg> \
  linesimplify --tolerance=1.0mm \
  linemerge --tolerance=0.5mm \
  reloop \
  linesort \
  write <optimized.svg>
```

1. **linesimplify**: Ramer-Douglas-Peucker → drop redundant collinear points
2. **linemerge**: Join adjacent segments within tolerance
3. **reloop**: Optimize closed-loop starting points (minimize distance to next path)
4. **linesort**: TSP solver with forward/reverse segment consideration

### Metrics

- **Blanking Distance**: Sum of Euclidean distances between path endpoints
- Minimized by nearest-neighbor heuristic with backtracking

## Stage 3: Galvo Conditioning

**Module**: `galvo_conditioner.py`

### Physical Constraints

Galvanometer scanners are **mass-on-spring systems**:
- Mirror inertia limits acceleration
- Sharp corners cause overshoot/ringing
- Long straight lines need intermediate points or they appear faint

### Sub-Stages

#### 3.1 Point Budgeting

$$\text{Budget} = \frac{\text{PPS}}{\text{FPS}} = \frac{30000}{30} = 1000 \text{ points/frame}$$

- Allocate points proportional to path length
- Reserve budget for blanked jumps (~10% overhead)
- Decimate paths that exceed allocation

#### 3.2 Uniform Spatial Resampling

**Problem**: Raw vectorization produces variable-density points.

**Solution**: Interpolate at uniform intervals:
- Calculate cumulative arc length along path
- Sample at fixed spatial intervals (< `max_segment_spacing`)
- Ensures visible lines even on long segments

#### 3.3 Corner Dwell Insertion

**Problem**: Sharp turns (>45°) cause ringing.

**Solution**: Repeat corner point N times (default 3):
```python
if angle > corner_threshold:
    for _ in range(corner_dwell_points):
        emit(corner_point)
```

Gives galvos time to decelerate before turning.

#### 3.4 Blanked Jump Handling

**Between Paths**:
1. **End dwell**: Repeat last point 1-2x (prevents tail cutoff)
2. **Set blanking bit** (`status = 0x40`)
3. **Interpolate jump**: Add points along transit vector (controls slew rate)
4. **Settle dwell**: Repeat first point 3-4x at destination (wait for ringing to settle)
5. **Re-enable beam** (`status = 0x00`)

## Stage 4: ILDA Binary Serialization

**Module**: `ilda_writer.py`

### Format 5 (2D True Color)

#### File Structure

```
[Header] 32 bytes
  ILDA marker (4 bytes): "ILDA"
  Reserved (3 bytes)
  Format code (1 byte): 5
  Frame name (8 bytes)
  Company name (8 bytes)
  Point count (uint16)
  Frame number (uint16)
  Total frames (uint16)
  Scanner head (uint8)
  Reserved (uint8)

[Point Records] 8 bytes × N
  X (int16): -32768 to 32767
  Y (int16): -32768 to 32767
  Status (uint8): 0x00 = on, 0x40 = blanked
  Red (uint8): 0 to 255
  Green (uint8): 0 to 255
  Blue (uint8): 0 to 255

[EOF Header] 32 bytes
  (Same structure, point count = 0)
```

#### Coordinate System

- Origin: Center of scanner range
- X axis: Horizontal (left = negative)
- Y axis: Vertical (down = negative)
- Range: ±32767 (full mirror deflection)

### Multi-Frame (Future)

For animations:
1. Write frame 0 with header
2. `append_frame()` for frames 1..N
3. EOF header records `total_frames`

## Performance Characteristics

### Time Complexity

| Stage | Complexity | Bottleneck |
|-------|-----------|------------|
| Vectorization | O(pixels) | Image processing |
| TSP (vpype) | O(n² → n log n) | Nearest-neighbor heuristic |
| Resampling | O(n × m) | Interpolation per path |
| ILDA Write | O(n) | I/O |

### Space Complexity

- **Input**: Raster image (W × H pixels)
- **Intermediate**: ~N paths × M points (sparse)
- **Output**: Point budget × 8 bytes

### Typical Pipeline

1000×1000 logo → 50 paths → 800 points/frame → 6.4 KB ILDA file

## Error Handling

### Budget Exceeded

**Cause**: Too many paths or too complex geometry
**Symptoms**: Frame rate drops, flicker
**Solutions**:
1. Increase PPS (hardware limit)
2. Decrease FPS (lower refresh rate)
3. Simplify image (fewer paths)
4. Increase `simplify_tolerance`

### Distorted Lines

**Cause**: Insufficient resampling or settle points
**Solutions**:
1. Reduce `max_segment_spacing`
2. Increase `corner_dwell_points`
3. Increase `jump_settle_points`

### Missing Paths

**Cause**: Vectorization threshold too high
**Solutions**:
1. Adjust preprocessing threshold
2. Try different vectorizer
3. Increase image contrast

## Testing Strategy

### Unit Tests

- `test_ilda_writer.py`: Binary format correctness
- `test_galvo_conditioner.py`: Point generation logic
- `test_path_optimizer.py`: TSP correctness (blanking distance)

### Integration Tests

- End-to-end pipeline on sample images
- Verify output with ILDA validator tools

### Regression Tests

- Known-good outputs for benchmark images
- Performance benchmarks (time, memory)

## Future Enhancements

### Planned

1. **Multi-frame support**: Animation sequences
2. **Color-per-path**: Parse SVG stroke colors
3. **Live preview**: Real-time OpenGL renderer
4. **Galvo calibration**: Geometric distortion correction
5. **Format variants**: ILDA Format 0 (indexed color), Format 1 (2D indexed)

### Research

- **Adaptive point budgeting**: Dynamic FPS based on complexity
- **Neural vectorization**: Deep learning-based tracing
- **Physics simulation**: Model actual galvo dynamics for validation
