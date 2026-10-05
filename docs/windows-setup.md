# Windows Setup Guide

## Prerequisites

- **Python 3.9+** - Download from [python.org](https://www.python.org/downloads/)
- **Git** (optional) - For cloning the repository

## Installation

### 1. Install Python Dependencies

Open PowerShell or Command Prompt in the project directory:

```powershell
cd C:\Users\zubair\Documents\GitHub\ilda-converter
pip install -e .
```

This installs all required dependencies:
- numpy (array processing)
- opencv-python (image processing)
- scikit-image (morphology operations)
- vpype (path optimization)
- svgpathtools (SVG parsing)
- shapely (geometry)
- Pillow (image loading)

### 2. Verify Installation

```powershell
ilda-convert --help
```

You should see the help text with all available options.

## Vectorization Methods on Windows

### ✅ Centerline (Default - Always Works)

Uses pure Python dependencies (OpenCV + scikit-image). Works out of the box on Windows.

```powershell
ilda-convert --vectorizer centerline --preprocess logo.png output.ilda
```

**Best for**: Text, thin lines, wireframes, technical drawings

### ⚠ Potrace (Optional - Requires Binary)

Potrace is a native binary that may not be available on Windows by default.

**Option 1: Download Pre-built Binary**
1. Download from: http://potrace.sourceforge.net/#downloading
2. Extract `potrace.exe`
3. Add to PATH or place in project directory

**Option 2: Skip Potrace**
Use centerline or vtracer instead.

```powershell
# Test if potrace is available
potrace --version
```

**Best for**: High-contrast black & white logos, silhouettes

### ⚠ VTracer (Optional - Requires Rust)

VTracer requires a Rust compiler to build native extensions on Windows.

**Installation Steps**:

1. **Install Rust** (one-time setup):
   ```powershell
   # Download and run rustup-init.exe from:
   # https://www.rust-lang.org/tools/install
   ```

2. **Install VTracer**:
   ```powershell
   pip install vtracer
   ```

3. **Test**:
   ```powershell
   python -c "import vtracer; print('VTracer OK')"
   ```

**If Rust installation fails**, skip VTracer and use centerline or potrace.

**Best for**: Full-color images, logos with gradients, complex shapes

## Quick Start (Windows)

### Basic Conversion

```powershell
# Default: centerline vectorization (always works)
ilda-convert logo.png output.ilda

# With preprocessing (recommended for photos)
ilda-convert --preprocess logo.png output.ilda

# Colored output
ilda-convert --color 255,0,0 logo.png red_logo.ilda
```

### Using Python Directly

```powershell
python -c "from ilda_converter import convert_image; convert_image('logo.png', 'output.ilda')"
```

### Using Python Script

Create `convert.py`:
```python
from ilda_converter import convert_image

stats = convert_image(
    "logo.png",
    "output.ilda",
    vectorizer="centerline",  # Works on all platforms
    color=(255, 255, 255),
    preprocess=True,
)

print(f"Generated {stats['total_points']} points")
```

Run it:
```powershell
python convert.py
```

## Common Windows Issues

### Issue: `pip` not recognized

**Solution**: Add Python Scripts to PATH
```powershell
# Add to PATH (adjust Python version):
$env:Path += ";C:\Python39\Scripts"
```

Or reinstall Python with "Add to PATH" checked.

### Issue: `ilda-convert` not recognized

**Solution 1**: Use full path
```powershell
python -m ilda_converter.cli logo.png output.ilda
```

**Solution 2**: Reinstall in editable mode
```powershell
pip install -e .
```

### Issue: OpenCV import error

**Solution**: Reinstall opencv-python
```powershell
pip uninstall opencv-python
pip install opencv-python
```

### Issue: vpype not found

**Solution**: Reinstall vpype
```powershell
pip install vpype --force-reinstall
```

### Issue: Long path names

Windows has a 260-character path limit. If you encounter path errors:

**Solution**: Enable long paths (Windows 10+)
1. Open Registry Editor (regedit)
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart

Or use shorter directory names.

## Testing Your Installation

Create a test image or use an existing one:

```powershell
# Test with all three vectorizers (if available)

# Centerline (always works)
ilda-convert --vectorizer centerline --preprocess test.png test_centerline.ilda

# Potrace (if installed)
ilda-convert --vectorizer potrace --preprocess test.png test_potrace.ilda

# VTracer (if Rust + vtracer installed)
ilda-convert --vectorizer vtracer test.png test_vtracer.ilda
```

## Performance on Windows

**Typical conversion times** (1000×1000 image on modern PC):
- Centerline: 2-5 seconds
- Potrace: 1-3 seconds (if available)
- VTracer: 3-8 seconds (if available)

**Memory usage**: ~200-500 MB during conversion

## Virtual Environment (Recommended)

Keep dependencies isolated:

```powershell
# Create venv
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Command Prompt)
.\venv\Scripts\activate.bat

# Install
pip install -e .

# Use
ilda-convert logo.png output.ilda

# Deactivate when done
deactivate
```

## Uninstallation

```powershell
pip uninstall ilda-converter
```

## Next Steps

1. Read the main README: `README.md`
2. Try examples: `python examples\basic_usage.py`
3. Check quick start: `docs\quickstart.md`

## Getting Help

If you encounter issues:
1. Check this guide's troubleshooting section
2. Verify Python version: `python --version` (should be 3.9+)
3. Check installed packages: `pip list | findstr ilda`
4. Run with verbose output: Add `--verbose` flag (if implemented)
