# Windows Quick Reference

## One-Time Setup

```powershell
# Navigate to project
cd C:\Users\zubair\Documents\GitHub\ilda-converter

# Create virtual environment
python -m venv .venv

# Activate (do this every time you open a new terminal)
.\.venv\Scripts\Activate.ps1

# Install
pip install -e .
```

## Daily Usage

### Activate Environment
```powershell
cd C:\Users\zubair\Documents\GitHub\ilda-converter
.\.venv\Scripts\Activate.ps1
```

### Basic Conversion
```powershell
ilda-convert mylogo.png output.ilda
```

### With Preprocessing
```powershell
ilda-convert --preprocess mylogo.png output.ilda
```

### Custom Color
```powershell
# Red
ilda-convert --color 255,0,0 logo.png red.ilda

# Green
ilda-convert --color 0,255,0 logo.png green.ilda

# Blue
ilda-convert --color 0,0,255 logo.png blue.ilda
```

### High Performance Scanner
```powershell
ilda-convert --pps 40000 --fps 30 logo.png output.ilda
```

## Python API

```python
from ilda_converter import convert_image

stats = convert_image(
    "logo.png",
    "output.ilda",
    vectorizer="centerline",
    color=(255, 255, 255),
    preprocess=True,
)

print(f"Generated {stats['total_points']} points")
```

## Troubleshooting

### "ilda-convert not found"
```powershell
# Make sure venv is activated
.\.venv\Scripts\Activate.ps1

# Or use full path
python -m ilda_converter.cli logo.png output.ilda
```

### "Point budget exceeded"
```powershell
# Option 1: Simplify more
ilda-convert --simplify 2.0 logo.png output.ilda

# Option 2: Lower frame rate
ilda-convert --fps 20 logo.png output.ilda

# Option 3: Higher PPS (if hardware supports)
ilda-convert --pps 40000 logo.png output.ilda
```

### PowerShell Execution Policy
If activation fails:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## File Locations

```
ilda-converter/
├── .venv/                  # Virtual environment (don't commit)
├── src/ilda_converter/     # Source code
├── tests/                  # Unit tests
├── examples/               # Example scripts
├── docs/                   # Documentation
│   ├── windows-setup.md    # Detailed Windows guide
│   ├── quickstart.md       # Quick start guide
│   └── architecture.md     # Technical details
└── README.md               # Main documentation
```

## Running Tests

```powershell
# Activate venv first
.\.venv\Scripts\Activate.ps1

# Install pytest
pip install pytest

# Run tests
python -m pytest tests/ -v
```

## Deactivate Environment

```powershell
deactivate
```

## Common Commands Cheat Sheet

| Task | Command |
|------|---------|
| Activate venv | `.\.venv\Scripts\Activate.ps1` |
| Basic convert | `ilda-convert logo.png out.ilda` |
| With preprocess | `ilda-convert -p logo.png out.ilda` |
| Custom color | `ilda-convert --color R,G,B logo.png out.ilda` |
| Higher quality | `ilda-convert --simplify 0.5 logo.png out.ilda` |
| Run tests | `python -m pytest tests/ -v` |
| Help | `ilda-convert --help` |
| Deactivate | `deactivate` |

## Need More Help?

- Full documentation: `README.md`
- Windows setup: `docs\windows-setup.md`
- Test results: `WINDOWS_TEST_RESULTS.md`
- Quick start: `docs\quickstart.md`
