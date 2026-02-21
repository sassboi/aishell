# Release (macOS)

## Build a distributable
```bash
cd /path/to/aishell
python3 -m pip install --upgrade build
python3 -m build
```

Artifacts are created under `dist/`:
- source tarball (`.tar.gz`)
- wheel (`.whl`)

## Test local install from wheel
```bash
pipx install dist/*.whl
aishell
```

## Publish to PyPI
```bash
python3 -m pip install --upgrade twine
twine upload dist/*
```
