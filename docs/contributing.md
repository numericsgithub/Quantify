# Contributing & Conventions

Thank you for your interest in contributing to the Brevitas QAT Framework!

## Guidelines
1. **Dependencies**: Add new packages to `requirements.txt` or `pyproject.toml` before importing.
2. **Skills**: If you identify a reusable pattern tightly coupled to this framework, create a new `.md` file in `docs/developer/`.
3. **Pitfalls**: Document common errors or debugging tips in `docs/developer/brevitas-pitfalls.md`.
4. **Testing**: Ensure all tests pass (`pytest tests/ -v`) before submitting changes.
5. **Code Style**: Follow PEP 8 and use type hints where applicable.

## Development Setup
```bash
conda create -n brevitas-qat python=3.12
conda activate brevitas-qat
pip install -e ".[dev]"
```

## Commit Messages
Use conventional commits:
- `feat: add new quantizer`
- `fix: correct dimension mismatch in test`
- `refactor: split models into modular package`
- `docs: update documentation structure`

For more details, see [Conventions](developer/conventions.md).
