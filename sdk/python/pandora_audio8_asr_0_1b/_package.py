"""Model package directory discovery shared by SDK clients."""
from pathlib import Path
import os


class ModelPackageError(RuntimeError):
    """The selected model package is absent, ambiguous or incomplete."""
    code = -2


def _config_name():
    return "manifest.json" if os.environ.get("AUDIO8_LEGACY_MANIFEST") == "1" else "models.json"


def resolve_model_dir(model_dir=None, *, default_container=None):
    """Resolve an explicit root or the unique immediate default package."""
    config = _config_name()
    if model_dir is not None:
        if not os.fspath(model_dir):
            raise ModelPackageError("Explicit model directory must not be empty")
        root = Path(model_dir).expanduser().resolve()
        if not (root / config).is_file():
            raise ModelPackageError(f"Model configuration not found: {root / config}. Select its package root.")
        return root
    if default_container is None:
        raise ModelPackageError("Select a model package explicitly with model_dir; the installed SDK does not guess a checkout")
    root = Path(default_container).expanduser().resolve()
    if (root / config).is_file():
        return root
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p / config).is_file()) if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ModelPackageError("Multiple model packages: " + ", ".join(map(str, candidates)) + ". Select one explicitly.")
    raise ModelPackageError(f"No model package at {root}/{config} or its immediate children. See docs/MODEL_SETUP.md.")
