"""Loads the single YAML config file that drives every pipeline stage."""
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Paths in default.yaml are relative to the workroom root (config/ lives one level below it).
    cfg["_project_root"] = str(config_path.resolve().parent.parent)
    return cfg


def resolve_path(cfg: dict, relative_path: str) -> Path:
    """Resolve a path from default.yaml, which are written relative to the workroom root."""
    return (Path(cfg["_project_root"]) / relative_path).resolve()
