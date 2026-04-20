from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    data["_config_path"] = str(path)
    data["_config_dir"] = str(path.parent)
    return data


def resolve_path(path_str: str | Path, *, base_dir: str | Path | None = None) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    base = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    return str((base / path).resolve())


def ensure_parent_dir(path_str: str | Path) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def load_json(path_str: str | Path) -> Dict[str, Any]:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Dict[str, Any], path_str: str | Path) -> None:
    ensure_parent_dir(path_str)
    with open(path_str, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(text: str, allowed_labels: list[str]) -> str:
    labels_str = ", ".join(allowed_labels)
    return (
        "Classify the banking customer intent.\n"
        "Return only one label from the allowed labels.\n\n"
        f"Allowed labels:\n{labels_str}\n\n"
        f"Message:\n{text}"
    )
