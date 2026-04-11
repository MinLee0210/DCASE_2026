"""I/O utilities for loading and saving various file formats."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
            return data if data is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")


def write_yaml(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    default_flow_style: bool = False,
    sort_keys: bool = False,
) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=default_flow_style,
                sort_keys=sort_keys,
                allow_unicode=True,
            )
    except IOError as e:
        raise IOError(f"Error writing YAML file {file_path}: {e}")
