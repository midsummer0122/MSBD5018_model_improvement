from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "api_key" / "api_config.yaml"


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            if not stripped.endswith(":"):
                raise ValueError(f"Unsupported YAML line {path}:{line_no}: {line}")
            current = stripped[:-1].strip()
            config[current] = {}
            continue
        if current is None or ":" not in stripped:
            raise ValueError(f"Unsupported YAML line {path}:{line_no}: {line}")
        key, raw_value = stripped.split(":", 1)
        config[current][key.strip()] = _parse_scalar(raw_value)
    return config


def load_yaml_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


@dataclass(slots=True)
class AppConfig:
    raw: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "AppConfig":
        return cls(raw=load_yaml_config(path))

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Missing config section: {name}")
        return value

