"""Nạp cấu hình YAML (hỗ trợ kế thừa qua khóa `base:`) và giải đường dẫn."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike = "config.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_cfg = load_config(path.parent / cfg.pop("base"))
        cfg = _deep_merge(base_cfg, cfg)
    cfg["_config_path"] = str(path)
    return cfg


def set_by_dotted(cfg: dict, dotted: str, value: Any) -> dict:
    """Trả về bản sao cfg với khóa dạng 'm2.window' được gán giá trị mới."""
    out = copy.deepcopy(cfg)
    node = out
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return out


class Paths:
    """Truy cập đường dẫn theo cấu hình: P.interim / 'x.parquet' ..."""

    def __init__(self, cfg: dict):
        p = cfg["paths"]
        root = Path(p["root"])
        self.root = root if root.is_absolute() else PROJECT_ROOT / root
        for key in ("raw", "interim", "processed", "labels", "benchmark", "models", "reports", "logs"):
            sub = Path(p[key])
            setattr(self, key, (self.root / sub).resolve())
        self.figures = self.reports / "figures"
        self.tables = self.reports / "tables"

    def ensure(self) -> "Paths":
        for d in (self.raw, self.interim, self.processed, self.labels, self.benchmark,
                  self.models, self.figures, self.tables, self.logs,
                  self.raw / "places", self.raw / "reviews", self.raw / "network", self.raw / "contributors"):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def resolve(self, rel: str) -> Path:
        """Đường dẫn tương đối so với root dữ liệu."""
        q = Path(rel)
        return q if q.is_absolute() else (self.root / q).resolve()
