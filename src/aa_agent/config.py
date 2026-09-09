"""Typed configuration. Everything tunable lives in config/config.yaml."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_PATH = Path(os.environ.get("AA_CONFIG", "config/config.yaml"))


class ProjectCfg(BaseModel):
    brand: str
    seed: int
    cache_path: Path


class ProviderCfg(BaseModel):
    name: str = ""
    base_url: str
    api_key_env: str
    rpm: int
    tpm: int
    rpd: int | None = None
    max_context_tokens: int
    enabled: bool = True

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class ModelRef(BaseModel):
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


class RoleCfg(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 512
    primary: ModelRef
    fallbacks: list[ModelRef] = Field(default_factory=list)

    @property
    def chain(self) -> list[ModelRef]:
        return [self.primary, *self.fallbacks]


class EmbeddingCfg(BaseModel):
    model: str
    dim: int
    dtype: str


class DataCfg(BaseModel):
    raw_csv: Path
    brand_parquet: Path
    split_quantile: float
    near_duplicate_cosine: float


class GoldenCfg(BaseModel):
    size: int
    stratum_a: int
    stratum_b: int


class Config(BaseModel):
    project: ProjectCfg
    providers: dict[str, ProviderCfg]
    roles: dict[str, RoleCfg]
    embedding: EmbeddingCfg
    data: DataCfg
    golden: GoldenCfg

    def provider(self, name: str) -> ProviderCfg:
        if name not in self.providers:
            raise KeyError(f"unknown provider {name!r}; have {sorted(self.providers)}")
        return self.providers[name]

    def role(self, name: str) -> RoleCfg:
        if name not in self.roles:
            raise KeyError(f"unknown role {name!r}; have {sorted(self.roles)}")
        return self.roles[name]


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    for name, block in raw.get("providers", {}).items():
        block["name"] = name
    return Config.model_validate(raw)
