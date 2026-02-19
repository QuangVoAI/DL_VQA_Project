"""Configuration system using dataclasses + YAML loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml


@dataclass
class DataConfig:
    """Data-related configuration."""
    hf_id: str = "HuggingFaceM4/A-OKVQA"
    train_ratio: float = 0.85
    freq_threshold: int = 2
    image_size: int = 224
    expand_rationales: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    embed_size: int = 300
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    use_pretrained_cnn: bool = True
    use_attention: bool = True


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    patience: int = 5
    beam_width: int = 5
    len_alpha: float = 0.6
    tf_start: float = 1.0
    tf_end: float = 0.5
    scheduler: str = "cosine"
    eta_min: float = 1e-6
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class Config:
    """Top-level project configuration.

    Can be loaded from YAML or constructed programmatically.

    Example usage::

        cfg = Config.from_yaml("configs/default.yaml")
        print(cfg.train.learning_rate)  # 3e-4
    """
    seed: int = 42
    device: str = "auto"
    log_dir: str = "logs"
    ckpt_dir: str = "checkpoints"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Model variant definitions
    model_variants: dict = field(default_factory=lambda: {
        "M1_Scratch_NoAttn":    {"use_pretrained_cnn": False, "use_attention": False},
        "M2_Scratch_Attn":      {"use_pretrained_cnn": False, "use_attention": True},
        "M3_Pretrained_NoAttn": {"use_pretrained_cnn": True,  "use_attention": False},
        "M4_Pretrained_Attn":   {"use_pretrained_cnn": True,  "use_attention": True},
    })

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Config instance with values from YAML.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        if "seed" in raw:
            cfg.seed = raw["seed"]
        if "device" in raw:
            cfg.device = raw["device"]
        if "log_dir" in raw:
            cfg.log_dir = raw["log_dir"]
        if "ckpt_dir" in raw:
            cfg.ckpt_dir = raw["ckpt_dir"]

        if "data" in raw:
            for k, v in raw["data"].items():
                if hasattr(cfg.data, k):
                    setattr(cfg.data, k, v)

        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)

        if "train" in raw:
            for k, v in raw["train"].items():
                if hasattr(cfg.train, k):
                    setattr(cfg.train, k, v)

        if "model_variants" in raw:
            cfg.model_variants = raw["model_variants"]

        return cfg

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to nested dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)
