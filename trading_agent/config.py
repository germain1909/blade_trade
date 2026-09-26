from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_BASE = "https://api.topstepx.com"
_DEFAULT_RTC = "https://rtc.topstepx.com"


@dataclass
class Config:
    username: str
    api_key: str
    base_url: str = _DEFAULT_BASE
    rtc_base_url: str = _DEFAULT_RTC
    data_dir: Path = field(default_factory=lambda: Path("data"))


def load_config() -> Config:
    username = os.getenv("TOPSTEP_USERNAME")
    api_key = os.getenv("TOPSTEP_API_KEY")

    if not username:
        raise ValueError("TOPSTEP_USERNAME not set in environment / .env")
    if not api_key:
        raise ValueError("TOPSTEP_API_KEY not set in environment / .env")

    return Config(
        username=username,
        api_key=api_key,
        base_url=os.getenv("TOPSTEP_BASE", _DEFAULT_BASE),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
    )
