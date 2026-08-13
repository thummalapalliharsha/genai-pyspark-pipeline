from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Optional

BASE_DIR: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = BASE_DIR / "data" / "raw"
PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"

DEFAULT_CUSTOMERS = 100
DEFAULT_PRODUCTS = 50
DEFAULT_ORDERS = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("genai_pipeline.config")


def ensure_dirs() -> None:
    """Create raw and processed data directories if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured data directories exist: %s %s", RAW_DIR, PROCESSED_DIR)


@dataclass
class Config:
    """Configuration dataclass for pipeline paths and parameters."""

    base_dir: Path = BASE_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    num_customers: int = DEFAULT_CUSTOMERS
    num_products: int = DEFAULT_PRODUCTS
    num_orders: int = DEFAULT_ORDERS


DEFAULT_CONFIG: Config = Config()
