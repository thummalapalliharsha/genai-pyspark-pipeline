"""Main entrypoint to generate synthetic datasets and save them as Parquet.

Usage: python main.py

The script creates customers, products, and orders using
`SyntheticDataGenerator` and writes Parquet files into `data/raw/`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEFAULT_CONFIG, ensure_dirs
from src.data_generator import SyntheticDataGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("genai_pipeline.main")


def sizeof_fmt(num: int, suffix: str = "B") -> str:
    """Human-readable file size."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Pi{suffix}"


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Save DataFrame as Parquet and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce datasets and save as Parquet.")
    parser.add_argument("--customers", type=int, default=100_000, help="Number of customers to generate")
    parser.add_argument("--products", type=int, default=10_000, help="Number of products to generate")
    parser.add_argument("--orders", type=int, default=5_000_000, help="Number of orders to generate")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")

    args = parser.parse_args(argv)

    try:
        ensure_dirs()
        out_dir = DEFAULT_CONFIG.raw_dir

        logger.info("Starting generation: customers=%d products=%d orders=%d", args.customers, args.products, args.orders)
        start = time.time()

        gen = SyntheticDataGenerator(seed=args.seed)
        customers = gen.generate_customers(args.customers)
        products = gen.generate_products(args.products)
        orders = gen.generate_orders(customers, products, args.orders)

        # Save as Parquet
        cust_path = out_dir / "customers.parquet"
        prod_path = out_dir / "products.parquet"
        orders_path = out_dir / "orders.parquet"

        logger.info("Saving Parquet files to %s", out_dir)
        save_parquet(customers, cust_path)
        save_parquet(products, prod_path)
        save_parquet(orders, orders_path)

        duration = time.time() - start
        sizes = {"customers": os.path.getsize(cust_path), "products": os.path.getsize(prod_path), "orders": os.path.getsize(orders_path)}

        print(f"Generation completed in {duration:.1f} seconds")
        for k, v in sizes.items():
            print(f"{k}: {sizeof_fmt(v)} -> {out_dir / (k + '.parquet')}")

        return 0

    except Exception as exc:  # pragma: no cover - top-level error handling
        logger.exception("Generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
