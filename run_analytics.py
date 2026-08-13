"""Run Spark analytics using `SalesAnalytics` and display results.

Usage: python run_analytics.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.config import DEFAULT_CONFIG
from src.spark_analytics import SalesAnalytics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("genai_pipeline.run_analytics")


def _show_and_time(df: Any, label: str) -> float:
    start = time.perf_counter()
    # DataFrame may be a PySpark DataFrame
    try:
        df.show(truncate=False, n=20)
    except Exception:
        # Fallback: print type and head if pandas
        try:
            print(df.head(20))
        except Exception:
            logger.exception("Unable to display result for %s", label)
    duration = time.perf_counter() - start
    print(f"{label} took {duration:.2f}s")
    return duration


def main(argv: list[str] | None = None) -> int:
    spark = None
    try:
        spark = SalesAnalytics.create_spark_session()

        raw = DEFAULT_CONFIG.raw_dir
        cust_path = raw / "customers.parquet"
        prod_path = raw / "products.parquet"
        orders_path = raw / "orders.parquet"

        logger.info("Loading Parquet files from %s", raw)
        orders_df = SalesAnalytics.load_parquet(spark, orders_path)
        products_df = SalesAnalytics.load_parquet(spark, prod_path)

        # Top customers
        print("\nTop customers by revenue:")
        top = SalesAnalytics.top_customers_by_revenue(orders_df, products_df, n=10)
        _show_and_time(top, "top_customers_by_revenue")

        # Sales by category
        print("\nSales by category:")
        cat = SalesAnalytics.sales_by_category(orders_df, products_df)
        _show_and_time(cat, "sales_by_category")

        # Monthly trends
        print("\nMonthly trends:")
        trends = SalesAnalytics.monthly_trends(orders_df, products_df)
        _show_and_time(trends, "monthly_trends")

        return 0

    except Exception as exc:  # pragma: no cover - top-level runner
        logger.exception("Analytics run failed: %s", exc)
        return 1

    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
