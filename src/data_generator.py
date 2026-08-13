"""Data generation utilities for synthetic e-commerce datasets.

Provides a `SyntheticDataGenerator` class that can produce large-scale
synthetic customers, products, and orders datasets. The implementation uses
Faker for realistic names/emails, NumPy for distributions (including Pareto),
and tqdm for progress bars. All primary methods return pandas DataFrames.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from src.config import DEFAULT_CONFIG, ensure_dirs

import logging
import numpy as np
import pandas as pd
from faker import Faker

# Optional tqdm for progress bars; fall back to identity if not installed
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without tqdm
    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger("genai_pipeline.data_generator")


@dataclass
class SyntheticDataGenerator:
    """Generate synthetic e-commerce datasets at scale.

    Attributes:
        seed: Optional random seed for reproducibility.
    """

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._fake = Faker()
        if self.seed is not None:
            Faker.seed(self.seed)
            self._rng = np.random.default_rng(self.seed)

    def generate_customers(self, num_customers: int = 100_000) -> pd.DataFrame:
        """Generate customers DataFrame.

        Columns: `customer_id`, `name`, `email`, `age`, `city`, `country`, `registration_date`.

        Age is drawn from a normal distribution centered at 35 and clipped to [18,90].
        """
        logger.info("Generating %d customers", num_customers)
        rows = []
        for cid in tqdm(range(1, num_customers + 1), desc="customers"):
            name = self._fake.name()
            email = self._fake.email()
            age = int(np.clip(self._rng.normal(35, 10), 18, 90))
            city = self._fake.city()
            country = self._fake.country()
            reg_date = self._fake.date_between(start_date="-5y", end_date="today").isoformat()
            rows.append((cid, name, email, age, city, country, reg_date))

        df = pd.DataFrame(rows, columns=["customer_id", "name", "email", "age", "city", "country", "registration_date"])
        logger.info("Customers generation complete: %d rows", len(df))
        return df

    def generate_products(self, num_products: int = 10_000) -> pd.DataFrame:
        """Generate products DataFrame.

        Columns: `product_id`, `name`, `category`, `price`, `stock`, `rating`.
        """
        logger.info("Generating %d products", num_products)
        categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
        rows = []
        for pid in tqdm(range(1, num_products + 1), desc="products"):
            name = f"{self._fake.word().capitalize()} {self._fake.word().capitalize()}"
            category = self._rng.choice(categories)
            price = round(self._rng.uniform(10.0, 500.0), 2)
            stock = int(self._rng.integers(0, 5000))
            rating = round(float(self._rng.uniform(1.0, 5.0)), 2)
            rows.append((pid, name, category, price, stock, rating))

        df = pd.DataFrame(rows, columns=["product_id", "name", "category", "price", "stock", "rating"])
        logger.info("Products generation complete: %d rows", len(df))
        return df

    def _customer_order_distribution(self, num_customers: int, num_orders: int) -> np.ndarray:
        """Produce an integer array of order counts per customer using a Pareto-shaped distribution.

        The implementation samples Pareto weights and uses a multinomial draw to
        allocate `num_orders` across `num_customers`. This yields a heavy-tailed
        distribution where a minority of customers account for a large share of orders.
        """
        # Pareto shape parameter controls tail-heaviness; tuned to approximate 80/20
        alpha = 1.5
        weights = self._rng.pareto(alpha, size=num_customers) + 1e-6
        probs = weights / weights.sum()
        counts = self._rng.multinomial(num_orders, probs)
        return counts

    def generate_orders(self, customers: pd.DataFrame, products: pd.DataFrame, num_orders: int = 1_000_000) -> pd.DataFrame:
        """Generate orders DataFrame linking customers and products.

        Columns: `order_id`, `customer_id`, `product_id`, `quantity`, `order_date`, `total`.

        Uses a Pareto distribution to allocate orders across customers so that a
        small fraction of customers account for a large fraction of orders.
        """
        num_customers = int(customers["customer_id"].max())
        num_products = int(products["product_id"].max())

        logger.info("Allocating %d orders across %d customers", num_orders, num_customers)
        counts = self._customer_order_distribution(num_customers, num_orders)

        # Build customer_id array by repeating customer ids according to their counts
        customer_ids = np.repeat(np.arange(1, num_customers + 1), counts)

        if len(customer_ids) != num_orders:
            # Safety: if multinomial produced rounding issues
            customer_ids = np.resize(customer_ids, num_orders)

        logger.info("Selecting products and quantities for %d orders", num_orders)
        product_ids = self._rng.integers(1, num_products + 1, size=num_orders)
        quantities = self._rng.integers(1, 11, size=num_orders)

        # Generate order timestamps between each customer's registration and now
        now_ts = datetime.utcnow().timestamp()
        two_years_ago_ts = (datetime.utcnow() - timedelta(days=365 * 2)).timestamp()
        order_ts = self._rng.uniform(two_years_ago_ts, now_ts, size=num_orders)
        order_dates = pd.to_datetime(order_ts, unit="s")

        # Map product_id to price using numpy indexing for speed
        prices = products.sort_values("product_id")["price"].to_numpy()
        # product_id starts at 1, array index starts at 0
        product_price_for_order = prices[product_ids - 1]

        totals = np.round(product_price_for_order * quantities, 2)

        df = pd.DataFrame(
            {
                "order_id": np.arange(1, num_orders + 1, dtype=np.int64),
                "customer_id": customer_ids,
                "product_id": product_ids,
                "quantity": quantities,
                "order_date": order_dates.astype(str),
                "total": totals,
            }
        )

        logger.info("Orders generation complete: %d rows", len(df))
        return df

    def generate_all(self, num_customers: int = 100_000, num_products: int = 10_000, num_orders: int = 1_000_000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate customers, products, and orders and return the three DataFrames.
        """
        customers = self.generate_customers(num_customers)
        products = self.generate_products(num_products)
        orders = self.generate_orders(customers, products, num_orders)
        return customers, products, orders


# Backwards-compatible helper using the class above
def create_fake_data(num_customers: int = 100_000, num_products: int = 10_000, num_orders: int = 1_000_000, seed: Optional[int] = None) -> Dict[str, Path]:
    """Create and return synthetic customers/products/orders DataFrames.

    Returns a dict with keys `customers`, `products`, `orders` mapping to DataFrames.
    """
    gen = SyntheticDataGenerator(seed=seed)
    customers, products, orders = gen.generate_all(num_customers=num_customers, num_products=num_products, num_orders=num_orders)

    # Save CSVs for backward compatibility with existing scripts/tests
    ensure_dirs()
    out_dir = DEFAULT_CONFIG.raw_dir
    customer_path = out_dir / "customers.csv"
    product_path = out_dir / "products.csv"
    orders_path = out_dir / "orders.csv"

    customers.to_csv(customer_path, index=False)
    products.to_csv(product_path, index=False)
    orders.to_csv(orders_path, index=False)

    logger.info("Saved generated datasets to %s", out_dir)
    return {"customers": customer_path, "products": product_path, "orders": orders_path}
