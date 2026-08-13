from pathlib import Path
import pandas as pd
from src import data_generator
from src.config import DEFAULT_CONFIG, ensure_dirs


def test_generate_and_save(tmp_path: Path):
    ensure_dirs()
    out = data_generator.create_fake_data(num_customers=10, num_products=5, num_orders=20, seed=42)
    assert all(p.exists() for p in out.values())

    # Load back and check columns
    c = pd.read_csv(out['customers'])
    p = pd.read_csv(out['products'])
    o = pd.read_csv(out['orders'])

    assert 'customer_id' in c.columns
    assert 'product_id' in p.columns
    assert 'order_id' in o.columns
    assert len(c) == 10
    assert len(p) == 5
    assert len(o) == 20
