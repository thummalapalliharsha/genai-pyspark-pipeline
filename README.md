# genai-pyspark-pipeline

Synthetic data generation and PySpark analytics pipeline.

Purpose
- Generate fake customer, product, and order data for testing
- Analyze the generated data using PySpark to produce business insights

Structure
- `src/` — Python source code
- `data/raw/` — generated CSV files
- `data/processed/` — analysis outputs
- `tests/` — pytest test cases
- `notebooks/` — example Jupyter notebooks

Quickstart
1. Create a virtualenv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Generate data and run analytics from Python REPL or a script:

```python
from src import data_generator, spark_analytics
from src.config import ensure_dirs

ensure_dirs()
data_generator.create_fake_data()
spark_analytics.run_all()
```

3. View outputs in `data/processed/`.

License: MIT
