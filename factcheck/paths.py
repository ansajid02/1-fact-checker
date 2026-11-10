from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 1-fact-checker/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
