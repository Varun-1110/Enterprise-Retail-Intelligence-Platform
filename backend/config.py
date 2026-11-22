"""
Configuration settings for the Enterprise Retail Intelligence Platform Backend.
Handles file paths, database connection strings, and server parameters dynamically.
"""

import os
from pathlib import Path

# Project root directory (C:\Guvi\final_project or equivalent)
BASE_DIR = Path(__file__).resolve().parent.parent

# Database configuration
DB_PATH = Path(os.getenv("M5_DB_PATH", BASE_DIR / "m5_forecasting.db"))

# Model configuration
MODEL_PATH = Path(os.getenv("M5_MODEL_PATH", BASE_DIR / "lightgbm_model.pkl"))

# Backend REST API configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

# Default data loading limit for fast interactive dashboards
DEFAULT_SAMPLE_LIMIT = int(os.getenv("DEFAULT_SAMPLE_LIMIT", 50000))

# M5 Model Feature definitions
MODEL_FEATURES = [
    "day_of_week", "day_of_month", "week_of_year", "month", "year",
    "is_weekend", "is_month_start", "is_month_end",
    "lag_28", "lag_35", "lag_42", "lag_56",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_std_7",
    "sell_price", "price_change_%", "price_vs_dept_avg", "price_vs_store_avg",
    "has_event", "event_type_encoded", "snap", "days_to_christmas",
    "item_id", "dept_id", "cat_id"
]
