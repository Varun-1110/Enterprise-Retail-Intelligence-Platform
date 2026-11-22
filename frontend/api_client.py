"""
Frontend API Client for Enterprise Retail Intelligence Platform.
Communicates with the backend REST API via HTTP requests.
Includes an intelligent fallback to backend service modules so the frontend
works reliably whether the backend server is running or running standalone.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import pandas as pd
import streamlit as st

# Add project root to sys.path so backend imports work from anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import API_BASE_URL
from backend import services
from backend import database


@st.cache_data(ttl=5)
def is_backend_online() -> bool:
    """Checks whether the backend REST API server is responsive."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/health", timeout=0.5)
        return resp.status_code == 200
    except Exception:
        return False


def render_backend_status():
    """Renders a status badge in the Streamlit sidebar indicating connection mode."""
    online = is_backend_online()
    if online:
        st.sidebar.success(f"🟢 **Backend API:** Connected (`{API_BASE_URL}`)")
    else:
        st.sidebar.info("🟡 **Backend Mode:** Direct Services (API offline)")


@st.cache_data(ttl=60)
def get_filters() -> Dict[str, Any]:
    """Retrieves available filter options."""
    if is_backend_online():
        try:
            resp = requests.get(f"{API_BASE_URL}/api/filters", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    return database.get_filter_options()


@st.cache_data(ttl=60)
def get_dashboard_payload(
    store: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieves complete dashboard data payload."""
    if is_backend_online():
        try:
            params = {}
            if store and store != "All":
                params["store"] = store
            if category and category != "All":
                params["category"] = category
            if department and department != "All":
                params["department"] = department
            resp = requests.get(f"{API_BASE_URL}/api/dashboard", params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    return services.get_dashboard_data(store, category, department)


@st.cache_data(ttl=60)
def get_analysis_payload() -> Dict[str, Any]:
    """Retrieves exploratory data analysis payload."""
    if is_backend_online():
        try:
            resp = requests.get(f"{API_BASE_URL}/api/analysis", timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    return services.get_analysis_data()


def post_prediction(feature_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Sends feature inputs to backend for sales demand prediction."""
    if is_backend_online():
        try:
            resp = requests.post(f"{API_BASE_URL}/api/predict", json=feature_dict, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    return services.predict_demand(feature_dict)


@st.cache_data(ttl=60)
def get_model_performance_payload() -> Dict[str, Any]:
    """Retrieves evaluation metrics and model details."""
    if is_backend_online():
        try:
            resp = requests.get(f"{API_BASE_URL}/api/model-performance", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    return services.get_model_performance()


def get_database_preview(
    search_item: Optional[str] = None,
    store: Optional[str] = None,
    category: Optional[str] = None,
    rows: int = 50
) -> Dict[str, Any]:
    """Retrieves filtered database records for table display."""
    if is_backend_online():
        try:
            params = {"rows": rows}
            if search_item:
                params["search"] = search_item
            if store and store != "All":
                params["store"] = store
            if category and category != "All":
                params["category"] = category
            resp = requests.get(f"{API_BASE_URL}/api/database-preview", params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    return services.query_database(search_item, store, category, preview_rows=rows)


@st.cache_data(ttl=120)
def get_raw_dataframe(limit: int = 50000) -> pd.DataFrame:
    """Provides direct DataFrame access for chart components requiring full series."""
    return database.load_dataset(limit=limit)
