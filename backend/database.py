"""
Database access layer for Enterprise Retail Intelligence Platform.
Executes queries against SQLite database and processes retail forecasting data.
"""

import sqlite3
import pandas as pd
from typing import Optional, Dict, Any, List
from backend.config import DB_PATH, DEFAULT_SAMPLE_LIMIT

# Global in-memory cache for fast querying
_cached_df: Optional[pd.DataFrame] = None
_cached_limit: Optional[int] = None


def get_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


def load_dataset(limit: int = DEFAULT_SAMPLE_LIMIT, force_reload: bool = False) -> pd.DataFrame:
    """
    Loads retail sales dataset from SQLite with caching.
    Ensures datetime and numerical columns are formatted properly.
    """
    global _cached_df, _cached_limit

    if _cached_df is not None and _cached_limit == limit and not force_reload:
        return _cached_df.copy()

    conn = get_connection()
    query = f"""
    SELECT
        date,
        sales,
        sell_price,
        item_id,
        dept_id,
        cat_id,
        store_id
    FROM final_dataset
    LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_name"] = df["date"].dt.day_name()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce").fillna(0.0)

    _cached_df = df
    _cached_limit = limit
    return _cached_df.copy()


def get_filter_options(limit: int = DEFAULT_SAMPLE_LIMIT) -> Dict[str, List[str]]:
    """Returns available unique stores, categories, and departments."""
    df = load_dataset(limit=limit)
    return {
        "stores": ["All"] + sorted([str(s) for s in df["store_id"].dropna().unique()]),
        "categories": ["All"] + sorted([str(c) for c in df["cat_id"].dropna().unique()]),
        "departments": ["All"] + sorted([str(d) for d in df["dept_id"].dropna().unique()]),
    }


def filter_dataset(
    df: pd.DataFrame,
    store: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None
) -> pd.DataFrame:
    """Applies filter criteria to dataset."""
    filtered = df
    if store and store != "All":
        filtered = filtered[filtered["store_id"] == store]
    if category and category != "All":
        filtered = filtered[filtered["cat_id"] == category]
    if department and department != "All":
        filtered = filtered[filtered["dept_id"] == department]
    return filtered


def get_kpis(
    store: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = DEFAULT_SAMPLE_LIMIT
) -> Dict[str, Any]:
    """Calculates KPI card metrics."""
    df = load_dataset(limit=limit)
    df = filter_dataset(df, store, category, department)

    total_sales = float(df["sales"].sum()) if not df.empty else 0.0
    unique_products = int(df["item_id"].nunique()) if not df.empty else 0
    unique_departments = int(df["dept_id"].nunique()) if not df.empty else 0
    avg_price = float(df["sell_price"].mean()) if not df.empty else 0.0

    return {
        "total_sales": total_sales,
        "unique_products": unique_products,
        "unique_departments": unique_departments,
        "avg_price": round(avg_price, 2),
        "total_records": len(df)
    }


def get_daily_sales_trend(
    store: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = DEFAULT_SAMPLE_LIMIT
) -> List[Dict[str, Any]]:
    """Calculates daily aggregated sales trend."""
    df = load_dataset(limit=limit)
    df = filter_dataset(df, store, category, department)
    if df.empty:
        return []
    daily = df.groupby(df["date"].dt.strftime("%Y-%m-%d"))["sales"].sum().reset_index()
    daily.columns = ["date", "sales"]
    return daily.to_dict(orient="records")


def get_category_sales(
    store: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = DEFAULT_SAMPLE_LIMIT
) -> List[Dict[str, Any]]:
    """Calculates total sales grouped by category."""
    df = load_dataset(limit=limit)
    df = filter_dataset(df, store, category, department)
    if df.empty:
        return []
    cat_sales = df.groupby("cat_id")["sales"].sum().reset_index()
    return cat_sales.to_dict(orient="records")


def get_department_sales(
    store: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = DEFAULT_SAMPLE_LIMIT
) -> List[Dict[str, Any]]:
    """Calculates total sales grouped by department."""
    df = load_dataset(limit=limit)
    df = filter_dataset(df, store, category, department)
    if df.empty:
        return []
    dept_sales = df.groupby("dept_id")["sales"].sum().reset_index()
    return dept_sales.to_dict(orient="records")


def get_top_products(
    n: int = 10,
    store: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = DEFAULT_SAMPLE_LIMIT
) -> List[Dict[str, Any]]:
    """Returns top N selling products."""
    df = load_dataset(limit=limit)
    df = filter_dataset(df, store, category, department)
    if df.empty:
        return []
    top = (
        df.groupby("item_id")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    return top.to_dict(orient="records")


def get_monthly_sales(limit: int = DEFAULT_SAMPLE_LIMIT) -> List[Dict[str, Any]]:
    """Calculates sales grouped by month."""
    df = load_dataset(limit=limit)
    if df.empty:
        return []
    monthly = df.groupby("month")["sales"].sum().reset_index()
    return monthly.to_dict(orient="records")


def get_weekday_sales(limit: int = DEFAULT_SAMPLE_LIMIT) -> List[Dict[str, Any]]:
    """Calculates sales grouped by day of week in chronological order."""
    df = load_dataset(limit=limit)
    if df.empty:
        return []
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = df.groupby("day_name")["sales"].sum().reindex(days_order).fillna(0).reset_index()
    weekday.columns = ["day_name", "sales"]
    return weekday.to_dict(orient="records")


def get_store_sales(limit: int = DEFAULT_SAMPLE_LIMIT) -> List[Dict[str, Any]]:
    """Calculates sales grouped by store."""
    df = load_dataset(limit=limit)
    if df.empty:
        return []
    store_sales = df.groupby("store_id")["sales"].sum().reset_index()
    return store_sales.to_dict(orient="records")


def get_correlation_matrix(limit: int = DEFAULT_SAMPLE_LIMIT) -> Dict[str, Dict[str, float]]:
    """Calculates correlation between sales and sell_price."""
    df = load_dataset(limit=limit)
    if df.empty:
        return {}
    corr = df[["sales", "sell_price"]].corr()
    return corr.to_dict()


def search_and_preview_data(
    search_item: Optional[str] = None,
    store: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = DEFAULT_SAMPLE_LIMIT,
    preview_rows: int = 50
) -> Dict[str, Any]:
    """Provides product search, filter, and preview table."""
    df = load_dataset(limit=limit)
    if search_item:
        df = df[df["item_id"].astype(str).str.contains(search_item, case=False, na=False)]
    if store and store != "All":
        df = df[df["store_id"] == store]
    if category and category != "All":
        df = df[df["cat_id"] == category]

    total_matched = len(df)
    preview = df.head(preview_rows).copy()
    preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")

    return {
        "total_matched": total_matched,
        "preview": preview.to_dict(orient="records"),
        "total_stores": int(df["store_id"].nunique()) if not df.empty else 0,
        "total_products": int(df["item_id"].nunique()) if not df.empty else 0
    }
