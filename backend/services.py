"""
Business Logic & Service Layer for Enterprise Retail Intelligence Platform.
Orchestrates data operations, aggregations, and ML model predictions.
"""

from typing import Dict, Any, List, Optional
from backend import database
from backend import ml_model


def get_dashboard_data(
    store: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 50000
) -> Dict[str, Any]:
    """Retrieves all aggregated data needed by the Dashboard page in one call."""
    kpis = database.get_kpis(store, category, department, limit)
    daily_trends = database.get_daily_sales_trend(store, category, department, limit)
    cat_sales = database.get_category_sales(store, category, department, limit)
    dept_sales = database.get_department_sales(store, category, department, limit)
    top_products = database.get_top_products(10, store, category, department, limit)
    filters = database.get_filter_options(limit)

    return {
        "kpis": kpis,
        "daily_trends": daily_trends,
        "category_sales": cat_sales,
        "department_sales": dept_sales,
        "top_products": top_products,
        "filter_options": filters
    }


def get_analysis_data(limit: int = 50000) -> Dict[str, Any]:
    """Retrieves all analytical trends for the Data Analysis page."""
    df = database.load_dataset(limit=limit)
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum())
    }
    monthly = database.get_monthly_sales(limit)
    weekday = database.get_weekday_sales(limit)
    stores = database.get_store_sales(limit)
    categories = database.get_category_sales(limit=limit)
    top_products = database.get_top_products(10, limit=limit)
    correlation = database.get_correlation_matrix(limit)

    return {
        "summary": summary,
        "monthly_sales": monthly,
        "weekday_sales": weekday,
        "store_sales": stores,
        "category_sales": categories,
        "top_products": top_products,
        "correlation": correlation
    }


def predict_demand(feature_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Executes sales demand prediction."""
    return ml_model.predict_sales(feature_dict)


def get_model_performance() -> Dict[str, Any]:
    """Retrieves model evaluation comparison and details."""
    return ml_model.get_model_evaluation_metrics()


def query_database(
    search_item: Optional[str] = None,
    store: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50000,
    preview_rows: int = 50
) -> Dict[str, Any]:
    """Searches and previews database records."""
    return database.search_and_preview_data(search_item, store, category, limit, preview_rows)
