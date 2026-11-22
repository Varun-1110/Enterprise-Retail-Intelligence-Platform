"""
REST API Application for Enterprise Retail Intelligence Platform.
Built using Starlette and Uvicorn. Exposes endpoints for KPIs, charts, data, and ML prediction.
"""

import json
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from backend.config import API_HOST, API_PORT
from backend import services
from backend import database


async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "Enterprise Retail Intelligence API",
        "version": "1.0.0"
    })


async def get_filter_options(request: Request) -> JSONResponse:
    """Endpoint returning available store, category, and department filters."""
    options = database.get_filter_options()
    return JSONResponse(options)


async def get_kpis(request: Request) -> JSONResponse:
    """Endpoint returning KPI metric values."""
    store = request.query_params.get("store")
    cat = request.query_params.get("category")
    dept = request.query_params.get("department")
    kpis = database.get_kpis(store, cat, dept)
    return JSONResponse(kpis)


async def get_daily_sales(request: Request) -> JSONResponse:
    """Endpoint returning daily sales timeseries."""
    store = request.query_params.get("store")
    cat = request.query_params.get("category")
    dept = request.query_params.get("department")
    trends = database.get_daily_sales_trend(store, cat, dept)
    return JSONResponse(trends)


async def get_category_sales(request: Request) -> JSONResponse:
    """Endpoint returning category sales distribution."""
    store = request.query_params.get("store")
    cat = request.query_params.get("category")
    dept = request.query_params.get("department")
    data = database.get_category_sales(store, cat, dept)
    return JSONResponse(data)


async def get_department_sales(request: Request) -> JSONResponse:
    """Endpoint returning department sales breakdown."""
    store = request.query_params.get("store")
    cat = request.query_params.get("category")
    dept = request.query_params.get("department")
    data = database.get_department_sales(store, cat, dept)
    return JSONResponse(data)


async def get_top_products(request: Request) -> JSONResponse:
    """Endpoint returning top N selling products."""
    n = int(request.query_params.get("n", 10))
    store = request.query_params.get("store")
    cat = request.query_params.get("category")
    dept = request.query_params.get("department")
    top = database.get_top_products(n, store, cat, dept)
    return JSONResponse(top)


async def get_monthly_sales(request: Request) -> JSONResponse:
    """Endpoint returning monthly sales aggregation."""
    data = database.get_monthly_sales()
    return JSONResponse(data)


async def get_weekday_sales(request: Request) -> JSONResponse:
    """Endpoint returning sales grouped by day of week."""
    data = database.get_weekday_sales()
    return JSONResponse(data)


async def get_store_sales(request: Request) -> JSONResponse:
    """Endpoint returning sales grouped by store."""
    data = database.get_store_sales()
    return JSONResponse(data)


async def get_correlation(request: Request) -> JSONResponse:
    """Endpoint returning correlation matrix."""
    data = database.get_correlation_matrix()
    return JSONResponse(data)


async def get_dashboard(request: Request) -> JSONResponse:
    """Aggregated endpoint returning complete payload for Dashboard page."""
    store = request.query_params.get("store")
    cat = request.query_params.get("category")
    dept = request.query_params.get("department")
    data = services.get_dashboard_data(store, cat, dept)
    return JSONResponse(data)


async def get_analysis(request: Request) -> JSONResponse:
    """Aggregated endpoint returning complete payload for Data Analysis page."""
    data = services.get_analysis_data()
    return JSONResponse(data)


async def search_database(request: Request) -> JSONResponse:
    """Endpoint for product search and data preview."""
    search_item = request.query_params.get("search")
    store = request.query_params.get("store")
    cat = request.query_params.get("category")
    limit = int(request.query_params.get("limit", 50000))
    preview_rows = int(request.query_params.get("rows", 50))
    res = services.query_database(search_item, store, cat, limit, preview_rows)
    return JSONResponse(res)


async def predict(request: Request) -> JSONResponse:
    """
    POST endpoint to predict future sales given feature attributes.
    Expects JSON body with product, date, price, and lag features.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    result = services.predict_demand(payload)
    return JSONResponse(result)


async def get_model_performance(request: Request) -> JSONResponse:
    """Endpoint returning evaluation metrics across all trained models."""
    metrics = services.get_model_performance()
    return JSONResponse(metrics)


routes = [
    Route("/api/health", health, methods=["GET"]),
    Route("/api/filters", get_filter_options, methods=["GET"]),
    Route("/api/kpi", get_kpis, methods=["GET"]),
    Route("/api/daily-sales", get_daily_sales, methods=["GET"]),
    Route("/api/category-sales", get_category_sales, methods=["GET"]),
    Route("/api/department-sales", get_department_sales, methods=["GET"]),
    Route("/api/top-products", get_top_products, methods=["GET"]),
    Route("/api/monthly-sales", get_monthly_sales, methods=["GET"]),
    Route("/api/weekday-sales", get_weekday_sales, methods=["GET"]),
    Route("/api/store-sales", get_store_sales, methods=["GET"]),
    Route("/api/correlation", get_correlation, methods=["GET"]),
    Route("/api/dashboard", get_dashboard, methods=["GET"]),
    Route("/api/analysis", get_analysis, methods=["GET"]),
    Route("/api/database-preview", search_database, methods=["GET"]),
    Route("/api/predict", predict, methods=["POST"]),
    Route("/api/model-performance", get_model_performance, methods=["GET"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=True, routes=routes, middleware=middleware)


def start_server():
    """Runs the backend server using uvicorn."""
    print(f"🚀 Starting Enterprise Retail Intelligence API on http://{API_HOST}:{API_PORT}")
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    start_server()
