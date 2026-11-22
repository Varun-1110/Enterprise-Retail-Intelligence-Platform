"""
Unified Launcher for Enterprise Retail Intelligence Platform.
Runs the decoupled Backend REST API and/or the Frontend Streamlit interface.

Usage:
    python run_app.py              # Launches both Backend and Frontend
    python run_app.py --frontend   # Launches Frontend only
    python run_app.py --backend    # Launches Backend REST API only
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def run_backend():
    """Starts the backend REST API server."""
    print("🚀 Starting Backend REST API on http://127.0.0.1:8000 ...")
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    return subprocess.Popen(cmd, cwd=str(ROOT_DIR))


def run_frontend():
    """Starts the frontend Streamlit application."""
    print("📊 Starting Frontend Streamlit Application ...")
    frontend_entry = ROOT_DIR / "frontend" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(frontend_entry),
        "--server.port", "8501",
        "--server.headless", "true"
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(description="Run Enterprise Retail Intelligence Platform")
    parser.add_argument("--backend", action="store_true", help="Launch Backend REST API only")
    parser.add_argument("--frontend", action="store_true", help="Launch Frontend UI only")
    args = parser.parse_args()

    procs = []

    try:
        if args.backend:
            p_back = run_backend()
            procs.append(p_back)
            p_back.wait()
        elif args.frontend:
            p_front = run_frontend()
            procs.append(p_front)
            p_front.wait()
        else:
            # Default: Start backend first, then frontend
            p_back = run_backend()
            procs.append(p_back)
            print("⏳ Waiting 2 seconds for backend initialization...")
            time.sleep(2)
            p_front = run_frontend()
            procs.append(p_front)

            # Wait for processes
            p_front.wait()
            p_back.terminate()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down platform services...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
 