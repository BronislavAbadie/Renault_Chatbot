#!/usr/bin/env python3
import os
import subprocess
import sys
import time


def main() -> int:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = os.getenv("BACKEND_PORT", "8000")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        backend_host,
        "--port",
        backend_port,
        "--reload",
    ]
    backend_proc = subprocess.Popen(backend_cmd)

    try:
        time.sleep(1)
        frontend_cmd = [sys.executable, os.path.join("Frontend", "frontend.py")]
        return subprocess.call(frontend_cmd)
    finally:
        if backend_proc.poll() is None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
