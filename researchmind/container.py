"""Run preparation, API and UI in one container, within the three-container budget."""

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    children = []

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        preparation = subprocess.Popen([sys.executable, "-m", "researchmind.bootstrap"])
        children.append(preparation)
        if preparation.wait() != 0:
            raise RuntimeError("Model/index preparation failed")
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "researchmind.api:app", "--host", "0.0.0.0", "--port", "8000"]
        )
        children.append(api)
        deadline = time.monotonic() + 180
        while True:
            if api.poll() is not None:
                raise RuntimeError("API exited before becoming ready")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("API readiness timeout")
            time.sleep(1)
        ui = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "ui.py",
                "--server.address=0.0.0.0",
                "--server.port=8501",
                "--server.headless=true",
            ],
            env=dict(os.environ, RESEARCH_API_URL="http://127.0.0.1:8000"),
        )
        children.append(ui)
        while api.poll() is None and ui.poll() is None:
            time.sleep(1)
        raise RuntimeError("An application process exited; restarting the container is required")
    finally:
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
        for process in reversed(children):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
