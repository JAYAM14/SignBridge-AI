"""
SignBridge AI — Single-command project launcher.
Starts the Flask backend and opens the frontend in a browser.
"""

import os
import sys
import subprocess
import threading
import webbrowser
import time
import signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


def check_dependencies():
    """Verify required Python packages are installed."""
    required = ["flask", "flask_cors", "cv2", "tensorflow", "numpy", "gtts", "sklearn"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


def check_model():
    """Warn if TFLite hand landmark model is not found."""
    model_path = os.path.join(BASE_DIR, "hand_landmark.tflite")
    if not os.path.exists(model_path):
        print("[WARNING] hand_landmark.tflite not found at project root.")
        print("  Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print("  Or place your hand_landmark.tflite in the project root.")


def start_backend():
    """Start the Flask backend server."""
    env = os.environ.copy()
    env["FLASK_ENV"] = "production"
    env["PYTHONPATH"] = BASE_DIR

    app_path = os.path.join(BACKEND_DIR, "app.py")
    proc = subprocess.Popen(
        [sys.executable, app_path],
        cwd=BASE_DIR,
        env=env,
    )
    return proc


def open_browser():
    """Open the frontend in the default browser after backend is ready."""
    time.sleep(2)
    frontend_path = os.path.join(FRONTEND_DIR, "index.html")
    # Try to open in browser
    try:
        webbrowser.open(f"http://localhost:5000")
        print("[INFO] Opened http://localhost:5000 in browser.")
    except Exception:
        print(f"[INFO] Open http://localhost:5000 in your browser.")


def main():
    print("=" * 60)
    print("  SignBridge AI — Real-Time ISL Communication Platform")
    print("=" * 60)

    check_dependencies()
    check_model()

    print("[INFO] Starting Flask backend on http://localhost:5000 ...")
    backend_proc = start_backend()

    # Open browser in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    def shutdown(sig, frame):
        print("\n[INFO] Shutting down SignBridge AI...")
        backend_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
