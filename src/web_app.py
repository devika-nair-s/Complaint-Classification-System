"""Serve a small local showcase UI for the complaint classifier."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import sys

import joblib


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
MODEL_PATH = ROOT / "models" / "complaint_classifier.pkl"
MARATHI_MODEL_PATH = ROOT / "models" / "marathi_complaint_classifier.pkl"
MAX_BODY_BYTES = 8192

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.test_model import classify_complaint  # noqa: E402


class ShowcaseHandler(BaseHTTPRequestHandler):
    model = None
    marathi_model = None
    assets = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    }

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; "
            "img-src 'self' data:; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self):
        asset = self.assets.get(self.path)
        if asset is None:
            self._json(404, {"error": "Not found"})
            return
        filename, content_type = asset
        body = (WEB_ROOT / filename).read_bytes()
        self._send(200, body, content_type)

    def do_POST(self):
        if self.path != "/api/classify":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "Invalid request"})
            return
        if length < 1 or length > MAX_BODY_BYTES:
            self._json(413, {"error": "Request must be under 8 KB."})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "Please send a valid JSON request."})
            return
        complaint = payload.get("complaint") if isinstance(payload, dict) else None
        if not isinstance(complaint, str) or len(complaint) > 2000:
            self._json(400, {"error": "Enter a complaint of up to 2,000 characters."})
            return
        self._json(
            200,
            classify_complaint(
                self.model,
                complaint,
                marathi_model=self.marathi_model,
                language=payload.get("language"),
            ),
        )

    def log_message(self, format, *args):
        # Avoid writing complaint text or other request details to the console.
        if self.path not in ("/api/classify",):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Run the complaint classification showcase.")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: localhost)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run: .venv/bin/python src/train.py")

    ShowcaseHandler.model = joblib.load(MODEL_PATH)
    if MARATHI_MODEL_PATH.exists():
        ShowcaseHandler.marathi_model = joblib.load(MARATHI_MODEL_PATH)
    server = ThreadingHTTPServer((args.host, args.port), ShowcaseHandler)
    print(f"Showcase ready at http://{args.host}:{args.port} (press Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShowcase stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
