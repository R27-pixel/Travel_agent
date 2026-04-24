"""
Small local web server for the AI Travel Planner frontend.

Run:
    python web_app.py
Then open:
    http://localhost:8000
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from multi_agent_system import run_travel_plan


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
HOST = "127.0.0.1"
PORT = 8000


class TravelPlannerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        route = unquote(self.path.split("?", 1)[0])
        if route == "/":
            route = "/index.html"

        file_path = (FRONTEND_DIR / route.lstrip("/")).resolve()
        if not str(file_path).startswith(str(FRONTEND_DIR.resolve())):
            self.send_error(403, "Forbidden")
            return

        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "Not found")
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(file_path.read_bytes())

    def do_POST(self) -> None:
        if self.path != "/api/plan":
            self.send_error(404, "Not found")
            return

        try:
            payload = self._read_json()
            destination = str(payload.get("destination", "")).strip()
            duration_days = int(payload.get("duration_days", 7))
            travel_style = str(payload.get("travel_style", "mid-range")).strip()
            interests = str(payload.get("interests", "")).strip()

            if not destination:
                raise ValueError("Destination is required.")
            if duration_days < 1 or duration_days > 30:
                raise ValueError("Duration must be between 1 and 30 days.")

            result = run_travel_plan(
                destination=destination,
                duration_days=duration_days,
                travel_style=travel_style,
                interests=interests,
            )

            self._send_json(
                {
                    "research_notes": result.get("research_notes", ""),
                    "itinerary": result.get("itinerary", ""),
                    "budget_breakdown": result.get("budget_breakdown", ""),
                    "final_report": result.get("final_report", ""),
                }
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body or "{}")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), TravelPlannerHandler)
    print(f"AI Travel Planner frontend is running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
