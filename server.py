#!/usr/bin/env python3
"""
Lokaler Bild-Viewer-Server
==========================
Startet einen einfachen HTTP-Server, der viewer.html zusammen mit den
Bild-Ordnern "Fahrzeug_erkannt" und "Kein_Fahrzeug" ausliefert und
einen JSON-API-Endpunkt bereitstellt.

Benutzung:
    python server.py          # startet auf http://localhost:8080
    python server.py 9090     # alternativer Port

Danach einfach http://localhost:8080 im Browser öffnen – die Bilder
werden automatisch geladen, ohne dass ein Ordner manuell ausgewählt
werden muss.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote, unquote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDERS = ["Fahrzeug_erkannt", "Kein_Fahrzeug"]


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def build_image_list() -> list[dict]:
    """Scannt beide Bild-Ordner und gibt eine sortierte Liste zurück."""
    records = []
    for folder in IMAGE_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg")):
                continue
            base = os.path.splitext(filename)[0]
            json_path = os.path.join(folder_path, base + ".json")
            meta = _read_json(json_path)
            if meta is None:
                meta = {
                    "timestamp": base,
                    "datetime_display": base,
                    "datetime_iso": base,
                    "has_vehicles": folder == "Fahrzeug_erkannt",
                    "vehicle_count": None if folder == "Fahrzeug_erkannt" else 0,
                    "detections": [],
                    "notification_title": None,
                    "notification_message": None,
                }
            records.append(
                {
                    "folder": folder,
                    "filename": filename,
                    "url": quote(folder) + "/" + quote(filename),
                    "meta": meta,
                }
            )
    # Neueste zuerst (nach datetime_iso oder timestamp)
    records.sort(
        key=lambda r: r["meta"].get("datetime_iso") or r["meta"].get("timestamp") or "",
        reverse=True,
    )
    return records


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN001
        # Minimales Logging
        print(f"  {self.address_string()}  {fmt % args}")

    def do_GET(self):  # noqa: N802
        path = unquote(self.path.split("?")[0])

        # ── API: Bildliste ───────────────────────────────────────────────
        if path == "/api/images":
            data = json.dumps(build_image_list(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # ── Statische Dateien ────────────────────────────────────────────
        # Sicherheits-Check: keine Pfad-Traversal-Versuche
        # os.path.realpath löst Symlinks auf; commonpath prüft korrekt ohne
        # startswith-Falsch-Positiv bei ähnlichen Ordnernamen.
        candidate = os.path.realpath(
            os.path.normpath(os.path.join(BASE_DIR, path.lstrip("/")))
        )
        try:
            common = os.path.commonpath([BASE_DIR, candidate])
        except ValueError:
            self._send_error(403, "Forbidden")
            return
        if common != BASE_DIR:
            self._send_error(403, "Forbidden")
            return
        real_path = candidate

        # Root → viewer.html
        if path == "/" or path == "":
            real_path = os.path.join(BASE_DIR, "viewer.html")

        if not os.path.isfile(real_path):
            self._send_error(404, "Not Found")
            return

        # Content-Type bestimmen
        ext = os.path.splitext(real_path)[1].lower()
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".json": "application/json; charset=utf-8",
            ".js":   "application/javascript",
            ".css":  "text/css",
            ".png":  "image/png",
        }
        ctype = content_types.get(ext, "application/octet-stream")

        with open(real_path, "rb") as fh:
            data = fh.read()

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, message: str) -> None:
        body = message.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), Handler)
    url = f"http://localhost:{port}"
    print()
    print("=" * 60)
    print("  MFG Rüsselbach – Bild-Viewer-Server")
    print(f"  Adresse : {url}")
    print("  Stoppen : Strg+C")
    print("=" * 60)
    print()
    # Browser automatisch öffnen
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer gestoppt.")


if __name__ == "__main__":
    main()
