"""Screenshots of the built console (desktop light and dark, phone) for the README."""

from __future__ import annotations

import http.server
import shutil
import socketserver
import threading
from functools import partial
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "web" / "dist"
OUT = ROOT / "docs" / "screenshots"


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


def main() -> None:
    handler = partial(Quiet, directory=str(DIST))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        OUT.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=shutil.which("google-chrome") or None)
            shots = [
                ("console", {"width": 1280, "height": 1000}, "light", False),
                ("console-dark", {"width": 1280, "height": 1000}, "dark", False),
                ("console-phone", {"width": 390, "height": 844}, "light", False),
            ]
            for name, size, scheme, full in shots:
                page = browser.new_page(viewport=size, device_scale_factor=1.5 if not full else 1, color_scheme=scheme, reduced_motion="reduce")
                page.goto(f"http://127.0.0.1:{port}/index.html")
                page.wait_for_selector("#fleet")
                page.wait_for_timeout(800)
                page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
            browser.close()
        httpd.shutdown()


if __name__ == "__main__":
    main()
