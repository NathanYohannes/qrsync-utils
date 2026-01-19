#!/usr/bin/env python3
"""
QR Timestamp Beacon - FastAPI Server
Displays QR codes with embedded timestamps at configurable refresh rates.
"""

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="QR Timestamp Beacon",
    description="High-frequency QR code timestamp display for camera calibration",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main QR code display page."""
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="QR Timestamp Beacon Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    print(f"\n🔲 QR Timestamp Beacon")
    print(f"   Open in browser: http://localhost:{args.port}")
    print(f"   Network access:  http://0.0.0.0:{args.port}\n")

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

