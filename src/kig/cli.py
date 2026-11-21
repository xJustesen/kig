"""Command line: `kig report` writes the report of one store and `kig serve` serves it on localhost."""

from __future__ import annotations

import argparse
import tempfile
import threading
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from kig.report import build_report
from kig.store import Store


class _Server(ThreadingHTTPServer):
    def __init__(self, store: Store, port: int, out: Path):
        super().__init__(("localhost", port), _Handler)
        self.store, self.out = store, out
        self.lock = threading.Lock()
        self.stamp = None

    def page(self) -> bytes:
        with self.lock:
            stats = [file.stat() for file in sorted(self.store.path.glob("*.parquet"))]
            stamp = [(stat.st_mtime_ns, stat.st_size) for stat in stats]
            if stamp != self.stamp:
                build_report(self.store, self.out)
                self.stamp = stamp
            return self.out.read_bytes()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        page = self.server.page()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)


def main(argv: list[str] | None = None) -> None:
    """Run the `kig` command with the arguments `argv` (list[str] | None); None reads the command line."""
    parser = argparse.ArgumentParser(prog="kig", description="Reports of a kig store.")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="write the report of a store as one HTML file")
    report.add_argument("store", help="directory of the store")
    report.add_argument("-o", "--out", default="report.html", help="HTML file to write (default: report.html)")
    report.add_argument("--title", default="kig report", help="title of the report")
    serve = commands.add_parser("serve", help="serve the report of a store on localhost")
    serve.add_argument("store", help="directory of the store")
    serve.add_argument("--port", type=int, default=8000, help="port to serve on (default: 8000)")
    args = parser.parse_args(argv)
    store = Store(args.store)
    if store.read("runs").empty:
        parser.error(f"no runs in the store {args.store}")

    if args.command == "report":
        print("wrote", build_report(store, args.out, args.title))
        return
    with (
        tempfile.TemporaryDirectory() as tmp,
        _Server(store, args.port, Path(tmp) / "report.html") as server,
    ):
        print(f"serving {args.store} at http://localhost:{server.server_address[1]}", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()
