"""BingX tape recorder: our own venue-native market-data archive.

BingX publishes no historical tick data and no aggregator sells it — the only
way a BingX microstructure dataset ever exists is if we write down the feed we
already receive. This records every trade print and a throttled best-bid/ask
stream to append-only daily CSV files, built to survive power cuts:

  * the hot path only enqueues (never touches disk) — a dedicated writer
    THREAD does all IO, so recording can never stall the event loop;
  * files are append-only, flushed AND fsync'd every couple of seconds — a
    hard power cut loses at most that window, and a torn final line is
    tolerated by the reader (skip, don't crash);
  * finished days gzip in the background and the archive is pruned oldest-
    first to a disk cap, so it can run for months unattended.

Format (CSV, '#'-prefixed session markers):
  trades-YYYY-MM-DD.csv : ts_ms,price,qty,buyer_maker(0/1)
  book-YYYY-MM-DD.csv   : ts_ms,bid,ask
"""
from __future__ import annotations

import gzip
import logging
import os
import queue
import threading
import time
from pathlib import Path

log = logging.getLogger("tape")

FSYNC_EVERY_S = 2.0        # durability window: a power cut loses at most this
QUEUE_SOFT_CAP = 60_000    # beyond this the recorder drops (and counts) events
                           # rather than growing without bound — recording must
                           # never be the reason the bot runs out of memory


def _day_of(ts_ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts_ms / 1000))


class TapeRecorder:
    def __init__(self, root: Path, max_disk_mb: int = 2000, book_ms: int = 250):
        self.root = Path(root)
        self.max_disk_mb = max(50, int(max_disk_mb))
        self.book_ms = max(0, int(book_ms))
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._book_ts: dict[str, int] = {}
        # stats (ints under the GIL — good enough for a status line)
        self.events = 0
        self.dropped = 0
        self.bytes_written = 0
        self.files_rotated = 0

    # -------------------------------------------------------------- hot path

    def record_trade(self, symbol: str, ts_ms: int, price: float, qty: float,
                     buyer_maker: bool) -> None:
        self._put(symbol, "trades", ts_ms, f"{ts_ms},{price:g},{qty:g},{1 if buyer_maker else 0}")

    def record_book(self, symbol: str, ts_ms: int, bid: float, ask: float) -> None:
        last = self._book_ts.get(symbol, 0)
        if ts_ms - last < self.book_ms:
            return
        self._book_ts[symbol] = ts_ms
        self._put(symbol, "book", ts_ms, f"{ts_ms},{bid:g},{ask:g}")

    def _put(self, symbol: str, kind: str, ts_ms: int, line: str) -> None:
        if self._thread is None or self._stop.is_set():
            return
        if self._q.qsize() > QUEUE_SOFT_CAP:
            self.dropped += 1
            return
        self._q.put((symbol, kind, ts_ms, line))
        self.events += 1

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="tape-writer", daemon=True)
            self._thread.start()
            log.info("tape recorder started -> %s (cap %d MB)", self.root, self.max_disk_mb)

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=10.0)
        self._thread = None

    # -------------------------------------------------------------- writer

    def _run(self) -> None:
        open_files: dict[tuple[str, str, str], object] = {}   # (sym, kind, day) -> fh
        last_sync = time.monotonic()
        fresh = True
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                item = None
            try:
                if item is not None:
                    sym, kind, ts_ms, line = item
                    day = _day_of(ts_ms)
                    key = (sym, kind, day)
                    fh = open_files.get(key)
                    if fh is None:
                        fh = self._open(sym, kind, day, fresh)
                        open_files[key] = fh
                        self._rotate_done_days(open_files, sym, kind, day)
                    fh.write(line + "\n")
                    self.bytes_written += len(line) + 1
                if time.monotonic() - last_sync >= FSYNC_EVERY_S:
                    for fh in open_files.values():
                        try:
                            fh.flush()
                            os.fsync(fh.fileno())
                        except (OSError, ValueError):
                            pass
                    last_sync = time.monotonic()
                    fresh = False
                if self._stop.is_set() and self._q.qsize() == 0:
                    break
            except Exception:  # noqa: BLE001 — the recorder must outlive anything
                log.exception("tape writer error")
                time.sleep(1.0)
        for fh in open_files.values():
            try:
                fh.flush()
                os.fsync(fh.fileno())
                fh.close()
            except (OSError, ValueError):
                pass
        log.info("tape recorder stopped (%d events, %d dropped)", self.events, self.dropped)

    def _open(self, sym: str, kind: str, day: str, fresh: bool):
        d = self.root / sym
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{kind}-{day}.csv"
        fh = open(path, "a", encoding="utf-8", buffering=1024 * 64)
        if fresh or fh.tell() > 0:
            # session marker: every process (re)start leaves a visible seam so
            # later analysis knows where recording gaps live. '#' lines are
            # comments to the reader.
            fh.write(f"#session,{int(time.time() * 1000)}\n")
        return fh

    def _rotate_done_days(self, open_files: dict, sym: str, kind: str, today: str) -> None:
        """A new day's file just opened: gzip any older raw day for this
        (sym, kind), close its handle, and prune the archive to the disk cap."""
        for key in [k for k in list(open_files) if k[0] == sym and k[1] == kind and k[2] != today]:
            try:
                open_files[key].flush()
                open_files[key].close()
            except (OSError, ValueError):
                pass
            open_files.pop(key, None)
        try:
            for raw in sorted((self.root / sym).glob(f"{kind}-*.csv")):
                if raw.stem.split("-", 1)[1] >= today:
                    continue
                gz = raw.with_suffix(".csv.gz")
                with open(raw, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
                    dst.write(src.read())
                raw.unlink()
                self.files_rotated += 1
            self._prune()
        except Exception as e:  # noqa: BLE001
            log.warning("tape rotation failed: %s", e)

    def _prune(self) -> None:
        """Oldest compressed days go first once the archive exceeds the cap;
        raw (in-progress) files are never pruned."""
        files = sorted(self.root.glob("*/*.csv.gz"), key=lambda p: p.name.split("-", 1)[-1])
        total = sum(p.stat().st_size for p in files) + \
            sum(p.stat().st_size for p in self.root.glob("*/*.csv"))
        cap = self.max_disk_mb * 1024 * 1024
        for p in files:
            if total <= cap:
                break
            try:
                total -= p.stat().st_size
                p.unlink()
            except OSError:
                pass

    # -------------------------------------------------------------- status

    def stats(self) -> dict:
        try:
            disk = sum(p.stat().st_size for p in self.root.glob("*/*.csv*"))
        except OSError:
            disk = 0
        return {"events": self.events, "dropped": self.dropped,
                "bytes_session": self.bytes_written,
                "disk_mb": round(disk / 1e6, 1), "cap_mb": self.max_disk_mb,
                "queued": self._q.qsize(),
                "running": self._thread is not None and self._thread.is_alive()}


def read_day(root: Path, symbol: str, kind: str, day: str) -> list[list[float]]:
    """Tolerant reader for audits and tests: skips '#' session markers and any
    torn/malformed line (a power cut mid-write leaves at most one)."""
    base = Path(root) / symbol / f"{kind}-{day}.csv"
    if base.exists():
        text = base.read_text(encoding="utf-8", errors="replace")
    else:
        gz = base.with_suffix(".csv.gz")
        if not gz.exists():
            return []
        text = gzip.open(gz, "rt", encoding="utf-8", errors="replace").read()
    want = 4 if kind == "trades" else 3
    out: list[list[float]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) != want:
            continue
        try:
            out.append([float(x) for x in parts])
        except ValueError:
            continue
    return out
