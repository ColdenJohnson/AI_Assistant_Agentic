from __future__ import annotations

import time


class PhaseTimer:
    """Simple helper to log latency checkpoints in milliseconds."""

    def __init__(self):
        self._start_ts: float | None = None
        self._last_ts: float | None = None
        self._active = True

    def checkpoint(self, label: str):
        if not self._active:
            return
        now = time.perf_counter()
        if self._start_ts is None:
            self._start_ts = now
            delta_ms = 0.0
            total_ms = 0.0
        else:
            delta_ms = (now - self._last_ts) * 1000.0
            total_ms = (now - self._start_ts) * 1000.0
        self._last_ts = now
        print(f"[LATENCY] {label}: +{delta_ms:.1f} ms (total {total_ms:.1f} ms)", flush=True)

    def stop(self):
        self._active = False
