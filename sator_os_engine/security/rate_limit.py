from __future__ import annotations

import time
from collections import defaultdict, deque


class SimpleRateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)  # (api_key, ip) -> timestamps

    def allow(self, api_key: str, ip: str) -> bool:
        now = time.time()
        window_start = now - 60.0
        dq = self._events[(api_key, ip)]
        while dq and dq[0] < window_start:
            dq.popleft()
        if len(dq) >= self.per_minute:
            return False
        dq.append(now)
        return True

    def sweep(self, max_keys: int | None) -> int:
        """Drop keys with an empty event window and enforce ``max_keys`` by deleting lexicographically first keys first."""
        now = time.time()
        window_start = now - 60.0
        removed = 0
        for key in list(self._events.keys()):
            dq = self._events[key]
            while dq and dq[0] < window_start:
                dq.popleft()
            if not dq:
                del self._events[key]
                removed += 1
        if max_keys is not None and len(self._events) > max_keys:
            extra = len(self._events) - max_keys
            for k in sorted(self._events.keys())[:extra]:
                del self._events[k]
                removed += 1
        return removed

    def __len__(self) -> int:
        return len(self._events)
