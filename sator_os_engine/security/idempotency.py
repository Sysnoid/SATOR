from __future__ import annotations

import time


class IdempotencyStore:
    def __init__(self, ttl_sec: int = 600) -> None:
        self.ttl_sec = ttl_sec
        self._store: dict[tuple[str, str], tuple[float, str]] = {}

    def put(self, api_key: str, idem_key: str, job_id: str) -> None:
        self._store[(api_key, idem_key)] = (time.time(), job_id)

    def get(self, api_key: str, idem_key: str) -> str | None:
        key = (api_key, idem_key)
        item = self._store.get(key)
        if not item:
            return None
        ts, job_id = item
        if time.time() - ts > self.ttl_sec:
            del self._store[key]
            return None
        return job_id

    def sweep_expired(self) -> int:
        """Remove all entries past ``ttl_sec`` (even if not read). Returns count removed."""
        now = time.time()
        dead = [k for k, (ts, _) in self._store.items() if (now - ts) > self.ttl_sec]
        for k in dead:
            del self._store[k]
        return len(dead)

    def __len__(self) -> int:
        return len(self._store)
