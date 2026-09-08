"""NPPES NPI Registry enrichment client (WORKFLOWS.md §1).

Queries the public NPPES API (https://npiregistry.cms.hhs.gov/api/, version
2.1) with an on-disk SQLite cache (TTL 90 days), polite exponential-backoff
retries, offline fallback to the last cached snapshot, and a full stub mode
so tests never touch the network.

Privacy: NPIs are hashed (sha256) by callers immediately after enrichment;
this module only ever *sends* the NPI to the public registry it came from.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import requests

NPPES_BASE_URL = "https://npiregistry.cms.hhs.gov/api/"
NPPES_VERSION = "2.1"
CACHE_TTL_DAYS = 90


def hash_id(value: str) -> str:
    """sha256 hex digest - the only identifier form allowed in the store."""
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()


def _parse_result(payload: dict) -> dict | None:
    """Extract {taxonomy, state, enumeration_date} from an NPPES 2.1 response."""
    results = payload.get("results") or []
    if not results:
        return None
    rec = results[0]
    taxonomy = None
    for tax in rec.get("taxonomies") or []:
        if tax.get("primary"):
            taxonomy = tax.get("code") or tax.get("desc")
            break
        if taxonomy is None:
            taxonomy = tax.get("code") or tax.get("desc")
    state = None
    for addr in rec.get("addresses") or []:
        if addr.get("address_purpose") == "LOCATION":
            state = addr.get("state")
            break
        if state is None:
            state = addr.get("state")
    enumeration_date = (rec.get("basic") or {}).get("enumeration_date")
    return {"taxonomy": taxonomy, "state": state,
            "enumeration_date": enumeration_date, "stale": False}


class NPPESClient:
    """NPPES API client with SQLite cache, retries, offline and stub modes.

    Stub mode (``stub={npi: {...}}``) serves responses from the given dict
    and never opens a network connection - used by tests and offline demos.
    """

    def __init__(self, cache_path: str | Path | None = None,
                 ttl_days: int = CACHE_TTL_DAYS,
                 stub: dict[str, dict] | None = None,
                 session: requests.Session | None = None,
                 max_retries: int = 3, backoff: float = 0.5,
                 timeout: float = 10.0):
        self.ttl_seconds = ttl_days * 86400
        self.stub = stub
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self._conn: sqlite3.Connection | None = None
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(cache_path))
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS npi_cache ("
                " npi TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at REAL NOT NULL)")
            self._conn.commit()

    # -- cache ---------------------------------------------------------------
    def _cache_get(self, npi: str, allow_stale: bool = False) -> dict | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT payload, fetched_at FROM npi_cache WHERE npi = ?", (npi,)
        ).fetchone()
        if row is None:
            return None
        payload, fetched_at = json.loads(row[0]), row[1]
        fresh = (time.time() - fetched_at) <= self.ttl_seconds
        if not fresh and not allow_stale:
            return None
        out = dict(payload) if payload else None
        if out is not None and not fresh:
            out["stale"] = True
        return out

    def _cache_put(self, npi: str, payload: dict | None) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO npi_cache (npi, payload, fetched_at) VALUES (?, ?, ?)",
            (npi, json.dumps(payload), time.time()))
        self._conn.commit()

    # -- network ---------------------------------------------------------------
    def _fetch(self, npi: str) -> dict | None:
        """GET the NPPES API with polite exponential-backoff retries."""
        params = {"version": NPPES_VERSION, "number": npi}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(NPPES_BASE_URL, params=params,
                                        timeout=self.timeout)
                if resp.status_code == 200:
                    return _parse_result(resp.json())
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self.backoff * (2**attempt))
                    continue
                resp.raise_for_status()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                time.sleep(self.backoff * (2**attempt))
        if last_exc is not None:
            raise ConnectionError(f"NPPES lookup failed for NPI: {last_exc}")
        raise ConnectionError("NPPES lookup failed: rate limited after retries")

    # -- public API --------------------------------------------------------------
    def lookup(self, npi: str) -> dict | None:
        """Look up an NPI -> {taxonomy, state, enumeration_date, stale}.

        Order: stub -> fresh cache -> network (cached) -> stale cache fallback.
        Returns None when the NPI is unknown to the registry.
        """
        npi = str(npi).strip()
        if self.stub is not None:
            hit = self.stub.get(npi)
            return dict(hit) if hit is not None else None

        cached = self._cache_get(npi)
        if cached is not None:
            return cached

        try:
            result = self._fetch(npi)
        except ConnectionError:
            stale = self._cache_get(npi, allow_stale=True)
            if stale is not None:
                stale["stale"] = True
                return stale
            raise
        self._cache_put(npi, result)
        return result

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> NPPESClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
