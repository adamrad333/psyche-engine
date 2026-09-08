"""NPPES client tests - stub mode only, no network (WORKFLOWS §1)."""

import time

import pytest

from psyche.enrich import NPPESClient, _parse_result, hash_id

STUB = {
    "1234567890": {"taxonomy": "207Q00000X", "state": "CA",
                   "enumeration_date": "2010-05-01", "stale": False},
}


def test_stub_mode_never_hits_network():
    client = NPPESClient(stub=STUB)
    out = client.lookup("1234567890")
    assert out["taxonomy"] == "207Q00000X"
    assert out["state"] == "CA"
    assert client.lookup("0000000000") is None


def test_hash_id_deterministic_and_normalized():
    assert hash_id("  ABC@Example.COM ") == hash_id("abc@example.com")
    assert len(hash_id("x")) == 64


def test_parse_nppes_21_response():
    payload = {
        "result_count": 1,
        "results": [{
            "basic": {"enumeration_date": "2012-03-14"},
            "taxonomies": [
                {"code": "207R00000X", "primary": False},
                {"code": "207Q00000X", "primary": True},
            ],
            "addresses": [
                {"address_purpose": "MAILING", "state": "TX"},
                {"address_purpose": "LOCATION", "state": "CA"},
            ],
        }],
    }
    out = _parse_result(payload)
    assert out == {"taxonomy": "207Q00000X", "state": "CA",
                   "enumeration_date": "2012-03-14", "stale": False}
    assert _parse_result({"results": []}) is None


def test_cache_roundtrip_and_freshness(tmp_path):
    cache = tmp_path / "npi.sqlite"
    client = NPPESClient(cache_path=cache, ttl_days=90)
    client._cache_put("1234567890", STUB["1234567890"])
    assert client._cache_get("1234567890")["state"] == "CA"
    client.close()

    # expired entries are only visible with allow_stale (offline fallback)
    client = NPPESClient(cache_path=cache, ttl_days=0)
    assert client._cache_get("1234567890") is None
    stale = client._cache_get("1234567890", allow_stale=True)
    assert stale is not None and stale["stale"] is True
    client.close()


def test_offline_fallback_marks_stale(tmp_path, monkeypatch):
    cache = tmp_path / "npi.sqlite"
    client = NPPESClient(cache_path=cache, ttl_days=1)
    payload = {"taxonomy": "207Q00000X", "state": "NY",
               "enumeration_date": "2011-01-01", "stale": False}
    client._cache_put("9999999999", payload)
    # age the entry beyond TTL
    conn = client._conn
    conn.execute("UPDATE npi_cache SET fetched_at = ?", (time.time() - 3 * 86400,))
    conn.commit()

    def boom(npi):
        raise ConnectionError("API down")
    monkeypatch.setattr(client, "_fetch", boom)
    out = client.lookup("9999999999")
    assert out["state"] == "NY" and out["stale"] is True

    # unknown NPI with the API down propagates the failure
    with pytest.raises(ConnectionError):
        client.lookup("8888888888")
    client.close()
