"""Tests for effective client IP when X-Forwarded-For is present."""

from __future__ import annotations

from unittest.mock import MagicMock

from sator_os_engine.security.client_ip import (
    effective_client_ip,
    parse_trusted_proxy_cidrs_csv,
)


def test_xff_ignored_when_peer_not_trusted():
    req = MagicMock()
    req.client.host = "203.0.113.5"
    req.headers = {"x-forwarded-for": "198.51.100.2, 10.0.0.1"}
    assert effective_client_ip(req, ["127.0.0.1"]) == "203.0.113.5"


def test_xff_used_when_peer_trusted():
    req = MagicMock()
    req.client.host = "127.0.0.1"
    req.headers = {"x-forwarded-for": "198.51.100.2, 10.0.0.1"}
    assert effective_client_ip(req, ["127.0.0.1"]) == "198.51.100.2"


def test_xff_used_for_cidr_trusted():
    req = MagicMock()
    req.client.host = "10.0.0.5"
    req.headers = {"x-forwarded-for": "198.51.100.9"}
    assert effective_client_ip(req, ["10.0.0.0/8"]) == "198.51.100.9"


def test_parse_csv():
    assert parse_trusted_proxy_cidrs_csv("") == []
    assert parse_trusted_proxy_cidrs_csv(" 127.0.0.1 , 10.0.0.0/8 ") == [
        "127.0.0.1",
        "10.0.0.0/8",
    ]
