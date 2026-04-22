from __future__ import annotations

import ipaddress

from starlette.requests import Request


def parse_trusted_proxy_cidrs_csv(raw: str) -> list[str]:
    s = (raw or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _peer_in_trusted_networks(peer: str, networks: list[str]) -> bool:
    if not peer or not networks:
        return False
    try:
        ip = ipaddress.ip_address(peer)
    except ValueError:
        return False
    for raw in networks:
        entry = (raw or "").strip()
        if not entry:
            continue
        if "/" in entry:
            try:
                if ip in ipaddress.ip_network(entry, strict=False):
                    return True
            except ValueError:
                continue
        else:
            try:
                if ip == ipaddress.ip_address(entry):
                    return True
            except ValueError:
                continue
    return False


def effective_client_ip(request: Request, trusted_proxy_cidrs: list[str]) -> str:
    """Client IP for policy (rate limit, allow/deny list).

    * ``X-Forwarded-For`` is used **only** when the **direct** TCP peer is listed in
    *trusted_proxy_cidrs* (single IPs or CIDRs). If the list is empty, XFF is **ignored**,
    which avoids trivial IP spoofing when the app is not behind a known proxy.
    """
    direct = request.client.host if request.client else ""
    if not direct:
        return ""
    if not _peer_in_trusted_networks(direct, trusted_proxy_cidrs):
        return direct
    xff = request.headers.get("x-forwarded-for")
    if not xff:
        return direct
    return xff.split(",")[0].strip() or direct
