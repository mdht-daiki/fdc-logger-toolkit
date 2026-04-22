from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse

from .api_client import APIError

DEFAULT_DB_API_BASE_URL = os.getenv("PORTFOLIO_DB_API_URL", "http://localhost:8000")
logger = logging.getLogger(__name__)


def _default_allowed_db_api_hosts() -> set[str]:
    hosts = {"localhost", "127.0.0.1", "::1"}
    parsed_default = urlparse(DEFAULT_DB_API_BASE_URL)
    if parsed_default.hostname:
        hosts.add(parsed_default.hostname.lower())
    return hosts


def _allowed_db_api_hosts() -> set[str]:
    env_hosts = os.getenv("PORTFOLIO_DB_API_ALLOWED_HOSTS", "")
    hosts = _default_allowed_db_api_hosts()
    for host in env_hosts.split(","):
        normalized = host.strip().lower()
        if normalized:
            hosts.add(normalized)
    return hosts


def _is_restricted_ip(ip_value: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip_value.is_private
        or ip_value.is_loopback
        or ip_value.is_link_local
        or ip_value.is_multicast
        or ip_value.is_reserved
        or ip_value.is_unspecified
    )


def validate_base_url(base_url: str) -> tuple[str, str]:
    raw_value = (base_url or "").strip()
    if not raw_value:
        logger.warning("Rejected empty db_api base URL")
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    parsed = urlparse(raw_value)
    log_target = f"{parsed.scheme or '-'}://{parsed.hostname or '-'}"

    has_credentials = bool(parsed.username or parsed.password or "@" in parsed.netloc)
    if has_credentials:
        logger.warning("Rejected db_api base URL with credentials: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    if parsed.scheme not in {"http", "https"}:
        logger.warning("Rejected db_api base URL with unsupported scheme: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    if not parsed.hostname:
        logger.warning("Rejected db_api base URL without hostname: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        logger.warning("Rejected db_api base URL with path/query/fragment: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    try:
        parsed_port = parsed.port
    except ValueError:
        logger.warning("Rejected db_api base URL with invalid port: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL") from None

    if parsed_port == 0:
        logger.warning("Rejected db_api base URL with disallowed port: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    hostname = parsed.hostname.lower()
    # すべてのケースでIP埋込URL（接続用）と元ホスト名（Host/SNI用）をatomicに返す
    resolve_host = hostname
    try:
        resolved = socket.getaddrinfo(resolve_host, parsed_port or 80, type=socket.SOCK_STREAM)
    except OSError:
        logger.warning("Rejected db_api base URL; hostname resolution failed: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL") from None

    resolved_ip: str | None = None
    allowed_hosts = _allowed_db_api_hosts()
    for result in resolved:
        sockaddr = result[4]
        if not sockaddr:
            continue
        candidate_ip = sockaddr[0]
        try:
            ip_value = ipaddress.ip_address(candidate_ip)
        except ValueError:
            logger.warning("Rejected db_api base URL; invalid resolved IP: %s", log_target)
            raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL") from None
        # allowed_hostsに含まれる場合は_is_restricted_ipチェックをスキップ
        if hostname not in allowed_hosts and _is_restricted_ip(ip_value):
            logger.warning("Rejected db_api base URL; restricted network target: %s", log_target)
            raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")
        if isinstance(candidate_ip, str):
            resolved_ip = candidate_ip
            break

    if not isinstance(resolved_ip, str):
        logger.warning("Rejected db_api base URL; could not resolve valid IP: %s", log_target)
        raise APIError(message="Invalid db_api base URL", code="INVALID_BASE_URL")

    # localhostはtrim/normalize後のraw_valueを返す
    if hostname == "localhost":
        return raw_value, hostname

    bracketed_ip = f"[{resolved_ip}]" if ":" in resolved_ip else resolved_ip
    ip_url = f"{parsed.scheme}://{bracketed_ip}"
    if parsed_port is not None:
        ip_url = f"{ip_url}:{parsed_port}"

    return ip_url, hostname
