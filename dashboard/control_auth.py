"""Short-lived dashboard control sessions backed by an operator secret."""

from __future__ import annotations

import hashlib
import hmac
import os
import time

CONTROL_SESSION_TTL_SECONDS = 12 * 60 * 60
CONTROL_COOKIE_NAME = "ascento_control_session"
MIN_CONTROL_TOKEN_LENGTH = 32


def configured_token() -> str | None:
    token = os.environ.get("ASCENTO_CONTROL_TOKEN", "")
    if len(token) < MIN_CONTROL_TOKEN_LENGTH:
        return None
    return token


def issue_session(token: str, *, now: int | None = None) -> str:
    expires = int(time.time()) if now is None else int(now)
    expires += CONTROL_SESSION_TTL_SECONDS
    payload = str(expires)
    signature = hmac.new(token.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def session_remaining_seconds(
    token: str | None, session: str | None, *, now: int | None = None
) -> int | None:
    if not token or not session:
        return None
    try:
        expires_text, signature = session.split(".", 1)
        expires = int(expires_text)
    except (ValueError, TypeError):
        return None
    current = int(time.time()) if now is None else int(now)
    if expires <= current or expires > current + CONTROL_SESSION_TTL_SECONDS:
        return None
    expected = hmac.new(token.encode("utf-8"), expires_text.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    return expires - current


def valid_session(token: str | None, session: str | None, *, now: int | None = None) -> bool:
    return session_remaining_seconds(token, session, now=now) is not None


def token_matches(expected: str | None, supplied: str) -> bool:
    return bool(expected) and hmac.compare_digest(expected, supplied)
