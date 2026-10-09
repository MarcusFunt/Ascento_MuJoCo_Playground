from dashboard.control_auth import (
    CONTROL_SESSION_TTL_SECONDS,
    issue_session,
    session_remaining_seconds,
    token_matches,
    valid_session,
)


def test_signed_control_session_expires_and_rejects_tampering():
    token = "a-test-secret-long-enough-to-be-valid"
    session = issue_session(token, now=1000)

    assert valid_session(token, session, now=1001) is True
    assert session_remaining_seconds(token, session, now=1001) == CONTROL_SESSION_TTL_SECONDS - 1
    assert valid_session(token, session, now=1000 + CONTROL_SESSION_TTL_SECONDS) is False
    assert valid_session("different-secret-long-enough-to-be-valid", session, now=1001) is False
    assert valid_session(token, session + "x", now=1001) is False
    assert token_matches(token, token) is True
    assert token_matches(token, "wrong") is False
