"""Tests for security hardening — auth, rate limiting, CORS, debug gating, headers.

No live services needed. Mocks auth credentials and tests security
mechanisms in isolation.
"""

import base64
import time

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_auth_header(user: str, password: str) -> dict:
    encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_auth_failures():
    """Reset auth failure tracking between tests."""
    import server

    server._auth_failures.clear()
    yield
    server._auth_failures.clear()


@pytest.fixture()
def _auth_creds(monkeypatch):
    """Set auth credentials so check_basic_auth can succeed."""
    import server

    monkeypatch.setattr(server, "CHAT_USER", "testuser")
    monkeypatch.setattr(server, "CHAT_PASS", "testpass")


@pytest.fixture()
def client(_auth_creds):
    """TestClient with auth credentials configured."""
    import server

    return TestClient(server.app, raise_server_exceptions=False)


# ===========================================================================
# _get_client_ip
# ===========================================================================


class TestGetClientIP:
    """CF-Connecting-IP is the canonical source behind Cloudflare."""

    def test_prefers_cf_connecting_ip(self):
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {"CF-Connecting-IP": "1.2.3.4"}
        request.client.host = "10.0.0.1"
        assert server._get_client_ip(request) == "1.2.3.4"

    def test_strips_whitespace(self):
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {"CF-Connecting-IP": "  1.2.3.4  "}
        request.client.host = "10.0.0.1"
        assert server._get_client_ip(request) == "1.2.3.4"

    def test_ignores_empty_cf_header(self):
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {"CF-Connecting-IP": ""}
        request.client.host = "10.0.0.1"
        assert server._get_client_ip(request) == "10.0.0.1"

    def test_ignores_whitespace_only_cf_header(self):
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {"CF-Connecting-IP": "   "}
        request.client.host = "10.0.0.1"
        assert server._get_client_ip(request) == "10.0.0.1"

    def test_does_not_use_xff(self):
        """X-Forwarded-For is client-spoofable — must not be used."""
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {"X-Forwarded-For": "spoofed.ip.here"}
        request.client.host = "10.0.0.1"
        assert server._get_client_ip(request) == "10.0.0.1"

    def test_no_client_returns_unknown(self):
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {}
        request.client = None
        assert server._get_client_ip(request) == "unknown"


# ===========================================================================
# check_basic_auth
# ===========================================================================


class TestCheckBasicAuth:
    def test_rejects_when_creds_not_configured(self, monkeypatch):
        """Auth must FAIL when env vars are missing — not silently pass."""
        from unittest.mock import MagicMock

        import server

        monkeypatch.setattr(server, "CHAT_USER", "")
        monkeypatch.setattr(server, "CHAT_PASS", "")
        request = MagicMock()
        request.headers = _basic_auth_header("anything", "anything")
        assert server.check_basic_auth(request) is False

    def test_accepts_valid_credentials(self, _auth_creds):
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = _basic_auth_header("testuser", "testpass")
        assert server.check_basic_auth(request) is True

    def test_rejects_wrong_password(self, _auth_creds):
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = _basic_auth_header("testuser", "wrong")
        assert server.check_basic_auth(request) is False

    def test_rejects_missing_header(self, _auth_creds):
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {}
        assert server.check_basic_auth(request) is False

    def test_rejects_malformed_base64(self, _auth_creds):
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {"Authorization": "Basic !!!notbase64!!!"}
        assert server.check_basic_auth(request) is False


# ===========================================================================
# Auth brute-force rate limiting
# ===========================================================================


class TestAuthBruteForce:
    """Tests use the actual _AUTH_FAILURE_LIMIT (15) from server config."""

    def test_failures_below_limit_return_401(self, client):
        """Failures below the limit all get 401 (auth rejected, not rate limited)."""
        import server

        for i in range(server._AUTH_FAILURE_LIMIT - 1):
            resp = client.get("/", headers=_basic_auth_header("wrong", "wrong"))
            assert resp.status_code == 401, f"Request {i + 1} should be 401"

    def test_limit_reached_returns_429(self, client):
        """After reaching the failure limit, the next request gets 429."""
        import server

        for _ in range(server._AUTH_FAILURE_LIMIT):
            client.get("/", headers=_basic_auth_header("wrong", "wrong"))
        resp = client.get("/", headers=_basic_auth_header("wrong", "wrong"))
        assert resp.status_code == 429

    def test_valid_auth_works_below_threshold(self, client):
        """Correct credentials still work after a few failures."""
        for _ in range(3):
            client.get("/", headers=_basic_auth_header("wrong", "wrong"))
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_failures_expire_after_window(self, client, monkeypatch):
        """Old failures are pruned — brute-force window is not permanent."""
        import server

        monkeypatch.setattr(server, "_AUTH_FAILURE_WINDOW", 0.01)
        for _ in range(server._AUTH_FAILURE_LIMIT):
            client.get("/", headers=_basic_auth_header("wrong", "wrong"))
        time.sleep(0.02)
        resp = client.get("/", headers=_basic_auth_header("wrong", "wrong"))
        assert resp.status_code == 401

    def test_successful_auth_resets_counter(self, client):
        """Correct login clears failure history — prevents self-lockout."""
        import server

        for _ in range(server._AUTH_FAILURE_LIMIT - 2):
            client.get("/", headers=_basic_auth_header("wrong", "wrong"))
        client.get("/", headers=_basic_auth_header("testuser", "testpass"))
        # Same number of failures should stay under threshold (counter was reset)
        for _ in range(server._AUTH_FAILURE_LIMIT - 2):
            client.get("/", headers=_basic_auth_header("wrong", "wrong"))
        resp = client.get("/", headers=_basic_auth_header("wrong", "wrong"))
        assert resp.status_code == 401


# ===========================================================================
# Debug endpoint gating
# ===========================================================================


class TestDebugEndpoints:
    def test_vitals_404_when_debug_disabled(self, client, monkeypatch):
        import server

        monkeypatch.setattr(server, "DEBUG_ENDPOINTS", False)
        resp = client.get("/vitals", headers=_basic_auth_header("testuser", "testpass"))
        assert resp.status_code == 404

    def test_debug_orient_404_when_debug_disabled(self, client, monkeypatch):
        import server

        monkeypatch.setattr(server, "DEBUG_ENDPOINTS", False)
        resp = client.get("/debug/orient", headers=_basic_auth_header("testuser", "testpass"))
        assert resp.status_code == 404


# ===========================================================================
# Security headers
# ===========================================================================


class TestSecurityHeaders:
    def test_nosniff_header(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_frame_deny_header(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_hsts_header(self, client):
        resp = client.get("/health")
        hsts = resp.headers.get("Strict-Transport-Security", "")
        assert "max-age=31536000" in hsts
        assert "includeSubDomains" in hsts

    def test_headers_on_401_responses(self, client):
        """Security headers must appear on auth failures, not just 200s."""
        resp = client.get("/", headers=_basic_auth_header("wrong", "wrong"))
        assert resp.status_code == 401
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"


# ===========================================================================
# CORS
# ===========================================================================


class TestCORS:
    def test_allows_chat_auran_llc_origin(self, client):
        resp = client.options(
            "/chat",
            headers={
                "Origin": "https://chat.auran.llc",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "https://chat.auran.llc"

    def test_blocks_unknown_origin(self, client):
        resp = client.options(
            "/chat",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "https://evil.example.com"


# ===========================================================================
# Rate limiter wiring
# ===========================================================================


class TestRateLimiterConfig:
    def test_limiter_uses_get_client_ip(self):
        import server

        assert server.limiter._key_func is server._get_client_ip
