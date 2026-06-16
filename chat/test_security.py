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
    """IP extraction behind ALB, with optional Cloudflare trust."""

    def test_prefers_cf_connecting_ip_when_trusted(self):
        from unittest.mock import MagicMock, patch

        import server

        request = MagicMock()
        request.headers = {"CF-Connecting-IP": "1.2.3.4"}
        request.client.host = "10.0.0.1"
        with patch.object(server, "TRUST_CF_HEADER", True):
            assert server._get_client_ip(request) == "1.2.3.4"

    def test_ignores_cf_header_when_untrusted(self):
        """Without TRUST_CF_HEADER, CF-Connecting-IP is client-spoofable."""
        from unittest.mock import MagicMock, patch

        import server

        request = MagicMock()
        request.headers = {"CF-Connecting-IP": "spoofed.ip", "X-Forwarded-For": "real.client.ip"}
        request.client.host = "10.0.0.83"
        with patch.object(server, "TRUST_CF_HEADER", False):
            assert server._get_client_ip(request) == "real.client.ip"

    def test_strips_whitespace(self):
        from unittest.mock import MagicMock, patch

        import server

        request = MagicMock()
        request.headers = {"CF-Connecting-IP": "  1.2.3.4  "}
        request.client.host = "10.0.0.1"
        with patch.object(server, "TRUST_CF_HEADER", True):
            assert server._get_client_ip(request) == "1.2.3.4"

    def test_ignores_empty_cf_header(self):
        from unittest.mock import MagicMock, patch

        import server

        request = MagicMock()
        request.headers = {"CF-Connecting-IP": ""}
        request.client.host = "10.0.0.1"
        with patch.object(server, "TRUST_CF_HEADER", True):
            assert server._get_client_ip(request) == "10.0.0.1"

    def test_ignores_whitespace_only_cf_header(self):
        from unittest.mock import MagicMock, patch

        import server

        request = MagicMock()
        request.headers = {"CF-Connecting-IP": "   "}
        request.client.host = "10.0.0.1"
        with patch.object(server, "TRUST_CF_HEADER", True):
            assert server._get_client_ip(request) == "10.0.0.1"

    def test_uses_rightmost_xff_behind_alb(self):
        """Behind ALB (private client.host), use rightmost XFF entry."""
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {"X-Forwarded-For": "spoofed.by.client, 203.0.113.42"}
        request.client.host = "10.0.0.83"
        assert server._get_client_ip(request) == "203.0.113.42"

    def test_ignores_xff_on_direct_connection(self):
        """Direct connection (public client.host) — XFF is untrusted."""
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {"X-Forwarded-For": "spoofed.ip.here"}
        request.client.host = "8.8.8.8"
        assert server._get_client_ip(request) == "8.8.8.8"

    def test_cf_ip_takes_priority_over_xff(self):
        """CF-Connecting-IP wins over XFF when trusted."""
        from unittest.mock import MagicMock, patch

        import server

        request = MagicMock()
        request.headers = {
            "CF-Connecting-IP": "198.51.100.5",
            "X-Forwarded-For": "10.0.0.1, 203.0.113.42",
        }
        request.client.host = "10.0.0.83"
        with patch.object(server, "TRUST_CF_HEADER", True):
            assert server._get_client_ip(request) == "198.51.100.5"

    def test_ignores_xff_for_public_172_range(self):
        """172.217.x.x (Google) is public despite starting with 172."""
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {"X-Forwarded-For": "attacker.ip.here"}
        request.client.host = "172.217.14.206"
        assert server._get_client_ip(request) == "172.217.14.206"

    def test_private_host_no_xff_returns_private_ip(self):
        """Behind ALB but no XFF header — fall through to client.host."""
        from unittest.mock import MagicMock

        import server

        request = MagicMock()
        request.headers = {}
        request.client.host = "10.0.0.83"
        assert server._get_client_ip(request) == "10.0.0.83"

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


# ===========================================================================
# Session cookie auth
# ===========================================================================


class TestSessionCookie:
    @pytest.fixture()
    def _session_env(self, monkeypatch):
        import server

        monkeypatch.setattr(server, "SESSION_SECRET_KEY", "test-secret-key-32bytes-long!!")
        monkeypatch.setattr(server, "CHAT_USER", "")
        monkeypatch.setattr(server, "CHAT_PASS", "")
        monkeypatch.setattr(server, "CF_TEAM_DOMAIN", "")
        monkeypatch.setattr(server, "CF_ACCESS_AUD", "")
        server._session_serializer = None

    @pytest.fixture()
    def session_client(self, _session_env):
        import server

        return TestClient(server.app, raise_server_exceptions=False)

    def _mint_cookie(self, identity: str) -> str:
        from itsdangerous import URLSafeTimedSerializer

        s = URLSafeTimedSerializer("test-secret-key-32bytes-long!!")
        return s.dumps({"sub": identity})

    def test_valid_cookie_authenticates(self, session_client):
        cookie = self._mint_cookie("olivia@auran.llc")
        resp = session_client.get("/health", cookies={"auran_session": cookie})
        assert resp.status_code == 200

    def test_valid_cookie_allows_protected_route(self, session_client):
        cookie = self._mint_cookie("olivia@auran.llc")
        resp = session_client.get("/chat/status", cookies={"auran_session": cookie})
        assert resp.status_code == 200

    def test_invalid_cookie_returns_401(self, session_client):
        resp = session_client.get("/chat/status", cookies={"auran_session": "garbage-value"})
        assert resp.status_code == 401

    def test_missing_cookie_returns_401(self, session_client):
        resp = session_client.get("/chat/status")
        assert resp.status_code == 401

    def test_expired_cookie_returns_401(self, session_client, monkeypatch):
        import server

        monkeypatch.setattr(server, "SESSION_MAX_AGE", 1)
        cookie = self._mint_cookie("olivia@auran.llc")
        time.sleep(2.1)
        resp = session_client.get("/chat/status", cookies={"auran_session": cookie})
        assert resp.status_code == 401

    def test_cookie_signed_with_wrong_key_returns_401(self, session_client):
        from itsdangerous import URLSafeTimedSerializer

        wrong_key_serializer = URLSafeTimedSerializer("wrong-key-entirely")
        bad_cookie = wrong_key_serializer.dumps({"sub": "attacker@evil.com"})
        resp = session_client.get("/chat/status", cookies={"auran_session": bad_cookie})
        assert resp.status_code == 401

    def test_empty_sub_in_cookie_returns_401(self, session_client):
        from itsdangerous import URLSafeTimedSerializer

        s = URLSafeTimedSerializer("test-secret-key-32bytes-long!!")
        cookie = s.dumps({"sub": ""})
        resp = session_client.get("/chat/status", cookies={"auran_session": cookie})
        assert resp.status_code == 401

    def test_cookie_refreshed_on_success(self, session_client):
        cookie = self._mint_cookie("olivia@auran.llc")
        resp = session_client.get("/chat/status", cookies={"auran_session": cookie})
        assert resp.status_code == 200
        set_cookie = resp.headers.get("set-cookie", "")
        assert "auran_session=" in set_cookie
        assert "httponly" in set_cookie.lower()
        assert "secure" in set_cookie.lower()


# ===========================================================================
# CF Access JWT auth
# ===========================================================================


class TestCFAccessJWT:
    @pytest.fixture()
    def _cf_env(self, monkeypatch):
        import server

        monkeypatch.setattr(server, "CF_TEAM_DOMAIN", "test-team")
        monkeypatch.setattr(server, "CF_ACCESS_AUD", "test-aud-tag")
        monkeypatch.setattr(server, "SESSION_SECRET_KEY", "test-secret-key-32bytes-long!!")
        monkeypatch.setattr(server, "CHAT_USER", "")
        monkeypatch.setattr(server, "CHAT_PASS", "")
        server._session_serializer = None

    @pytest.fixture()
    def cf_client(self, _cf_env):
        import server

        return TestClient(server.app, raise_server_exceptions=False)

    def test_valid_jwt_authenticates(self, cf_client, monkeypatch):
        import server

        async def mock_validate(token):
            if token == "valid-cf-token":
                return "olivia@auran.llc"
            return None

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)
        resp = cf_client.get("/chat/status", cookies={"CF_Authorization": "valid-cf-token"})
        assert resp.status_code == 200

    def test_valid_jwt_mints_session_cookie(self, cf_client, monkeypatch):
        import server

        async def mock_validate(token):
            return "olivia@auran.llc"

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)
        resp = cf_client.get("/chat/status", cookies={"CF_Authorization": "valid-cf-token"})
        assert resp.status_code == 200
        set_cookie = resp.headers.get("set-cookie", "")
        assert "auran_session=" in set_cookie

    def test_invalid_jwt_returns_401(self, cf_client, monkeypatch):
        import server

        async def mock_validate(token):
            return None

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)
        resp = cf_client.get("/chat/status", cookies={"CF_Authorization": "bad-token"})
        assert resp.status_code == 401

    def test_missing_cf_cookie_returns_401(self, cf_client, monkeypatch):
        import server

        async def mock_validate(token):
            return "olivia@auran.llc"

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)
        resp = cf_client.get("/chat/status")
        assert resp.status_code == 401


# ===========================================================================
# Auth chain fallback order
# ===========================================================================


class TestAuthChainFallback:
    @pytest.fixture()
    def _all_methods(self, monkeypatch):
        import server

        monkeypatch.setattr(server, "SESSION_SECRET_KEY", "test-secret-key-32bytes-long!!")
        monkeypatch.setattr(server, "CF_TEAM_DOMAIN", "test-team")
        monkeypatch.setattr(server, "CF_ACCESS_AUD", "test-aud-tag")
        monkeypatch.setattr(server, "CHAT_USER", "testuser")
        monkeypatch.setattr(server, "CHAT_PASS", "testpass")
        server._session_serializer = None

    @pytest.fixture()
    def chain_client(self, _all_methods):
        import server

        return TestClient(server.app, raise_server_exceptions=False)

    def _mint_cookie(self, identity: str) -> str:
        from itsdangerous import URLSafeTimedSerializer

        s = URLSafeTimedSerializer("test-secret-key-32bytes-long!!")
        return s.dumps({"sub": identity})

    def test_cookie_takes_priority_over_jwt_and_basic(self, chain_client, monkeypatch):
        """Cookie is checked first — valid cookie means JWT never checked."""
        import server

        jwt_called = []

        async def mock_validate(token):
            jwt_called.append(True)
            return "other@auran.llc"

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)
        cookie = self._mint_cookie("olivia@auran.llc")
        resp = chain_client.get(
            "/chat/status",
            cookies={"auran_session": cookie, "CF_Authorization": "some-token"},
            headers=_basic_auth_header("testuser", "testpass"),
        )
        assert resp.status_code == 200
        assert jwt_called == []

    def test_jwt_fallback_when_no_cookie(self, chain_client, monkeypatch):
        import server

        async def mock_validate(token):
            return "olivia@auran.llc"

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)
        resp = chain_client.get("/chat/status", cookies={"CF_Authorization": "valid-token"})
        assert resp.status_code == 200

    def test_basic_auth_fallback_when_no_cookie_no_jwt(self, chain_client, monkeypatch):
        import server

        async def mock_validate(token):
            return None

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)
        resp = chain_client.get("/chat/status", headers=_basic_auth_header("testuser", "testpass"))
        assert resp.status_code == 200

    def test_basic_auth_mints_cookie_during_transition(self, chain_client, monkeypatch):
        import server

        async def mock_validate(token):
            return None

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)
        resp = chain_client.get("/chat/status", headers=_basic_auth_header("testuser", "testpass"))
        assert resp.status_code == 200
        set_cookie = resp.headers.get("set-cookie", "")
        assert "auran_session=" in set_cookie

    def test_all_methods_fail_returns_401(self, chain_client, monkeypatch):
        import server

        async def mock_validate(token):
            return None

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)
        resp = chain_client.get(
            "/chat/status",
            cookies={"CF_Authorization": "bad-token"},
            headers=_basic_auth_header("wrong", "wrong"),
        )
        assert resp.status_code == 401


# ===========================================================================
# Unified rate limiting (protects all auth methods)
# ===========================================================================


class TestUnifiedRateLimiting:
    @pytest.fixture()
    def _cf_only(self, monkeypatch):
        import server

        monkeypatch.setattr(server, "CF_TEAM_DOMAIN", "test-team")
        monkeypatch.setattr(server, "CF_ACCESS_AUD", "test-aud-tag")
        monkeypatch.setattr(server, "SESSION_SECRET_KEY", "test-secret-key-32bytes-long!!")
        monkeypatch.setattr(server, "CHAT_USER", "")
        monkeypatch.setattr(server, "CHAT_PASS", "")
        server._session_serializer = None

    @pytest.fixture()
    def cf_rate_client(self, _cf_only):
        import server

        return TestClient(server.app, raise_server_exceptions=False)

    def test_jwt_failures_trigger_lockout(self, cf_rate_client, monkeypatch):
        """Rate limiting fires even without Basic Auth — protects JWT path."""
        import server

        async def mock_validate(token):
            return None

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)

        for _ in range(server._AUTH_FAILURE_LIMIT):
            cf_rate_client.get("/chat/status", cookies={"CF_Authorization": "bad-token"})

        resp = cf_rate_client.get("/chat/status", cookies={"CF_Authorization": "bad-token"})
        assert resp.status_code == 429

    def test_valid_jwt_resets_failures(self, cf_rate_client, monkeypatch):
        """Successful JWT auth clears failure counters."""
        import server

        call_count = [0]

        async def mock_validate(token):
            call_count[0] += 1
            if call_count[0] <= 5:
                return None
            return "olivia@auran.llc"

        monkeypatch.setattr(server, "_validate_cf_jwt", mock_validate)

        for _ in range(5):
            cf_rate_client.get("/chat/status", cookies={"CF_Authorization": "bad"})

        assert len(server._auth_failures.get("testclient", [])) > 0

        cf_rate_client.get("/chat/status", cookies={"CF_Authorization": "now-valid"})
        assert server._auth_failures.get("testclient") is None

    def test_no_auth_configured_returns_401_not_500(self, monkeypatch):
        """With no auth methods configured, returns 401 (fail-closed)."""
        import server

        monkeypatch.setattr(server, "CF_TEAM_DOMAIN", "")
        monkeypatch.setattr(server, "CF_ACCESS_AUD", "")
        monkeypatch.setattr(server, "SESSION_SECRET_KEY", "")
        monkeypatch.setattr(server, "CHAT_USER", "")
        monkeypatch.setattr(server, "CHAT_PASS", "")

        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.get("/chat/status")
        assert resp.status_code == 401


# ===========================================================================
# JWT validation unit tests
# ===========================================================================


class TestValidateCFJWT:
    def test_rejects_token_with_wrong_issuer(self):
        import jwt as pyjwt
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa

        import server

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        token = pyjwt.encode(
            {"email": "olivia@auran.llc", "aud": "test-aud", "iss": "https://evil.com"},
            private_key,
            algorithm="RS256",
            headers={"kid": "test-kid"},
        )

        jwk = pyjwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
        jwk["kid"] = "test-kid"

        server.CF_TEAM_DOMAIN = "test-team"
        server.CF_ACCESS_AUD = "test-aud"
        server._jwks_cache = {"keys": [jwk], "fetched_at": time.monotonic()}

        import asyncio

        result = asyncio.run(server._validate_cf_jwt(token))
        assert result is None

    def test_accepts_token_with_correct_claims(self):
        import jwt as pyjwt
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa

        import server

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        token = pyjwt.encode(
            {
                "email": "olivia@auran.llc",
                "aud": "test-aud",
                "iss": "https://test-team.cloudflareaccess.com",
                "exp": time.time() + 300,
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "test-kid"},
        )

        jwk = pyjwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
        jwk["kid"] = "test-kid"

        server.CF_TEAM_DOMAIN = "test-team"
        server.CF_ACCESS_AUD = "test-aud"
        server._jwks_cache = {"keys": [jwk], "fetched_at": time.monotonic()}

        import asyncio

        result = asyncio.run(server._validate_cf_jwt(token))
        assert result == "olivia@auran.llc"

    def test_kid_mismatch_skips_key(self):
        import jwt as pyjwt
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa

        import server

        real_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        token = pyjwt.encode(
            {
                "email": "olivia@auran.llc",
                "aud": "test-aud",
                "iss": "https://test-team.cloudflareaccess.com",
                "exp": time.time() + 300,
            },
            real_key,
            algorithm="RS256",
            headers={"kid": "real-kid"},
        )

        wrong_jwk = pyjwt.algorithms.RSAAlgorithm.to_jwk(wrong_key.public_key(), as_dict=True)
        wrong_jwk["kid"] = "wrong-kid"
        real_jwk = pyjwt.algorithms.RSAAlgorithm.to_jwk(real_key.public_key(), as_dict=True)
        real_jwk["kid"] = "real-kid"

        server.CF_TEAM_DOMAIN = "test-team"
        server.CF_ACCESS_AUD = "test-aud"
        server._jwks_cache = {"keys": [wrong_jwk, real_jwk], "fetched_at": time.monotonic()}

        import asyncio

        result = asyncio.run(server._validate_cf_jwt(token))
        assert result == "olivia@auran.llc"
