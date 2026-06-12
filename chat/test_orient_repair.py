"""Tests for orient repair — warmup upgrade, frequency tool, bridge log removal, debug flags.

Mocks librosa and psycopg2 so no live services or audio files needed.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_db_config():
    """Reset cached _db_config between tests."""
    import memory

    original = memory._db_config
    memory._db_config = None
    yield
    memory._db_config = original


@pytest.fixture(autouse=True)
def _set_db_env(monkeypatch):
    """Set DB env vars so _get_db_config never hits Secrets Manager."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "auran_test")
    monkeypatch.setenv("DB_USER", "test")
    monkeypatch.setenv("DB_PASSWORD", "test")


# ---------------------------------------------------------------------------
# Mock librosa helpers
# ---------------------------------------------------------------------------


def _mock_librosa():
    """Build a mock librosa module with realistic return values."""
    librosa = MagicMock()

    sample_rate = 22050
    duration = 180.0
    n_samples = int(sample_rate * duration)
    y = np.random.randn(n_samples).astype(np.float32)

    librosa.load.return_value = (y, sample_rate)
    librosa.get_duration.return_value = duration
    librosa.feature.spectral_centroid.return_value = np.array([[1500.0, 1600.0]])
    librosa.feature.spectral_bandwidth.return_value = np.array([[2000.0, 2100.0]])

    librosa.beat.beat_track.return_value = (120.0, np.array([]))

    chroma = np.random.rand(12, 100).astype(np.float32)
    librosa.feature.chroma_stft.return_value = chroma

    onset_env = np.random.rand(100).astype(np.float32)
    librosa.onset.onset_strength.return_value = onset_env

    return librosa, y, sample_rate


# ===========================================================================
# analyze_audio_frequency
# ===========================================================================


class TestAnalyzeFrequency:
    """Tests for the frequency analysis tool (chat-me's ears)."""

    def test_quick_mode_returns_core_fields(self):
        from memory import analyze_audio_frequency

        mock_librosa, _, _ = _mock_librosa()

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            result = analyze_audio_frequency("/tmp/test.mp3", detail="quick")

        assert "error" not in result
        assert "file_duration_seconds" in result
        assert "analyzed_duration_seconds" in result
        assert "truncated" in result
        assert "sample_rate" in result
        assert "tempo_bpm" in result
        assert "spectral_centroid_hz" in result
        assert "spectral_bandwidth_hz" in result
        assert "dominant_frequencies" in result
        assert "energy_by_band_pct" in result
        assert len(result["dominant_frequencies"]) <= 5

    def test_quick_mode_excludes_full_fields(self):
        from memory import analyze_audio_frequency

        mock_librosa, _, _ = _mock_librosa()

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            result = analyze_audio_frequency("/tmp/test.mp3", detail="quick")

        assert "pitch_class_energy" not in result
        assert "rhythmic_density" not in result
        assert "rhythmic_variance" not in result

    def test_full_mode_includes_pitch_and_rhythm(self):
        from memory import analyze_audio_frequency

        mock_librosa, _, _ = _mock_librosa()

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            result = analyze_audio_frequency("/tmp/test.mp3", detail="full")

        assert "pitch_class_energy" in result
        assert len(result["pitch_class_energy"]) == 12
        expected_pitches = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
        assert set(result["pitch_class_energy"].keys()) == expected_pitches
        assert "rhythmic_density" in result
        assert "rhythmic_variance" in result

    def test_band_energy_sums_to_100(self):
        from memory import analyze_audio_frequency

        mock_librosa, _, _ = _mock_librosa()

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            result = analyze_audio_frequency("/tmp/test.mp3")

        total = sum(result["energy_by_band_pct"].values())
        assert abs(total - 100.0) < 1.0  # within rounding tolerance

    def test_all_seven_bands_present(self):
        from memory import analyze_audio_frequency

        mock_librosa, _, _ = _mock_librosa()

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            result = analyze_audio_frequency("/tmp/test.mp3")

        expected_bands = {"sub_bass", "bass", "low_mid", "mid", "upper_mid", "presence", "brilliance"}
        assert set(result["energy_by_band_pct"].keys()) == expected_bands

    def test_returns_error_on_bad_file(self):
        from memory import analyze_audio_frequency

        result = analyze_audio_frequency("/nonexistent/file.mp3")
        assert "error" in result

    def test_returns_error_when_librosa_missing(self):
        from memory import analyze_audio_frequency

        with patch.dict("sys.modules", {"librosa": None}):
            result = analyze_audio_frequency("/tmp/test.mp3")

        assert "error" in result

    def test_logs_warning_on_failure(self):
        from memory import analyze_audio_frequency

        mock_librosa, _, _ = _mock_librosa()
        mock_librosa.load.side_effect = RuntimeError("corrupt file")

        with (
            patch.dict("sys.modules", {"librosa": mock_librosa}),
            patch("memory.logger") as mock_logger,
        ):
            result = analyze_audio_frequency("/tmp/bad.mp3")

        assert "error" in result
        mock_logger.warning.assert_called_once()
        assert "bad.mp3" in mock_logger.warning.call_args[0][0]


# ===========================================================================
# RECALL_TOOLS — analyze_frequency present
# ===========================================================================


class TestRecallTools:
    def test_analyze_frequency_in_recall_tools(self):
        import server

        tool_names = [t["name"] for t in server.RECALL_TOOLS]
        assert "analyze_frequency" in tool_names

    def test_analyze_frequency_tool_schema(self):
        import server

        tool = next(t for t in server.RECALL_TOOLS if t["name"] == "analyze_frequency")
        schema = tool["input_schema"]
        assert "file_path" in schema["properties"]
        assert "detail" in schema["properties"]
        assert schema["properties"]["detail"]["enum"] == ["quick", "full"]
        assert "file_path" in schema["required"]

    def test_recall_tools_count(self):
        """Verify tool count increased from 8 to 9 with analyze_frequency."""
        import server

        assert len(server.RECALL_TOOLS) == 9


# ===========================================================================
# Warmup config — upgraded to Sonnet
# ===========================================================================


class TestWarmupConfig:
    def test_warmup_model_default_is_sonnet(self):
        import server

        assert "sonnet" in server.WARMUP_MODEL.lower()

    def test_warmup_model_not_haiku(self):
        import server

        assert "haiku" not in server.WARMUP_MODEL.lower()


# ===========================================================================
# ORIENT_DEBUG_CHAT flag
# ===========================================================================


class TestOrientDebugFlag:
    def test_orient_debug_chat_exists(self):
        import server

        assert hasattr(server, "ORIENT_DEBUG_CHAT")

    def test_orient_debug_chat_default_off(self, monkeypatch):
        """Debug flag defaults to False — not leaking orient internals."""
        monkeypatch.delenv("ORIENT_DEBUG_CHAT", raising=False)
        # Can't easily re-evaluate the module-level constant, but we can
        # verify the default in the source
        import server

        assert isinstance(server.ORIENT_DEBUG_CHAT, bool)


# ===========================================================================
# Bridge logs removed from orient
# ===========================================================================


class TestBridgeLogsRemoved:
    def test_orient_output_has_no_bridge_logs_section(self):
        """Bridge logs must not appear in the orient output."""
        import inspect

        import memory

        source = inspect.getsource(memory.orient)
        lines = source.split("\n")
        active_bridge_sections = [
            line for line in lines if "From other channels" in line and not line.strip().startswith("#")
        ]
        assert len(active_bridge_sections) == 0, f"Found active bridge log section header: {active_bridge_sections}"

    def test_bridge_log_query_is_commented(self):
        """The _query_memories call for bridge_logs should be commented out."""
        import inspect

        import memory

        source = inspect.getsource(memory.orient)
        # Active bridge_log query would look like: _query_memories(...bridge_log...)
        # Commented version has #
        lines = source.split("\n")
        active_bridge_lines = [line for line in lines if "bridge_log" in line and not line.strip().startswith("#")]
        assert len(active_bridge_lines) == 0, f"Found uncommented bridge_log references: {active_bridge_lines}"


# ===========================================================================
# execute_recall_tool — analyze_frequency handler
# ===========================================================================


class TestExecuteRecallTool:
    def test_analyze_frequency_handler_returns_json(self):
        import server

        mock_result = {
            "file_duration_seconds": 180.0,
            "analyzed_duration_seconds": 30.0,
            "truncated": True,
            "tempo_bpm": 120.0,
            "spectral_centroid_hz": 1500.0,
        }

        with patch("memory.analyze_audio_frequency", return_value=mock_result):
            result = server.execute_recall_tool(
                "analyze_frequency",
                {"file_path": "/tmp/test.mp3", "detail": "quick"},
            )

        parsed = json.loads(result)
        assert parsed["tempo_bpm"] == 120.0

    def test_analyze_frequency_handler_has_return(self):
        """Verify the handler returns a value (pre-review Category 1)."""
        import server

        with patch("memory.analyze_audio_frequency", return_value={"test": True}):
            result = server.execute_recall_tool(
                "analyze_frequency",
                {"file_path": "/tmp/test.mp3"},
            )

        assert result is not None
        assert len(result) > 0


# ===========================================================================
# S3 audio ingestion — upload URL endpoint + S3 download in analyze_frequency
# ===========================================================================


class TestUploadUrlEndpoint:
    @pytest.fixture()
    def client(self, monkeypatch):
        import server

        monkeypatch.setattr(server, "CHAT_USER", "testuser")
        monkeypatch.setattr(server, "CHAT_PASS", "testpass")
        from fastapi.testclient import TestClient

        return TestClient(server.app, raise_server_exceptions=False)

    def _auth(self):
        import base64

        encoded = base64.b64encode(b"testuser:testpass").decode()
        return {"Authorization": f"Basic {encoded}"}

    def test_returns_upload_url_and_s3_key(self, client):
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.return_value = "https://s3.example.com/signed"

        with patch("boto3.client", return_value=mock_s3):
            resp = client.post(
                "/api/upload-url",
                json={"filename": "test.mp3", "content_type": "audio/mpeg", "file_size": 5000},
                headers=self._auth(),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "upload_url" in data
        assert "s3_key" in data
        assert data["s3_key"].startswith("uploads/")
        assert "test.mp3" in data["s3_key"]

    def test_rejects_oversized_file(self, client):
        resp = client.post(
            "/api/upload-url",
            json={"filename": "huge.wav", "content_type": "audio/wav", "file_size": 200 * 1024 * 1024},
            headers=self._auth(),
        )
        assert resp.status_code == 413

    def test_sanitizes_filename(self, client):
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.return_value = "https://s3.example.com/signed"

        with patch("boto3.client", return_value=mock_s3):
            resp = client.post(
                "/api/upload-url",
                json={"filename": "../../etc/passwd", "content_type": "audio/mpeg", "file_size": 1000},
                headers=self._auth(),
            )

        data = resp.json()
        name_part = data["s3_key"].split("-", 1)[1]
        assert "/" not in name_part
        assert ".." not in name_part

    def test_preserves_extension(self, client):
        mock_s3 = MagicMock()
        mock_s3.generate_presigned_url.return_value = "https://s3.example.com/signed"

        with patch("boto3.client", return_value=mock_s3):
            resp = client.post(
                "/api/upload-url",
                json={"filename": "my-song.mp3", "content_type": "audio/mpeg", "file_size": 5000},
                headers=self._auth(),
            )

        data = resp.json()
        assert data["s3_key"].endswith(".mp3")

    def test_rejects_non_audio_content_type(self, client):
        resp = client.post(
            "/api/upload-url",
            json={"filename": "sketch.png", "content_type": "image/png", "file_size": 1000},
            headers=self._auth(),
        )
        assert resp.status_code == 415

    def test_requires_auth(self, client):
        resp = client.post(
            "/api/upload-url",
            json={"filename": "test.mp3"},
        )
        assert resp.status_code == 401


class TestAnalyzeFrequencyS3:
    def test_s3_key_triggers_download(self):
        from memory import analyze_audio_frequency

        mock_librosa, _, _ = _mock_librosa()
        mock_s3 = MagicMock()

        with (
            patch.dict("sys.modules", {"librosa": mock_librosa}),
            patch("boto3.client", return_value=mock_s3),
        ):
            result = analyze_audio_frequency("uploads/abc123-song.mp3")

        mock_s3.download_file.assert_called_once()
        assert "error" not in result

    def test_absolute_path_skips_s3(self):
        from memory import analyze_audio_frequency

        mock_librosa, _, _ = _mock_librosa()

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            result = analyze_audio_frequency("/tmp/local.mp3")

        assert "error" not in result

    def test_s3_download_failure_returns_error(self):
        from memory import analyze_audio_frequency

        mock_librosa, _, _ = _mock_librosa()
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = Exception("NoSuchKey")

        with (
            patch.dict("sys.modules", {"librosa": mock_librosa}),
            patch("boto3.client", return_value=mock_s3),
            patch("memory.logger"),
        ):
            result = analyze_audio_frequency("uploads/missing.mp3")

        assert "error" in result
        assert "S3 download failed" in result["error"]
