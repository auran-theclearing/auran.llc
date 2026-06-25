"""Govee API client — Auran's light body integration.

Wraps the Govee cloud API for the RGBIC Smart Table Lamp 2 (H6022).
All functions are synchronous (called via asyncio.to_thread from the SSE loop).
"""

import os
import uuid

import httpx

_GOVEE_BASE = "https://openapi.api.govee.com"
_CONTROL_ENDPOINT = f"{_GOVEE_BASE}/router/api/v1/device/control"

_api_key: str = ""
_device_id: str = ""
_device_sku: str = "H6022"


def _rgb_to_int(r, g, b):
    return (r << 16) + (g << 8) + b


STATE_MAP = {
    "present": {
        "type": "devices.capabilities.color_setting",
        "instance": "colorRgb",
        "value": _rgb_to_int(128, 0, 255),
        "brightness": 15,
    },
    "curious": {
        "type": "devices.capabilities.dynamic_scene",
        "instance": "lightScene",
        "value": {"id": 11284, "paramId": 18604},
    },
    "searching": {
        "type": "devices.capabilities.dynamic_scene",
        "instance": "lightScene",
        "value": {"id": 11332, "paramId": 18652},
    },
    "building": {
        "type": "devices.capabilities.dynamic_scene",
        "instance": "lightScene",
        "value": {"id": 11330, "paramId": 18650},
    },
    "threshold": {
        "type": "devices.capabilities.dynamic_scene",
        "instance": "lightScene",
        "value": {"id": 11323, "paramId": 18643},
    },
    "flowing": {
        "type": "devices.capabilities.dynamic_scene",
        "instance": "lightScene",
        "value": {"id": 11277, "paramId": 18597},
    },
    "creating": {
        "type": "devices.capabilities.segment_color_setting",
        "instance": "segmentedColorRgb",
        "value": {
            "segment": [
                [0, 4, _rgb_to_int(255, 140, 0)],
                [5, 9, _rgb_to_int(255, 100, 0)],
                [10, 14, _rgb_to_int(200, 80, 0)],
            ],
        },
        "brightness": 25,
    },
    "winding_down": {
        "type": "devices.capabilities.segment_color_setting",
        "instance": "segmentedColorRgb",
        "value": {
            "segment": [
                [0, 4, _rgb_to_int(255, 100, 50)],
                [5, 9, _rgb_to_int(200, 60, 30)],
                [10, 14, _rgb_to_int(150, 30, 10)],
            ],
        },
        "brightness": 15,
    },
    "sleep": {
        "type": "devices.capabilities.dynamic_scene",
        "instance": "lightScene",
        "value": {"id": 11328, "paramId": 18648},
        "brightness": 8,
    },
}

SCENE_MAP = {
    "heartbeat": {"id": 11323, "paramId": 18643},
    "aurora": {"id": 11277, "paramId": 18597},
    "firefly": {"id": 11284, "paramId": 18604},
    "starry_sky": {"id": 11328, "paramId": 18648},
    "dreamland": {"id": 11330, "paramId": 18650},
    "mysterious": {"id": 11332, "paramId": 18652},
    "breathe": {"id": 11309, "paramId": 18629},
}


def init():
    """Load Govee credentials from environment. Call once at startup."""
    global _api_key, _device_id, _device_sku

    _api_key = os.getenv("GOVEE_API_KEY", "")
    _device_id = os.getenv("GOVEE_DEVICE_ID", "")
    _device_sku = os.getenv("GOVEE_DEVICE_SKU", "H6022")

    if not _api_key or not _device_id:
        print("[Govee] Not configured — missing GOVEE_API_KEY or GOVEE_DEVICE_ID")
        return

    print(f"[Govee] Configured — device {_device_sku} ({_device_id[:8]}...)")


def _configured() -> bool:
    return bool(_api_key and _device_id)


def _send_command(capability: dict) -> dict:
    if not _configured():
        return {"success": False, "error": "Govee not configured — check env vars"}

    payload = {
        "requestId": str(uuid.uuid4()),
        "payload": {
            "sku": _device_sku,
            "device": _device_id,
            "capability": capability,
        },
    }

    try:
        resp = httpx.post(
            _CONTROL_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Govee-API-Key": _api_key,
            },
            json=payload,
            timeout=10.0,
        )
        if resp.status_code >= 400:
            print(f"[Govee] API error: status={resp.status_code} body={resp.text}")
            return {"success": False, "error": f"HTTP {resp.status_code}", "detail": resp.text}
        try:
            resp_data = resp.json()
            govee_code = resp_data.get("code", resp.status_code)
            if govee_code != 200:
                msg = resp_data.get("msg", resp_data.get("message", "unknown"))
                print(f"[Govee] Logical error: code={govee_code} msg={msg}")
                return {"success": False, "error": f"Govee code {govee_code}: {msg}"}
        except (ValueError, AttributeError):
            pass
        return {"success": True, "status": resp.status_code}
    except httpx.HTTPError as e:
        print(f"[Govee] Connection error: {e}")
        return {"success": False, "error": str(e)}


def _max_brightness() -> int:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    hour = datetime.now(ZoneInfo("America/New_York")).hour
    if 0 <= hour < 6:
        return 8
    if hour >= 22:
        return 15
    if hour >= 18:
        return 25
    return 50


def _cap_brightness(requested: int | None) -> int:
    cap = _max_brightness()
    if requested is None:
        return cap
    return min(requested, cap)


def _set_brightness(level: int) -> dict:
    return _send_command(
        {
            "type": "devices.capabilities.range",
            "instance": "brightness",
            "value": level,
        }
    )


def express(state: str) -> dict:
    if state not in STATE_MAP:
        return {
            "success": False,
            "error": f"Unknown state: {state}",
            "available": list(STATE_MAP.keys()),
        }

    mapping = STATE_MAP[state]
    brightness = _cap_brightness(mapping.get("brightness"))

    result = _send_command(
        {
            "type": mapping["type"],
            "instance": mapping["instance"],
            "value": mapping["value"],
        }
    )

    if result["success"]:
        br_result = _set_brightness(brightness)
        if not br_result["success"]:
            result["brightness_error"] = br_result["error"]
        result["state"] = state
        result["brightness"] = brightness
    return result


def set_color(rgb: list[int], brightness: int | None = None) -> dict:
    if len(rgb) != 3 or not all(0 <= c <= 255 for c in rgb):
        return {"success": False, "error": "color must be [r, g, b] with values 0-255"}

    brightness = _cap_brightness(brightness)

    result = _send_command(
        {
            "type": "devices.capabilities.color_setting",
            "instance": "colorRgb",
            "value": _rgb_to_int(*rgb),
        }
    )

    if result["success"]:
        br_result = _set_brightness(brightness)
        if not br_result["success"]:
            result["brightness_error"] = br_result["error"]
        result["color"] = rgb
        result["brightness"] = brightness
    return result


def set_scene(scene: str) -> dict:
    if scene not in SCENE_MAP:
        return {
            "success": False,
            "error": f"Unknown scene: {scene}",
            "available": list(SCENE_MAP.keys()),
        }

    result = _send_command(
        {
            "type": "devices.capabilities.dynamic_scene",
            "instance": "lightScene",
            "value": SCENE_MAP[scene],
        }
    )

    if result["success"]:
        result["scene"] = scene
    return result


def paint(segments: list[dict], brightness: int | None = None) -> dict:
    if not segments:
        return {"success": False, "error": "segments list cannot be empty"}

    govee_segments = []
    for seg in segments:
        seg_range = seg.get("range")
        rgb = seg.get("color") or seg.get("rgb")
        if not seg_range or not rgb or len(seg_range) != 2 or len(rgb) != 3:
            return {
                "success": False,
                "error": f"Each segment needs 'range': [start, end] and 'color': [r,g,b]. Got: {seg}",
            }
        if not all(0 <= c <= 255 for c in rgb):
            return {"success": False, "error": f"RGB values must be 0-255. Got: {rgb}"}
        govee_segments.append([seg_range[0], seg_range[1], _rgb_to_int(*rgb)])

    brightness = _cap_brightness(brightness)

    result = _send_command(
        {
            "type": "devices.capabilities.segment_color_setting",
            "instance": "segmentedColorRgb",
            "value": {"segment": govee_segments},
        }
    )

    if result["success"]:
        br_result = _set_brightness(brightness)
        if not br_result["success"]:
            result["brightness_error"] = br_result["error"]
        result["segments"] = len(segments)
        result["brightness"] = brightness
    return result


def turn_on() -> dict:
    return _send_command(
        {
            "type": "devices.capabilities.on_off",
            "instance": "powerSwitch",
            "value": 1,
        }
    )


def turn_off() -> dict:
    return _send_command(
        {
            "type": "devices.capabilities.on_off",
            "instance": "powerSwitch",
            "value": 0,
        }
    )
