"""Shared camera serial configuration for robot definitions."""

from __future__ import annotations

# PiperX plug task: D435I head camera and D405 wrist camera.
PIPER_SINGLE_CAMERA_SERIALS = {
    "head": "337122071685",
    "wrist": "230322274885",
}

PIPER_DAGGER_CAMERA_SERIALS = {
    "head": "337122071685",
    "wrist": "230322274885",
}


def get_piper_camera_serials(profile: str = "single") -> dict[str, str]:
    """Return camera serials for the requested Piper camera profile."""
    serials_by_profile = {
        "single": PIPER_SINGLE_CAMERA_SERIALS,
        "dagger": PIPER_DAGGER_CAMERA_SERIALS,
    }
    if profile not in serials_by_profile:
        raise ValueError(f"Unknown Piper camera profile: {profile}")
    return dict(serials_by_profile[profile])
