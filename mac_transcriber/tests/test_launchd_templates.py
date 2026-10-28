import plistlib
from pathlib import Path


def test_installed_launchd_templates_are_valid_service_plists():
    root = Path(__file__).resolve().parents[1] / "launchd"
    for suffix in ("transcriber", "tunnel", "reprocess"):
        label = f"com.slack-zoom.gigaam-{suffix}"
        with (root / f"{label}.plist.example").open("rb") as source:
            plist = plistlib.load(source)
        assert plist["Label"] == label
        assert plist["ProgramArguments"]
        assert plist["RunAtLoad"] is True
