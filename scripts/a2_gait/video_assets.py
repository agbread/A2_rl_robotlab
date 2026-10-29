"""Locate video fonts without requiring root or a system-wide font install."""
import os
from pathlib import Path
import sys


def find_video_font():
    override = os.environ.get("A2_VIDEO_FONT")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"A2_VIDEO_FONT does not exist: {path}")
        return str(path.absolute())
    roots = [Path(sys.prefix) / "share/fonts", Path.home() / ".local/share/fonts",
             Path.home() / ".fonts", Path("/usr/share/fonts")]
    for root in roots:
        if root.is_dir():
            matches = sorted(root.rglob("DejaVuSans.ttf"))
            if matches:
                return str(matches[0])
    raise FileNotFoundError("Install conda-forge::font-ttf-dejavu in the MuJoCo environment "
                            "or set A2_VIDEO_FONT to a readable .ttf file")
