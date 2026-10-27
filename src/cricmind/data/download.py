"""Download the pinned Cricsheet snapshot into data/raw/<DATA_SNAPSHOT>/ (D-020, D-021).

uv run python -m cricmind.data.download [--force]
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from cricmind import config

MANIFEST = "MANIFEST.json"
CHUNK = 1 << 20


def _fetch(url: str, dest: Path) -> dict:
    """Stream url to dest via .part (no half files on failure); return its manifest entry."""
    req = urllib.request.Request(url, headers={"User-Agent": "cricmind-research/0.1"})
    sha, size = hashlib.sha256(), 0
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
        last_modified = resp.headers.get("Last-Modified")
        while chunk := resp.read(CHUNK):
            sha.update(chunk)
            size += len(chunk)
            out.write(chunk)
    part.replace(dest)
    return {"url": url, "sha256": sha.hexdigest(), "bytes": size, "last_modified": last_modified}


def download(force: bool = False, raw_dir: Path | None = None, urls: dict[str, str] | None = None) -> Path:
    """Fetch all sources; the manifest is written last, so its presence means a complete snapshot."""
    raw_dir = raw_dir or config.DATA_RAW / config.DATA_SNAPSHOT
    urls = urls or config.CRICSHEET_URLS
    manifest_path = raw_dir / MANIFEST
    if manifest_path.exists() and not force:
        return raw_dir  # D-021: no silent re-download
    if force:
        shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = {name: _fetch(url, raw_dir / name) for name, url in urls.items()}
    for name, entry in files.items():
        if name.endswith(".zip"):
            with zipfile.ZipFile(raw_dir / name) as zf:
                # Count before the D-020 scope filter, so the parser can report how many it dropped.
                entry["n_match_files"] = sum(n.endswith(".json") for n in zf.namelist())

    manifest = {"snapshot": config.DATA_SNAPSHOT, "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return raw_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download, replacing the snapshot")
    path = download(force=ap.parse_args().force)
    print(f"snapshot ready: {path}")
