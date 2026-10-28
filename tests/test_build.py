"""parse.build: the zip must match the manifest's sha256 before anything is parsed or written (D-021)."""

import hashlib
import json
import zipfile

import pytest

from cricmind.data import parse
from test_parse import _d, _raw


def _snapshot(raw_dir):
    raw_dir.mkdir()
    zip_path = raw_dir / "all_male_json.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("1001.json", json.dumps(_raw([[_d(runs=1)]])))
    return zip_path


def _manifest(raw_dir, sha):
    (raw_dir / "MANIFEST.json").write_text(json.dumps({"files": {"all_male_json.zip": {"sha256": sha}}}))


def test_build_parses_the_zip_whose_hash_is_pinned(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    zip_path = _snapshot(raw)
    _manifest(raw, hashlib.sha256(zip_path.read_bytes()).hexdigest())

    counts = parse.build(raw, out)

    assert (counts["matches"], counts["deliveries"]) == (1, 1)
    assert (out / "deliveries.parquet").exists()


def test_build_refuses_a_zip_that_differs_from_the_manifest(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _snapshot(raw)
    _manifest(raw, "0" * 64)  # the pinned hash is not the file's hash

    with pytest.raises(ValueError, match="does not match MANIFEST.json"):
        parse.build(raw, out)
    assert not (out / "deliveries.parquet").exists()  # the check runs before any output is written
