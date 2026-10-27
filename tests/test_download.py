import hashlib
import json
import zipfile

from cricmind.data import download as dl


def _make_sources(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    with zipfile.ZipFile(src / "m.zip", "w") as zf:
        zf.writestr("1.json", "{}")
        zf.writestr("2.json", "{}")
        zf.writestr("README.txt", "x")
    (src / "people.csv").write_text("identifier,name\n")
    return {name: (src / name).as_uri() for name in ("m.zip", "people.csv")}


def test_download_writes_manifest_and_skips_without_force(tmp_path):
    urls, raw = _make_sources(tmp_path), tmp_path / "raw"
    dl.download(raw_dir=raw, urls=urls)

    files = json.loads((raw / "MANIFEST.json").read_text())["files"]
    assert files["m.zip"]["n_match_files"] == 2
    assert files["m.zip"]["sha256"] == hashlib.sha256((raw / "m.zip").read_bytes()).hexdigest()
    assert files["people.csv"]["bytes"] == len("identifier,name\n")

    (raw / "people.csv").write_text("edited")
    dl.download(raw_dir=raw, urls=urls)  # manifest exists -> untouched
    assert (raw / "people.csv").read_text() == "edited"
    dl.download(force=True, raw_dir=raw, urls=urls)
    assert (raw / "people.csv").read_text() == "identifier,name\n"
