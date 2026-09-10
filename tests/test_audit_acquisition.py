from pathlib import Path

import pytest

from solverpilot.benchmark.acquire import _fetch_role


@pytest.mark.parametrize("filename", ["../escape", "sub/file", r"..\escape", r"C:\escape", "C:escape", "file:stream", ".", ".."])
def test_download_rejects_escaping_filenames_before_io(tmp_path, monkeypatch, filename):
    def unexpected_download(*args, **kwargs):
        pytest.fail("invalid destination reached the downloader")
    monkeypatch.setattr("solverpilot.benchmark.acquire._download", unexpected_download)
    with pytest.raises(ValueError, match="filename"):
        _fetch_role(role="archive", url="https://example.invalid/archive.zip", filename=filename,
                    expected_sha256=None, target=tmp_path, timeout_s=1, retries=1, force=True)


def test_local_download_filename_preserved(tmp_path, monkeypatch):
    def download(url, dest, **kwargs):
        dest.write_bytes(b"audit")
    monkeypatch.setattr("solverpilot.benchmark.acquire._download", download)
    result = _fetch_role(role="archive", url="https://example.invalid/archive.zip", filename=None,
                         expected_sha256=None, target=tmp_path, timeout_s=1, retries=1, force=True)
    assert Path(result.path) == tmp_path / "archive.zip"
    assert result.size_bytes == 5
