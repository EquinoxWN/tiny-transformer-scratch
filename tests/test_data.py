import hashlib
import io

import pytest

from tiny_transformer_scratch import data

BODY = b"Once upon a time, there was a little dog.\n" * 100


@pytest.fixture
def served(monkeypatch):
    """Serve BODY for every URL and pin the expected file to it."""
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(served.body)

    served.body = BODY
    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setitem(
        data.FILES, "valid", data.DataFile("v.txt", hashlib.sha256(BODY).hexdigest(), len(BODY))
    )
    served.urls = urls
    return served


def test_a_verified_download_lands_in_place(tmp_path, served):
    path = data.download("valid", tmp_path)
    assert path.read_bytes() == BODY
    assert served.urls == [data.BASE_URL + "v.txt"]
    assert data.REVISION in served.urls[0]


def test_a_tampered_download_is_refused_and_removed(tmp_path, served):
    served.body = BODY.replace(b"dog", b"cat")
    with pytest.raises(data.IntegrityError, match="SHA-256"):
        data.download("valid", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_download_larger_than_published_is_cut_off(tmp_path, served):
    served.body = BODY + b"x" * (2 << 20)
    with pytest.raises(data.IntegrityError, match="more than"):
        data.download("valid", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_existing_copy_is_checked_not_trusted(tmp_path, served):
    (tmp_path / "v.txt").write_bytes(BODY[:-1] + b"?")
    with pytest.raises(data.IntegrityError, match="delete it"):
        data.download("valid", tmp_path)
    assert served.urls == []


def test_an_existing_verified_copy_is_reused(tmp_path, served):
    (tmp_path / "v.txt").write_bytes(BODY)
    assert data.download("valid", tmp_path).read_bytes() == BODY
    assert served.urls == []
