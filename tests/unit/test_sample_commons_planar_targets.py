"""Commons 候选只保留明确照片许可与可信缩略图地址。"""

from __future__ import annotations

import io

import scripts.sample_commons_planar_targets as commons_sampler
from scripts.sample_commons_planar_targets import (
    CATEGORIES,
    _allowed_thumbnail_url,
    _metadata_eligible,
    _title_family,
)


def test_framed_artwork_category_is_an_explicit_audit_source() -> None:
    assert CATEGORIES["framed-artwork"] == "Framed paintings in the National Gallery, London"


def test_commons_metadata_filter_does_not_promote_unknown_license_or_external_host() -> None:
    candidate = {
        "mime": "image/jpeg",
        "width": 2400,
        "height": 1600,
        "photo_license_metadata": "CC BY-SA 4.0",
        "thumbnail_url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/a1/sample.jpg/480px-sample.jpg",
    }
    assert _metadata_eligible(candidate)
    assert not _metadata_eligible({**candidate, "photo_license_metadata": "unknown"})
    assert not _metadata_eligible({**candidate, "height": 600})
    assert not _metadata_eligible({**candidate, "thumbnail_url": "https://example.com/sample.jpg"})
    assert not _allowed_thumbnail_url("http://thumb.wikimedia.org/wikipedia/commons/sample.jpg")


def test_numbered_views_share_sampling_family() -> None:
    assert _title_family("File:Gallery room View 03.jpg") == _title_family("File:Gallery room View 04.jpg")
    assert _title_family("File:London - Tate Britain - Room A.jpg") == _title_family(
        "File:London - Tate Britain - Room B.jpg"
    )


def test_commons_request_retries_a_transient_timeout(monkeypatch) -> None:
    attempts = 0

    def fake_urlopen(_request, *, timeout):
        nonlocal attempts
        attempts += 1
        assert timeout == 40
        if attempts == 1:
            raise TimeoutError("transient")
        return io.BytesIO(b'{"query": {}}')

    monkeypatch.setattr(commons_sampler, "urlopen", fake_urlopen)
    monkeypatch.setattr(commons_sampler.time, "sleep", lambda _seconds: None)
    assert commons_sampler._request_json({"action": "query"}) == {"query": {}}
    assert attempts == 2
