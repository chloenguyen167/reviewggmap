"""Kiểm thử scraper Playwright trên trang Google Maps giả lập (không cần Internet).

Bỏ qua nếu máy chưa cài Playwright/Chromium.
"""
import os

import pandas as pd
import pytest

from src.utils.config import load_config, set_by_dotted

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def server():
    from tests.mock_gmaps import serve
    srv = serve()
    yield srv
    srv.shutdown()


def _cfg(tmp_path):
    cfg = load_config("config.yaml")
    cfg = set_by_dotted(cfg, "paths.root", str(tmp_path / "data"))
    pw = cfg["collect"]["playwright"]
    pw.update({"scroll_pause_ms": [150, 250], "max_idle_rounds": 4, "delay_between_places_s": [0, 0],
               "headless": True})
    exe = os.environ.get("PW_CHROMIUM")          # tùy chọn: đường dẫn Chromium khác phiên bản
    if exe:
        pw["executable_path"] = exe
    return cfg


def test_scrape_mock_place_end_to_end(server, tmp_path):
    from src.collect import parse_raw
    from src.collect.scrape_playwright import MapsScraper
    from src.utils.config import Paths
    from tests.mock_gmaps import N_REVIEWS

    cfg = _cfg(tmp_path)
    url = f"http://127.0.0.1:{server.server_address[1]}/place/Quan+Test"
    row = {"place_key": "mock-place", "name": "Quán Test", "url": url, "user_rating_count": N_REVIEWS}
    with MapsScraper(cfg) as s:
        stats = s.scrape_place(row)
    assert stats["fetched"] == N_REVIEWS, stats

    P = Paths(cfg)
    recs = pd.read_json(P.raw / "reviews" / "pw_mock-place.jsonl", lines=True)
    assert recs["review_id"].is_unique
    assert not recs["text"].str.contains("…").any(), "nút 'Thêm' chưa được bấm hết"
    assert recs["contributor_id"].notna().all()
    assert len(list((P.raw / "network" / "mock-place").glob("*.txt"))) >= 4, "chưa bắt được gói RPC"

    # Parse: ngày lấy từ RPC phải khớp khoảng ngày tương đối và được dùng cho hầu hết review
    parse_raw.run(cfg)
    df = pd.read_parquet(P.interim / "reviews_parsed.parquet")
    assert len(df) == N_REVIEWS
    assert (df["date_source"] == "rpc").mean() > 0.9
    assert df["rating"].between(1, 5).all()
    assert df["reviewer_local_guide"].sum() == len([i for i in range(N_REVIEWS) if i % 4 == 0])
    assert df["has_owner_response"].sum() == len([i for i in range(N_REVIEWS) if i % 7 == 0])

    # Chạy lại: địa điểm đã xong phải được bỏ qua (checkpoint)
    with MapsScraper(cfg) as s:
        again = s.scrape_place(row)
    assert again["fetched"] == N_REVIEWS
