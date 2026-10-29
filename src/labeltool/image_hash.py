"""Phát hiện ảnh dùng lại giữa các review (cùng một ảnh đăng ở nhiều review / nhiều quán).

URL ảnh khác nhau ngay cả khi cùng một tấm ảnh được tải lên lại, nên phải so nội dung:
  1. tải ảnh thu nhỏ (mặc định 64px, ~3 KB/ảnh) — có bộ nhớ đệm, chạy lại chỉ tải ảnh mới
  2. dHash 64 bit theo chiều ngang + 64 bit theo chiều dọc (bền với nén lại / đổi kích thước)
  3. hai ảnh "giống nhau" khi khoảng cách Hamming của CẢ HAI hash ≤ hash_max_dist;
     ứng viên tìm bằng LSH (8 dải × 8 bit của hash ngang: Hamming ≤ 7 chắc chắn trùng ít nhất 1 dải)
  4. gom nhóm bằng union-find. Ảnh gần như phẳng (ít thông tin) bị bỏ để tránh ghép nhầm.

Đầu ra: interim/image_hashes.parquet (bộ đệm), processed/features_img.parquet
        (review_id, idx, photo, img_group: -1 = không trùng ảnh nào)
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

from src.utils.common import get_logger
from src.utils.config import Paths

DEFAULTS = {"thumb": "=w64-h64-k-no", "workers": 16, "hash_max_dist": 6, "min_bits": 8, "max_bucket": 3000}
_local = threading.local()


def dhash(img: Image.Image) -> tuple[int, int]:
    g = img.convert("L")
    h = np.asarray(g.resize((9, 8), Image.Resampling.BOX), dtype=np.int16)
    v = np.asarray(g.resize((8, 9), Image.Resampling.BOX), dtype=np.int16)
    bits_h = (h[:, 1:] > h[:, :-1]).flatten()
    bits_v = (v[1:, :] > v[:-1, :]).flatten()
    to_int = lambda b: int("".join("1" if x else "0" for x in b), 2)
    return to_int(bits_h), to_int(bits_v)


def _signed(x: int) -> int:
    return x - (1 << 64) if x >= (1 << 63) else x


def _fetch(url: str, timeout: float) -> tuple[str, int | None, int | None]:
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    for _ in range(2):
        try:
            r = s.get(url, timeout=timeout)
            if r.status_code == 200:
                h, v = dhash(Image.open(BytesIO(r.content)))
                return url, _signed(h), _signed(v)
            if r.status_code in (403, 404, 410):
                break
        except Exception:
            continue
    return url, None, None


def download_hashes(photos: list[str], cache: Path, c: dict, log, timeout: float = 15) -> pd.DataFrame:
    frame = lambda rows: pd.DataFrame(rows, columns=["photo", "dh", "dv"]).astype({"dh": "Int64", "dv": "Int64"})
    old = pd.read_parquet(cache).astype({"dh": "Int64", "dv": "Int64"}) if cache.exists() else frame([])
    old = old.dropna(subset=["dh"])                    # ảnh lỗi tải lần trước (thường do mạng) → thử lại
    todo = sorted(set(photos) - set(old["photo"]))
    log.info("Băm ảnh: %d ảnh đã có trong bộ đệm, cần tải %d ảnh", len(old), len(todo))
    rows, done = [], 0

    def flush():
        nonlocal old, rows
        if rows:
            old = pd.concat([old, frame(rows)], ignore_index=True)
            old.to_parquet(cache, index=False)
            rows = []

    with ThreadPoolExecutor(max_workers=c["workers"]) as ex:
        futs = {ex.submit(_fetch, p + c["thumb"], timeout): p for p in todo}
        for f in as_completed(futs):
            _, h, v = f.result()
            rows.append((futs[f], h, v))
            done += 1
            if done % 5000 == 0:
                flush()
                log.info("  … %d/%d", done, len(todo))
    flush()
    return old


def group_images(h: np.ndarray, v: np.ndarray, max_dist: int, max_bucket: int) -> np.ndarray:
    """Trả về mã nhóm cho từng ảnh (-1 = không giống ảnh nào khác)."""
    n = len(h)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hu, vu = h.view(np.uint64), v.view(np.uint64)
    for band in range(8):
        key = (hu >> np.uint64(8 * band)) & np.uint64(0xFF)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1
        for idx in np.split(order, cuts):
            if len(idx) < 2 or len(idx) > max_bucket:
                continue
            dh = np.bitwise_count(hu[idx][:, None] ^ hu[idx][None, :])
            dv = np.bitwise_count(vu[idx][:, None] ^ vu[idx][None, :])
            ii, jj = np.nonzero(np.triu((dh <= max_dist) & (dv <= max_dist), 1))
            for a, b in zip(idx[ii], idx[jj]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    size = pd.Series(roots).map(pd.Series(roots).value_counts()).to_numpy()
    _, gid = np.unique(roots, return_inverse=True)
    return np.where(size > 1, gid, -1)


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("img_hash", P.logs)
    c = {**DEFAULTS, **cfg.get("labeltool", {}).get("image_hash", {})}
    media = pd.read_parquet(P.interim / "review_media.parquet")
    hashes = download_hashes(media["photo"].unique().tolist(), P.interim / "image_hashes.parquet", c, log)
    ok = hashes.dropna(subset=["dh", "dv"]).copy()
    ok["dh"], ok["dv"] = ok["dh"].astype(np.int64), ok["dv"].astype(np.int64)
    bits = np.bitwise_count(ok["dh"].to_numpy().view(np.uint64))
    ok = ok[(bits >= c["min_bits"]) & (bits <= 64 - c["min_bits"])].reset_index(drop=True)
    ok["img_group"] = group_images(ok["dh"].to_numpy(), ok["dv"].to_numpy(), c["hash_max_dist"], c["max_bucket"])
    out = media.merge(ok[["photo", "img_group"]], on="photo", how="left")
    out["img_group"] = out["img_group"].fillna(-1).astype(int)
    # chỉ giữ nhóm có ảnh từ ≥ 2 review khác nhau (nhiều ảnh gần giống trong cùng một review không tính)
    n_rev = out[out["img_group"] >= 0].groupby("img_group")["review_id"].nunique()
    out.loc[out["img_group"].map(n_rev).fillna(0) < 2, "img_group"] = -1
    path = P.processed / "features_img.parquet"
    out[["review_id", "idx", "photo", "img_group"]].to_parquet(path, index=False)
    g = out[out["img_group"] >= 0]
    log.info("Ảnh: %d ảnh băm được / %d (lỗi tải %d) | %d nhóm ảnh trùng, %d review có ảnh trùng với review khác",
             len(hashes.dropna(subset=["dh"])), media["photo"].nunique(), int(hashes["dh"].isna().sum()),
             g["img_group"].nunique(), g["review_id"].nunique())
    return path
