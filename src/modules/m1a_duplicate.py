"""Module 1a — Phát hiện review gần trùng (near-duplicate), không giám sát.

Blocking : MinHash-LSH trên shingle k-từ, toàn corpus (không block theo quán, vì
           review trùng GIỮA CÁC QUÁN mới là dấu hiệu spam mạnh nhất).
Matching : Jaccard thật trên tập shingle của các cặp ứng viên;
           (tùy chọn) cosine của sentence-embedding tiếng Việt để bắt câu viết lại.
Chỉ đo TRÙNG LẶP VĂN BẢN. Mọi bằng chứng thời gian (bản trùng dồn vào vài ngày) thuộc về M2; việc
phân biệt chiến dịch với câu khen sáo rỗng lặp lại tự nhiên được để cho bước gộp điểm
(trùng lặp cao + bất thường thời gian). (Phiên bản trước có thêm "mức bất ngờ Poisson" theo thời gian,
đã gỡ để các module độc lập với nhau — giả định của phép gộp Fisher.)
Đầu ra   : processed/features_m1a.parquet (1 dòng / review), reports/tables/m1a_top_clusters.csv
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, write_table
from src.utils.config import Paths


def shingles(tokens: list[str], k: int) -> set[str]:
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def candidate_pairs_lsh(sh: list[set], num_perm: int, threshold: float, seed: int) -> set[tuple[int, int]]:
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mhs = []
    for i, s in enumerate(sh):
        m = MinHash(num_perm=num_perm, seed=seed)
        for g in s:
            m.update(g.encode("utf-8"))
        lsh.insert(str(i), m)
        mhs.append(m)
    pairs = set()
    for i, m in enumerate(mhs):
        for j in lsh.query(m):
            j = int(j)
            if j != i:
                pairs.add((min(i, j), max(i, j)))
    return pairs


def candidate_pairs_embedding(texts: list[str], cfg_m: dict) -> dict[tuple[int, int], float]:
    """Cặp có cosine cao theo sentence-embedding (tùy chọn)."""
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    model = SentenceTransformer(cfg_m["embedding_model"])
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    k = min(cfg_m["embedding_neighbors"] + 1, len(texts))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(emb)
    dist, idx = nn.kneighbors(emb)
    out = {}
    for i in range(len(texts)):
        for d, j in zip(dist[i], idx[i]):
            if j != i and 1 - d >= cfg_m["embedding_threshold"]:
                out[(min(i, j), max(i, j))] = float(1 - d)
    return out


def compute(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = cfg["m1a"]
    elig = df[(df["n_words"] >= c["min_words"])].reset_index(drop=True)
    toks = [t.split() for t in elig["text_norm"]]
    sh = [shingles(t, c["shingle_k"]) for t in toks]
    pairs = candidate_pairs_lsh(sh, c["num_perm"], c["lsh_threshold"], cfg["project"]["seed"])

    sims: dict[tuple[int, int], float] = {p: jaccard(sh[p[0]], sh[p[1]]) for p in pairs}
    if c.get("use_embeddings"):
        try:
            for p, v in candidate_pairs_embedding(elig["text_norm"].tolist(), c).items():
                sims[p] = max(sims.get(p, 0.0), v)
        except Exception as e:                                  # thiếu thư viện / không tải được model
            get_logger("m1a").warning("Bỏ qua embedding: %s", e)

    edges = [(i, j, s) for (i, j), s in sims.items() if s >= c["sim_threshold"]]
    n = len(elig)
    nbr = defaultdict(list)
    uf = _UF(n)
    for i, j, s in edges:
        nbr[i].append((j, s))
        nbr[j].append((i, s))
        uf.union(i, j)

    place = elig["place_key"].to_numpy()
    user = elig["reviewer_hash"].to_numpy()
    rows = []
    for i in range(n):
        nb = nbr.get(i, [])
        other_places = {place[j] for j, _ in nb if place[j] != place[i]}
        other_users = {user[j] for j, _ in nb if user[j] != user[i] and user[j] is not None}
        max_sim = max((s for _, s in nb), default=0.0)
        raw = (max_sim * (1 + 0.5 * np.log1p(len(other_places)))
               * (1 + 0.5 * np.log1p(len(other_users)))) if nb else 0.0
        reason = (f"gần trùng (sim={max_sim:.2f}) với {len(nb)} review, ở {len(other_places)} quán khác"
                  if nb else "")
        rows.append({"review_id": elig.at[i, "review_id"], "dup_max_sim": max_sim, "dup_n_neighbors": len(nb),
                     "dup_cross_place": len(other_places), "dup_cross_user": len(other_users),
                     "dup_cluster_id": uf.find(i) if nb else -1, "s_dup_raw": raw, "reason_dup": reason})
    feat = pd.DataFrame(rows)
    if not feat.empty:
        size = feat[feat["dup_cluster_id"] >= 0].groupby("dup_cluster_id").size()
        feat["dup_cluster_size"] = feat["dup_cluster_id"].map(size).fillna(0).astype(int)

    # Review không đủ điều kiện (ngắn / không chữ) → NaN (Fusion sẽ bỏ qua module này cho họ)
    feat = df[["review_id"]].merge(feat, on="review_id", how="left")

    # Bảng cụm lớn nhất để đọc tay / làm case study
    cl = feat[feat["dup_cluster_id"].fillna(-1) >= 0].merge(
        df[["review_id", "place_key", "rating", "review_day", "text"]], on="review_id")
    top = (cl.groupby("dup_cluster_id")
             .agg(size=("review_id", "size"), n_places=("place_key", "nunique"),
                  mean_rating=("rating", "mean"), first_day=("review_day", "min"), last_day=("review_day", "max"),
                  example=("text", "first"))
             .sort_values(["n_places", "size"], ascending=False).head(20).reset_index())
    return feat, top


def run(cfg: dict) -> Path:
    P = Paths(cfg).ensure()
    log = get_logger("m1a", P.logs)
    df = pd.read_parquet(P.interim / "reviews_clean.parquet")
    feat, top = compute(df, cfg)
    out = P.processed / "features_m1a.parquet"
    feat.to_parquet(out, index=False)
    write_table(top, P.tables / "m1a_top_clusters.csv")
    has = feat["dup_n_neighbors"].fillna(0) > 0
    cl_size = feat.loc[has].groupby("dup_cluster_id").size()
    log.info("M1a (Jaccard ≥ %.2f): %d review đủ điều kiện, %d có review gần trùng, %d cụm "
             "(%d cụm ≥ 3 review, %d cụm trải ≥ 2 quán)", cfg["m1a"]["sim_threshold"],
             feat["dup_max_sim"].notna().sum(), has.sum(), len(cl_size), int((cl_size >= 3).sum()),
             int((feat.loc[has].groupby("dup_cluster_id")["dup_cross_place"].max() > 0).sum()))
    return out
