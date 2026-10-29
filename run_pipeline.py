"""Điều phối pipeline.

Ví dụ:
  # Chạy thử toàn bộ trên dữ liệu DEMO (không cần mạng)
  python run_pipeline.py --config config.demo.yaml --stages demo,all,demo_labels,eval

  # Dữ liệu thật
  python run_pipeline.py --stages search        # tìm địa điểm → sửa tay places_selected.csv
  python run_pipeline.py --stages reviews       # lấy review (Playwright hoặc SerpApi, theo config)
  python run_pipeline.py --stages benchmark     # tải ViSpamReviews v2 (cần HF_TOKEN trong .env) cho M1b
  python run_pipeline.py --stages all           # parse → clean → quality → eda → m1a,m1b,m2,m3 → fusion → sample
  python run_pipeline.py --stages ws            # PHƯƠNG PHÁP CHÍNH: 8 luật (LF) → label model Dawid–Skene → ws_scores.parquet
  python run_pipeline.py --stages contributors  # (tùy chọn) lịch sử reviewer nghi vấn, rồi chạy lại parse..fusion
  # ... 2 người gán nhãn điền data/labels/annotator_A.csv, annotator_B.csv (item_id 1..human_n) ...
  python run_pipeline.py --stages llm_export    # tạo batch CSV + PROMPT.md để Claude (claude.ai) gán toàn bộ gold set
  python run_pipeline.py --stages llm_import    # gộp CSV kết quả từ claude.ai → annotator_C.csv
  python run_pipeline.py --stages eval          # agreement → metrics → ablation → sensitivity

  # Công cụ gán nhãn theo quán (1 file HTML, giao diện kiểu Google Maps)
  python run_pipeline.py --stages labeltool_all # media → img_hash → labeltool → data/labels/labeltool/*.html

Nhóm bước:
  all  = parse,clean,quality,eda,m1a,m1b,m2,m3,fusion,sample
  eval = agreement,metrics,ablation,sensitivity
  labeltool_all = media,img_hash,labeltool
"""
from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

from src.utils.common import get_logger
from src.utils.config import Paths, load_config

GROUPS = {
    "all": ["parse", "clean", "quality", "eda", "m1a", "m1b", "m2", "m3", "fusion", "sample"],
    "eval": ["agreement", "metrics", "ablation", "sensitivity"],
    "labeltool_all": ["media", "img_hash", "labeltool"],
}


def _stage(name: str, cfg: dict):
    src = cfg["collect"]["source"]
    if name == "search":
        if src == "serpapi":
            from src.collect.serpapi_client import run_search
        else:
            from src.collect.scrape_playwright import run_search
        return run_search(cfg)
    if name == "reviews":
        if src == "serpapi":
            from src.collect.serpapi_client import run_reviews
        else:
            from src.collect.scrape_playwright import run_reviews
        return run_reviews(cfg)
    if name == "contributors":
        import pandas as pd
        P = Paths(cfg)
        scored = pd.read_parquet(P.processed / "scored_reviews.parquet")
        pmap = pd.read_parquet(P.interim / "private_reviewer_map.parquet")
        top = (scored.groupby("reviewer_hash")["suspicion"].max().sort_values(ascending=False)
               .head(cfg["collect"]["contributors"]["top_n"]).index)
        ids = pmap[pmap["reviewer_hash"].isin(top)]["contributor_id"].astype(str).tolist()
        if src == "serpapi":
            from src.collect.serpapi_client import run_contributors
        else:
            from src.collect.scrape_playwright import run_contributors
        return run_contributors(cfg, ids)
    mod = {
        "demo": "src.synthetic.make_demo_data", "benchmark": "src.collect.download_benchmark", "parse": "src.collect.parse_raw", "clean": "src.preprocess.clean",
        "quality": "src.preprocess.quality", "eda": "src.preprocess.eda", "m1a": "src.modules.m1a_duplicate",
        "m1b": "src.modules.m1b_content", "m1b_phobert": "src.modules.m1b_phobert", "m2": "src.modules.m2_temporal", "m3": "src.modules.m3_behavior",
        "fusion": "src.fusion.score", "ws": "src.fusion.ws_label_model", "sample": "src.evaluation.sample_gold",
        "demo_labels": "src.synthetic.simulate_annotations", "agreement": "src.evaluation.agreement",
        "metrics": "src.evaluation.metrics",
        "media": "src.labeltool.media", "img_hash": "src.labeltool.image_hash", "labeltool": "src.labeltool.build",
    }
    if name == "llm_export":
        from src.evaluation.llm_label import run_export
        return run_export(cfg)
    if name == "llm_import":
        from src.evaluation.llm_label import run_import
        return run_import(cfg)
    if name == "ablation":
        from src.evaluation.ablation import run_ablation
        return run_ablation(cfg)
    if name == "sensitivity":
        from src.evaluation.ablation import run_sensitivity
        return run_sensitivity(cfg)
    if name not in mod:
        raise SystemExit(f"Bước không hợp lệ: {name}")
    import importlib
    return importlib.import_module(mod[name]).run(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stages", default="all", help="danh sách bước, cách nhau bằng dấu phẩy")
    args = ap.parse_args(argv)
    load_dotenv()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    log = get_logger("pipeline", P.logs)
    stages = []
    for s in args.stages.split(","):
        stages += GROUPS.get(s.strip(), [s.strip()])
    log.info("Config: %s | dữ liệu: %s | các bước: %s", args.config, P.root, ", ".join(stages))
    for s in stages:
        t0 = time.time()
        log.info("▶ %s", s)
        out = _stage(s, cfg)
        log.info("✔ %s (%.1fs) → %s", s, time.time() - t0, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
