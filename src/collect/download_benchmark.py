"""Tải ViSpamReviews v2 (HuggingFace, repo "gated") về data/benchmark/<m1b.benchmark_dir>/.

Một lần duy nhất:
  1. Đăng nhập https://huggingface.co, mở https://huggingface.co/datasets/sonlam1102/vispamdetection2
     và bấm đồng ý chia sẻ thông tin (repo duyệt tự động).
  2. Tạo token (Settings → Access Tokens, quyền "Read"), ghi vào .env: HF_TOKEN=hf_...
  3. python run_pipeline.py --stages benchmark,m1b
"""
from __future__ import annotations

import os
from pathlib import Path

from src.utils.common import get_logger
from src.utils.config import Paths

REPO_ID = "sonlam1102/vispamdetection2"
FILES = ["train.csv", "dev.csv", "test.csv"]


def run(cfg: dict) -> Path:
    P = Paths(cfg)
    log = get_logger("benchmark", P.logs)
    out = P.benchmark / cfg["m1b"]["benchmark_dir"]
    out.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError as e:
        raise SystemExit("Cần cài: pip install huggingface_hub") from e
    token = os.environ.get("HF_TOKEN") or None
    for name in FILES:
        if (out / name).exists():
            log.info("Đã có %s", name)
            continue
        try:
            hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=out, token=token)
        except GatedRepoError as e:
            raise SystemExit(
                f"Không có quyền tải {REPO_ID}. Đăng nhập HuggingFace, bấm đồng ý trên trang dataset, "
                "rồi đặt HF_TOKEN trong .env (xem docstring của src/collect/download_benchmark.py).") from e
        log.info("Đã tải %s", name)
    model = P.models / cfg["m1b"]["model_file"]
    if model.exists():      # mô hình cũ (nếu có) được train khi chưa có benchmark → xóa để M1b học lại
        model.unlink()
        log.info("Đã xóa %s để M1b huấn luyện lại trên benchmark", model.name)
    return out
