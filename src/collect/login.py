"""Mở Chrome có giao diện để đăng nhập Google một lần; phiên được lưu vào profile dùng cho user_data_dir.

Chạy: python -m src.collect.login [--profile .chrome_profile] [--chrome "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
Profile tạo bằng Chrome thật chỉ dùng được với Chrome thật → đặt cùng đường dẫn vào collect.playwright.executable_path.
"""
import argparse

MAPS = "https://www.google.com/maps?hl=vi"
SIGNIN = "https://accounts.google.com/ServiceLogin?hl=vi&continue=https%3A%2F%2Fwww.google.com%2Fmaps%3Fhl%3Dvi"


def check(ctx) -> bool:
    """Đã đăng nhập trên .google.com (tên miền Maps dùng) và Maps không ở chế độ hạn chế."""
    has_sid = any(c["name"] == "SID" and c["domain"] == ".google.com" for c in ctx.cookies())
    page = ctx.new_page()
    page.goto(MAPS)
    page.wait_for_timeout(5000)
    limited = "chế độ bị hạn chế" in page.inner_text("body")
    page.close()
    print(f"  cookie SID@.google.com: {has_sid} | Maps chế độ hạn chế: {limited}")
    return has_sid and not limited


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=".chrome_profile")
    ap.add_argument("--chrome", default=None, help="đường dẫn Chrome thật (Google hay chặn đăng nhập từ Chromium)")
    a = ap.parse_args()
    kw = {"executable_path": a.chrome} if a.chrome else {}
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            a.profile, headless=False, locale="vi-VN",
            args=["--disable-blink-features=AutomationControlled"], ignore_default_args=["--enable-automation"], **kw)
        page = ctx.new_page()
        page.goto(SIGNIN)
        while True:
            input("Đăng nhập (nên dùng tài khoản phụ), đợi tới khi Google Maps mở ra, rồi nhấn Enter... ")
            if check(ctx):
                break
            print("Chưa đăng nhập được trên www.google.com. Thử lại trong cửa sổ trình duyệt.")
            page.goto(SIGNIN)
        ctx.close()
    print(f"OK. Đã lưu profile tại {a.profile}. Đặt collect.playwright.user_data_dir: {a.profile} trong config.yaml.")


if __name__ == "__main__":
    main()
