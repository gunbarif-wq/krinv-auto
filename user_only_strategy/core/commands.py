from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Callable, Optional


def parse_control_command(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None
    key = "".join(raw.split())

    if key == "매매중지":
        return "pause_trading"
    if key == "매매재개":
        return "resume_trading"
    if key == "종목선정중지":
        return "cancel_select"
    if key == "종목선정":
        return "select"
    if key == "보유":
        return "holdings"
    if "모니터" in key:
        return "status"
    if key == "알림중지":
        return "mute_alerts"
    if key == "알림재개":
        return "unmute_alerts"
    if key in {"봇재부팅", "봇업데이트", "업데이트", "update", "restart"}:
        return "self_update"
    return None


def run_self_update_and_exec(
    *,
    repo_root: str,
    notifier_send: Callable[[str], None],
    persist_state: Callable[[], None],
    last_self_update_at_text: str,
    kst_tz,
) -> str:
    now_text = datetime.now(kst_tz).strftime("%Y-%m-%d %H:%M:%S")
    try:
        if last_self_update_at_text:
            last_dt = datetime.fromisoformat(last_self_update_at_text.replace(" ", "T"))
            if (datetime.now(kst_tz) - last_dt).total_seconds() < 90:
                notifier_send("업데이트 무시 | 직전 재기동 직후")
                return last_self_update_at_text
    except Exception:
        pass

    last_self_update_at_text = now_text
    notifier_send("업데이트 시작 | git pull --ff-only")
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "pull", "--ff-only", "origin", "master"],
            text=True,
            capture_output=True,
            timeout=90,
        )
        out_lines = (proc.stdout or "").strip().splitlines()
        err_lines = (proc.stderr or "").strip().splitlines()
        if proc.returncode != 0:
            detail = err_lines[-1] if err_lines else "git pull failed"
            notifier_send(f"업데이트 실패 | {detail[:180]}")
            return last_self_update_at_text
        summary = out_lines[-1] if out_lines else "ok"
        notifier_send(f"업데이트 완료 | {summary[:180]}")
    except Exception as exc:
        notifier_send(f"업데이트 예외 | {type(exc).__name__} | {str(exc)[:160]}")
        return last_self_update_at_text

    try:
        persist_state()
        notifier_send("재기동 시작")
        time.sleep(2.0)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as exc:
        notifier_send(f"재기동 실패 | {type(exc).__name__} | {str(exc)[:160]}")
    return last_self_update_at_text

