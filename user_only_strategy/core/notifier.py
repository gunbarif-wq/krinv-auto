from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests

from . import policy


@dataclass(frozen=True)
class NotifierConfig:
    log_path: Path
    telegram_token: str
    telegram_chat_id: str
    message_prefix: str = ""


class Notifier:
    def __init__(self, cfg: NotifierConfig, *, kst_tz, is_alerts_muted: Callable[[], bool]):
        self.log_path = cfg.log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.telegram_token = (cfg.telegram_token or "").strip()
        self.telegram_chat_id = (cfg.telegram_chat_id or "").strip()
        self.message_prefix = (cfg.message_prefix or "").strip()
        self._kst_tz = kst_tz
        self._is_alerts_muted = is_alerts_muted

    def send(self, text: str) -> None:
        now_local = datetime.now(self._kst_tz)
        if not policy.can_notify(now_local, alerts_muted=bool(self._is_alerts_muted())):
            return

        ts = now_local.strftime("%H:%M:%S")
        body = f"[{self.message_prefix}] {text}" if self.message_prefix else text
        line = f"[{ts}] {body}"

        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        if self.telegram_token and self.telegram_chat_id:
            sent = False
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                r = requests.post(url, json={"chat_id": self.telegram_chat_id, "text": line}, timeout=8)
                sent = bool(getattr(r, "ok", False))
            except Exception:
                sent = False
            if not sent:
                try:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    subprocess.run(
                        [
                            "curl.exe" if os.name == "nt" else "curl",
                            "-sS",
                            "-X",
                            "POST",
                            url,
                            "-d",
                            f"chat_id={self.telegram_chat_id}",
                            "--data-urlencode",
                            f"text={line}",
                        ],
                        check=False,
                        timeout=8,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

    def send_force(self, text: str) -> None:
        # Force-send regardless of time/mute policy. Use sparingly for boot/update confirmation.
        now_local = datetime.now(self._kst_tz)
        ts = now_local.strftime("%H:%M:%S")
        body = f"[{self.message_prefix}] {text}" if self.message_prefix else text
        line = f"[{ts}] {body}"

        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        if self.telegram_token and self.telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                requests.post(url, json={"chat_id": self.telegram_chat_id, "text": line}, timeout=8)
            except Exception:
                pass
