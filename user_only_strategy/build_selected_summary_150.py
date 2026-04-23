import argparse
import json
from pathlib import Path

from user_only_strategy.monday_custom_timing_bot import (
    fetch_candidate_universe,
    get_access_token,
    load_dotenv,
)


def _read_symbols_from_summary(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    out: set[str] = set()
    for key in ("selected_symbols", "symbols"):
        rows = data.get(key, [])
        if isinstance(rows, list):
            for s in rows:
                t = str(s).strip()
                if t.isdigit():
                    out.add(t.zfill(6))
    # Some summaries store per_symbol keys only.
    per_symbol = data.get("per_symbol", {})
    if isinstance(per_symbol, dict):
        for k in per_symbol.keys():
            t = str(k).strip()
            if t.isdigit():
                out.add(t.zfill(6))
    return out


def main() -> None:
    load_dotenv(".env")

    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--app-key", required=True)
    p.add_argument("--app-secret", required=True)
    p.add_argument("--exclude-summary", required=True, help="Existing summary.json whose symbols should be excluded")
    p.add_argument("--out-summary", required=True, help="Path to write the new summary.json")
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--pool-size", type=int, default=800)
    args = p.parse_args()

    exclude_path = Path(args.exclude_summary)
    if not exclude_path.is_absolute():
        exclude_path = (Path.cwd() / exclude_path).resolve()
    excluded = _read_symbols_from_summary(exclude_path)

    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    universe = fetch_candidate_universe(
        base_url=args.base_url,
        token=token,
        app_key=args.app_key,
        app_secret=args.app_secret,
        max_universe=max(200, int(args.pool_size)),
        extra_symbols=[],
    )

    selected: list[str] = []
    seen: set[str] = set()
    for symbol, _name in universe:
        s = str(symbol).strip()
        if not s.isdigit():
            continue
        s = s.zfill(6)
        if s in excluded or s in seen:
            continue
        seen.add(s)
        selected.append(s)
        if len(selected) >= int(args.count):
            break

    out_path = Path(args.out_summary)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_symbols": selected,
        "excluded_count": len(excluded),
        "selected_count": len(selected),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} symbols to {out_path}")


if __name__ == "__main__":
    main()

