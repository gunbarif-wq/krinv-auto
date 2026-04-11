from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from user_only_strategy.krx_symbol_names import DEFAULT_SYMBOL_NAME_FILE, refresh_symbol_name_map_from_krx


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh KRX symbol-name cache file")
    parser.add_argument("--out", default=DEFAULT_SYMBOL_NAME_FILE)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    count = refresh_symbol_name_map_from_krx(args.out, verbose=args.verbose)
    print(f"saved={count} path={args.out}")


if __name__ == "__main__":
    main()
