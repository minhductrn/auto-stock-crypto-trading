"""CLI: fetch daily US-equity candles via yfinance, validate, and store them.

Run from the repository root::

    python -m data_pipeline.fetch_candles --symbols AAPL MSFT SPY \
        --start 2010-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Sequence

from .types import DateRange
from .storage import write_quarantine, write_raw_bars
from .validation import validate_bars
from .yfinance_provider import YFinanceProvider

logger = logging.getLogger("data_pipeline.fetch_candles")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch US-equity daily candles (yfinance).")
    p.add_argument("--symbols", nargs="+", required=True, help="Tickers, e.g. AAPL MSFT SPY")
    p.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument("--out", type=Path, default=Path("data/raw/bars_1d"))
    p.add_argument("--quarantine", type=Path, default=Path("data/quarantine/bars_1d"))
    p.add_argument("--max-quarantine-share", type=float, default=0.02)
    return p.parse_args(argv)


def run(symbols: Sequence[str], rng: DateRange, out: Path, quarantine: Path,
        max_quarantine_share: float = 0.02) -> int:
    """Fetch, validate and persist bars. Returns the number of valid rows written."""
    provider = YFinanceProvider()
    bars = provider.fetch_bars_1d(symbols, rng)
    logger.info("Fetched %d rows for %d symbols", len(bars), len(set(symbols)))
    valid, bad = validate_bars(bars, max_quarantine_share=max_quarantine_share)
    write_quarantine(bad, quarantine)
    if not valid.empty:
        write_raw_bars(valid.drop(columns=["dq_flags"]), out)
    logger.info("Wrote %d valid rows to %s; quarantined %d", len(valid), out, len(bad))
    return len(valid)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    run(args.symbols, DateRange(args.start, args.end), args.out, args.quarantine,
        args.max_quarantine_share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
