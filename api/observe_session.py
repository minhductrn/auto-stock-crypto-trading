"""Publish a read-only dashboard snapshot for a completed market session."""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from api import server
from data_pipeline.storage import read_bars
from data_pipeline.types import DateRange
from data_pipeline.validation import validate_bars
from data_pipeline.yfinance_provider import YFinanceProvider


def publish(session, bars_dir, output, *, fetch=False, chart_symbols=()):
    calendar = xcals.get_calendar("XNYS")
    now = pd.Timestamp.now(tz="UTC")
    target = pd.Timestamp(session)
    if not calendar.is_session(target) or calendar.session_close(target) > now:
        raise ValueError("Select a completed NYSE session.")

    history = read_bars(bars_dir)
    history = history.loc[pd.to_datetime(history.session_date) <= target]
    history = history.drop_duplicates(["symbol", "session_date"], keep="last")
    symbols = sorted(history.symbol.unique().tolist())
    if not symbols:
        raise ValueError("No local universe is available for this session.")

    bars = history.loc[pd.to_datetime(history.session_date) == target]
    source = "yfinance" if fetch else "Local daily bars"
    error = None
    try:
        if fetch:
            bars = YFinanceProvider().fetch_bars_1d(symbols, DateRange(session, session))
        if bars.empty or not pd.to_datetime(bars.session_date).eq(target).all():
            raise ValueError("The selected session is missing from the data.")
        if not np.isfinite(bars[["open", "high", "low", "close", "volume"]]).all().all():
            raise ValueError("The selected session contains non-finite prices or volume.")
        valid, _ = validate_bars(bars, max_quarantine_share=0)
        missing = sorted(set(symbols) - set(valid.symbol))
        if missing:
            raise ValueError("Missing session bars: " + ", ".join(missing))
    except (ValueError, server.DataContractError) as exc:
        error = str(exc)
    except server.SdfTradeError:
        error = "The data provider could not supply the selected session."

    today = pd.Timestamp(now.tz_convert("America/New_York").date())
    latest = calendar.date_to_session(today, direction="previous")
    if calendar.session_close(latest) > now:
        latest = calendar.previous_session(latest)
    status = "Incomplete" if error else "Current" if target == latest else "Stale"
    if error:
        signals, model_status, blocked = [], "data-unavailable", [{"reason": error}]
        candles = {}
    else:
        combined = pd.concat([history.loc[pd.to_datetime(history.session_date) < target], valid])
        signals, model_status, _, blocked = server.build_signals(combined)
        candles = {}
        for symbol, rows in combined.sort_values("session_date").groupby("symbol"):
            rows = rows.tail(500)
            rows = rows.assign(time=server._date_str(rows.session_date))
            candles[symbol] = rows[["time", "open", "high", "low", "close", "volume"]].to_dict("records")

    references = []
    for symbol in sorted(set(chart_symbols) - set(candles)):
        rows = YFinanceProvider().fetch_bars_1d(
            [symbol], DateRange(session - timedelta(days=750), session),
        )
        rows = rows.loc[pd.to_datetime(rows.session_date) <= target]
        rows, _ = validate_bars(rows, max_quarantine_share=0)
        if rows.empty:
            raise ValueError(f"No chart data returned for {symbol}.")
        rows = rows.sort_values("session_date").tail(500)
        rows = rows.assign(time=server._date_str(rows.session_date))
        candles[symbol] = rows[["time", "open", "high", "low", "close", "volume"]].to_dict("records")
        references.append(symbol)

    payload = {
        "dashboard": {
            "symbols": sorted(candles), "referenceSymbols": references, "signals": signals, "blockedMlEntries": blocked,
            "risk": server._risk_limits(server.DEFAULT_RISK_CONFIG), "modelStatus": model_status,
            "asOf": session.isoformat(), "source": source, "dataStatus": status,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "nextSession": calendar.next_session(target).date().isoformat(),
            "staleAfter": calendar.session_close(calendar.next_session(target)).isoformat(),
        },
        "candles": candles,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, allow_nan=False)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as temp:
        temp.write(content)
    os.replace(temp.name, output)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=date.fromisoformat)
    parser.add_argument("--bars", type=Path, default=server.BARS_DIR)
    parser.add_argument("--output", type=Path, default=Path("frontend/public/observation.json"))
    parser.add_argument("--fetch", action="store_true", help="Fetch the selected day from yfinance.")
    parser.add_argument("--chart-symbol", action="append", default=[], type=str.upper,
                        help="Add a Yahoo Finance ticker for chart display only. Repeat as needed.")
    args = parser.parse_args()
    try:
        status = publish(args.session, args.bars, args.output, fetch=args.fetch, chart_symbols=args.chart_symbol)
    except (ValueError, server.SdfTradeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"{args.session}: {status} · {args.output}")
    return int(status == "Incomplete")


if __name__ == "__main__":
    raise SystemExit(main())
