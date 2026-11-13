"""FastAPI bridge: parquet candles + XGB/LGBM signals + Alpaca *paper* trading -> React UI.

Run from the repository root::

    export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
    uvicorn api.server:app --reload --port 8000

Frontend: ``VITE_API_URL=http://localhost:8000 npm run dev`` (origin http://localhost:5173).

JSON shapes mirror ``frontend/src/types/index.ts`` 1-to-1 (camelCase keys):
    GET  /dashboard         -> Dashboard  (+ optional accountEquity / modelStatus / asOf)
    GET  /candles/{symbol}  -> Candle[]
    POST /orders/execute    -> OrderResult
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `uvicorn api.server:app` and `python api/server.py`
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from data_pipeline.errors import (  # noqa: E402
    DataContractError,
    RiskLimitError,
    SdfTradeError,
)
from data_pipeline.storage import read_bars  # noqa: E402
from risk_management.config import DEFAULT_RISK_CONFIG, RiskConfig  # noqa: E402
from risk_management.exits import compute_exit_levels  # noqa: E402
from risk_management.liquidity import compute_tradable  # noqa: E402
from strategies.features import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_SET_VERSION,
    build_features,
    feature_hash,
)
from strategies.models import LGBMStageModel, XGBStageModel, ensemble_proba  # noqa: E402
from strategies.signals import generate_signals  # noqa: E402

logger = logging.getLogger("api.server")

BARS_DIR = Path(os.getenv("BARS_DIR", ROOT / "data" / "raw" / "bars_1d"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "artifacts" / "models" / "stage1_run_001"))
RUN_ID = MODEL_DIR.name
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

MAX_ACCOUNT_RISK = 0.01  # hard cap: 1% of account equity lost if the stop-loss is hit
BARS_TTL_SECONDS = 60.0
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

app = FastAPI(title="auto-stock-trading API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(DataContractError)
async def _data_contract_handler(_: Request, exc: DataContractError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": f"Data unavailable: {exc}"})


@app.exception_handler(RiskLimitError)
async def _risk_handler(_: Request, exc: RiskLimitError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": f"Risk limit: {exc}"})


@app.exception_handler(SdfTradeError)
async def _sdf_handler(_: Request, exc: SdfTradeError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------- data access

_bars_lock = Lock()
_bars_cache: dict[str, Any] = {"ts": 0.0, "df": None}


def load_bars() -> pd.DataFrame:
    """Read every parquet part under ``data/raw/bars_1d`` (cached for a short TTL)."""
    with _bars_lock:
        now = time.monotonic()
        df = _bars_cache["df"]
        if df is not None and now - _bars_cache["ts"] < BARS_TTL_SECONDS:
            return df
        df = read_bars(BARS_DIR).drop_duplicates(subset=["symbol", "session_date"], keep="last")
        df["symbol"] = df["symbol"].astype(str)
        _bars_cache.update(ts=now, df=df)
        return df


def _date_str(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.strftime("%Y-%m-%d")


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------- models / signals

def load_models() -> tuple[dict[str, Any], list[str]]:
    """Validate the feature contract before loading artifacts from the selected run."""
    try:
        manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, ["Model manifest is missing or unreadable."]
    if not isinstance(manifest, dict) or (
        manifest.get("fs_version") != FEATURE_SET_VERSION
        or manifest.get("feature_names") != list(FEATURE_COLUMNS)
        or manifest.get("feature_hash_sha256") != feature_hash()
    ):
        return {}, ["Model manifest is incompatible with the inference feature contract."]

    models: dict[str, Any] = {}
    notes: list[str] = []
    for name, cls in (("xgb", XGBStageModel), ("lgbm", LGBMStageModel)):
        try:
            models[name] = cls.load(str(MODEL_DIR))
        except Exception as exc:
            logger.warning("Failed to load %s model: %s", name, exc)
            notes.append(f"{name}: model artifact unavailable.")
    return models, notes


def _score(feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep each model's valid probabilities aligned with the input rows."""
    models, notes = load_models()
    X = feats[list(FEATURE_COLUMNS)]
    scores = pd.DataFrame(np.nan, index=X.index, columns=["xgb", "lgbm"])
    for name, model in models.items():
        try:
            prediction = model.predict_proba(X)
            if not isinstance(prediction, pd.Series) or not prediction.index.equals(X.index):
                raise ValueError("Prediction index does not match the input")
            numeric = pd.to_numeric(prediction, errors="coerce")
            scores[name] = numeric.where(numeric.between(0.0, 1.0))
        except Exception as exc:
            logger.warning("%s inference failed: %s", name, exc)
            notes.append(f"{name}: inference failed.")
    return scores, notes


def _risk_limits(cfg: RiskConfig) -> dict[str, float | int]:
    return {
        "pThreshold": cfg.p_threshold,
        "topK": cfg.top_k,
        "minNames": cfg.min_names,
        "maxWeight": cfg.max_weight,
        "turnoverCap": cfg.turnover_cap,
        "minPrice": cfg.min_price,
        "minAdv20Dollar": cfg.min_adv20_dollar,
        "costBpsPerSide": cfg.cost_bps_per_side,
    }


def build_signals(
    bars: pd.DataFrame, cfg: RiskConfig = DEFAULT_RISK_CONFIG,
) -> tuple[list[dict], str, str | None, list[dict]]:
    """Rank symbols with two valid model scores and report blocked entries separately."""
    if bars.empty:
        return [], "data-unavailable", None, [{"reason": "No market data is available."}]

    work = bars.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    latest_date = work["session_date"].max()
    if pd.isna(latest_date):
        return [], "data-unavailable", None, [{"reason": "No market session date is available."}]
    as_of = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
    feats = build_features(work)
    tradable = compute_tradable(work, cfg)[["symbol", "session_date", "tradable"]]
    symbols = work[["symbol"]].drop_duplicates()
    latest = symbols.merge(feats[feats["session_date"] == latest_date], on="symbol", how="left")
    latest = latest.merge(tradable, on=["symbol", "session_date"], how="left")
    latest["tradable"] = latest["tradable"].eq(True)

    finite = np.isfinite(latest[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    ready = finite & (latest["atr14_pct"] > 0)
    blocked = [
        {"symbol": symbol, "reason": "Insufficient feature data for the latest session."}
        for symbol in latest.loc[~ready, "symbol"]
    ]
    if not ready.any():
        return [], "no-eligible-rows", as_of, blocked

    candidates = latest.loc[ready].copy()
    scores, notes = _score(candidates)
    complete = scores.notna().all(axis=1)
    for index, row in scores.loc[~complete].iterrows():
        missing = ", ".join(row.index[row.isna()])
        reason = " ".join(notes) if notes else f"Invalid prediction from {missing}."
        blocked.append({
            "symbol": candidates.at[index, "symbol"],
            "reason": reason,
            "pXgb": _finite(row["xgb"]),
            "pLgbm": _finite(row["lgbm"]),
        })

    valid_scores = scores.loc[complete]
    probs = candidates.loc[complete, ["session_date", "symbol", "atr14_pct", "tradable"]].copy()
    probs["p_xgb"] = valid_scores["xgb"]
    probs["p_lgbm"] = valid_scores["lgbm"]
    probs["p_long"] = ensemble_proba({"xgb": probs["p_xgb"], "lgbm": probs["p_lgbm"]})
    sig = generate_signals(probs, RUN_ID, cfg)
    status = "model-unavailable"
    if complete.any():
        status = "xgb+lgbm" if complete.all() else "partial-predictions"
    out = [
        {
            "symbol": str(r.symbol),
            "pXgb": float(r.p_xgb),
            "pLgbm": float(r.p_lgbm),
            "pLong": float(r.p_long),
            "rank": int(r.rank),
            "targetWeight": float(r.target_weight),
        }
        for r in sig.sort_values("rank").itertuples(index=False)
    ]
    return out, status, as_of, blocked


# --------------------------------------------------------------------------- Alpaca helpers


def _trading_client():
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise HTTPException(503, "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(503, "alpaca-py is not installed (pip install alpaca-py)") from exc
    return TradingClient(key, secret, paper=True)  # paper=True is hard-wired: never live


def _account_equity() -> float | None:
    """Paper account equity, or None if Alpaca is unreachable / unconfigured."""
    try:
        return float(_trading_client().get_account().equity)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alpaca equity unavailable: %s", exc)
        return None


def _latest_price(symbol: str, fallback: float) -> float:
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        client = StockHistoricalDataClient(key, secret)
        trade = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        price = _finite(trade[symbol].price)
        if price and price > 0:
            return price
    except Exception as exc:  # noqa: BLE001
        logger.info("latest trade unavailable for %s (%s); using last parquet close", symbol, exc)
    return fallback


# --------------------------------------------------------------------------- endpoints


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "barsDir": str(BARS_DIR), "modelDir": str(MODEL_DIR)}


@app.get("/dashboard")
def dashboard() -> dict[str, Any]:
    try:
        bars = load_bars()
    except DataContractError as exc:
        logger.warning("Dashboard market data unavailable: %s", exc)
        bars = pd.DataFrame(columns=["symbol"])
    signals, model_status, as_of, blocked = build_signals(bars)
    return {
        "symbols": sorted(bars["symbol"].unique().tolist()),
        "signals": signals,
        "blockedMlEntries": blocked,
        "risk": _risk_limits(DEFAULT_RISK_CONFIG),
        # optional extras (see frontend Dashboard type)
        "accountEquity": _account_equity(),
        "modelStatus": model_status,
        "asOf": as_of,
    }


@app.get("/candles/{symbol}")
def candles(symbol: str, limit: int = Query(500, ge=1, le=5000)) -> list[dict[str, Any]]:
    sym = symbol.upper()
    bars = load_bars()
    sub = bars[bars["symbol"] == sym].sort_values("session_date")
    if sub.empty:
        raise HTTPException(404, f"No bars for symbol {sym}")
    sub = sub.dropna(subset=["open", "high", "low", "close", "volume"]).tail(limit)
    return [
        {
            "time": t,
            "open": float(o),
            "high": float(h),
            "low": float(lo),
            "close": float(c),
            "volume": float(v),
        }
        for t, o, h, lo, c, v in zip(
            _date_str(sub["session_date"]), sub["open"], sub["high"], sub["low"], sub["close"], sub["volume"]
        )
    ]


class OrderRequest(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    notional_or_qty: float = Field(gt=0, description="Dollars if unit='notional', shares if unit='qty'")
    unit: Literal["notional", "qty"] = "notional"


class OrderResult(BaseModel):
    orderId: str
    clientOrderId: str
    symbol: str
    side: str
    qty: float
    status: str
    entryPrice: float | None = None
    stopPrice: float | None = None
    takeProfitPrice: float | None = None
    riskDollars: float | None = None
    riskPctOfEquity: float | None = None
    cappedBy: list[str] = []
    message: str = ""


@app.post("/orders/execute", response_model=OrderResult)
def execute_order(req: OrderRequest) -> OrderResult:
    """Submit a paper order.

    * ``buy``: market bracket order (ATR stop-loss + take-profit from ``compute_exit_levels``),
      whole shares, GTC. Size is capped so that a stop-out loses <= 1% of account equity, and by
      the per-name ``max_weight`` and buying power. The 10-session time exit is NOT expressible in a
      bracket order and must be handled by a separate job.
    * ``sell``: flattens the whole position (open orders for the symbol are cancelled first).
    """
    symbol = req.symbol.strip().upper()
    if not SYMBOL_RE.match(symbol):
        raise HTTPException(422, f"Invalid symbol {req.symbol!r}")

    try:
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
        from alpaca.trading.requests import (
            ClosePositionRequest,
            GetOrdersRequest,
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(503, "alpaca-py is not installed (pip install alpaca-py)") from exc

    client = _trading_client()
    try:
        account = client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        if getattr(account, "trading_blocked", False):
            raise HTTPException(409, "Alpaca account is trading-blocked")
        positions = {p.symbol: float(p.qty) for p in client.get_all_positions()}

        # ------------------------------------------------------------------ sell = flatten
        if req.side == "sell":
            held = positions.get(symbol, 0.0)
            if held <= 0:
                raise HTTPException(409, f"No long position in {symbol} to sell")
            open_orders = client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], nested=True)
            )
            for o in open_orders:
                client.cancel_order_by_id(str(o.id))
            order = client.close_position(symbol, close_options=ClosePositionRequest(percentage="100"))
            return OrderResult(
                orderId=str(order.id),
                clientOrderId=str(order.client_order_id),
                symbol=symbol,
                side="sell",
                qty=held,
                status=str(getattr(order.status, "value", order.status)),
                message=f"Closing {held:g} {symbol} (cancelled {len(open_orders)} open order(s))",
            )

        # ------------------------------------------------------------------ buy = bracket
        bars = load_bars()
        sub = bars[bars["symbol"] == symbol].sort_values("session_date")
        if sub.empty:
            raise HTTPException(404, f"No bars for symbol {symbol}; cannot size a stop-loss")
        atr_pct = _finite(build_features(sub).iloc[-1]["atr14_pct"])
        last_close = float(sub.iloc[-1]["close"])
        if not atr_pct or atr_pct <= 0:
            raise HTTPException(422, f"Not enough history to compute ATR for {symbol}")

        price = _latest_price(symbol, last_close)
        levels = compute_exit_levels(price, atr_pct * last_close)
        stop_px = round(levels.stop_px, 2)
        take_px = round(levels.target_px, 2)
        risk_per_share = price - stop_px
        if stop_px <= 0 or risk_per_share < 0.02 or take_px <= price + 0.01:
            raise HTTPException(422, f"Degenerate stop/target for {symbol} (stop={stop_px}, target={take_px})")

        requested = math.floor(req.notional_or_qty / price) if req.unit == "notional" else math.floor(req.notional_or_qty)
        caps = {
            "risk_1pct": math.floor(MAX_ACCOUNT_RISK * equity / risk_per_share),
            "max_weight": math.floor(DEFAULT_RISK_CONFIG.max_weight * equity / price - positions.get(symbol, 0.0)),
            "buying_power": math.floor(buying_power / price),
        }
        qty = min(requested, *caps.values())
        capped_by = [name for name, cap in caps.items() if cap < requested]
        if qty < 1:
            raise HTTPException(
                422,
                f"Order size rounds to 0 shares (requested={requested}, caps={caps}); "
                f"1% risk budget = ${MAX_ACCOUNT_RISK * equity:,.2f}, stop distance = ${risk_per_share:.2f}/sh",
            )

        order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_px),
                stop_loss=StopLossRequest(stop_price=stop_px),
                client_order_id=f"ui-{uuid.uuid4().hex[:20]}",
            )
        )
        risk_dollars = qty * risk_per_share
        return OrderResult(
            orderId=str(order.id),
            clientOrderId=str(order.client_order_id),
            symbol=symbol,
            side="buy",
            qty=float(qty),
            status=str(getattr(order.status, "value", order.status)),
            entryPrice=round(price, 2),
            stopPrice=stop_px,
            takeProfitPrice=take_px,
            riskDollars=round(risk_dollars, 2),
            riskPctOfEquity=round(risk_dollars / equity, 5),
            cappedBy=capped_by,
            message=f"Bracket BUY {qty} {symbol} @ ~{price:.2f}, SL {stop_px}, TP {take_px}",
        )
    except HTTPException:
        raise
    except APIError as exc:
        logger.warning("Alpaca rejected order for %s: %s", symbol, exc)
        raise HTTPException(502, f"Alpaca error: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api.server:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=False)
