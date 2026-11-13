"""``sdf_fs_v1`` feature matrix: price/volume, SDF geometry, cross-sectional and market context.

Implements the 82-column contract of ``docs/STAGE1_BASELINE_TRADING.md`` (sections 4 and 5)
in fixed column order:

* Group P (16): price / volume / momentum / volatility features.
* Group S (53): signed-distance-field geometry for the five frozen corridors
  (``don20``, ``don55``, ``bb20``, ``va60``, ``swg``), single lines (VWAP20, PDH, PDL, POC60),
  composites and lagged memory.
* Group X (10): same-date cross-sectional ranks over the tradable universe plus market
  context (SPY, VIX).
* Group C (3): sector id and calendar fields.

Causality. Every feature at ``(symbol, t)`` uses only bars with ``session_date <= t``.
Rolling windows are trailing. Donchian and value-area walls are shifted by one session so
they exclude ``t``. A swing at ``j`` is used only from session ``j + 3``. Cross-sectional
ranks use only same-date information. Market context joins on ``session_date`` and looks
backward only.

Adjustment invariance. Every feature except ``dollar_vol_20_log`` and the tradable mask is a
ratio, log-return or ATR-normalised distance, so it is invariant to a constant positive
rescaling of one symbol's price history. Those two use ``close_raw`` / ``volume_raw`` when
the bars carry them and fall back to ``close`` / ``volume`` otherwise.

Market context sources (all optional, resolved in this order):

* SPY: ``context["SPY"]``, else rows with ``symbol == "SPY"`` in ``bars``, else a causal
  equal-weight index built from the universe's own bars (a documented proxy).
* VIX: ``context["^VIX"]``, else rows with ``symbol == "^VIX"`` in ``bars``, else a
  realised-volatility proxy ``100 * sqrt(252) * mkt_rv_20``. ``^VIX`` rows are never
  emitted as stock rows.

dtypes. All feature columns are ``float32`` except the four categorical columns
(``sdf_nearest_k``, ``sector_id``, ``dow``, ``month``), which are ``int8`` per spec section 5.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

from data_pipeline.errors import DataContractError

FEATURE_SET_VERSION = "sdf_fs_v1"
KEYS: tuple[str, ...] = ("symbol", "session_date")

# --------------------------------------------------------------------------- column contract

_GROUP_P: tuple[str, ...] = (
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "rv_20", "rv_60", "atr14_pct", "gap_pct", "range_pct", "close_loc", "vol_z20",
    "dollar_vol_20_log", "rsi14", "macd_hist_atr", "beta_60",
)
_SDF_KEYS: tuple[str, ...] = ("don20", "don55", "bb20", "va60", "swg")
_LAG_KEYS: tuple[str, ...] = ("don20", "bb20", "va60", "swg")
_LAGS: tuple[int, ...] = (1, 2, 3, 5, 10)
_GROUP_S_CORRIDOR: tuple[str, ...] = tuple(
    f"sdf_{k}{suf}" for k in _SDF_KEYS for suf in ("", "_pos", "_width", "_grad5", "_curv")
)
_GROUP_S_LINES: tuple[str, ...] = (
    "sdf_vwap20", "sdf_pdh", "sdf_pdl", "sdf_poc60",
    "sdf_min_abs", "sdf_n_up_breach", "sdf_n_dn_breach", "sdf_nearest_k",
)
_GROUP_S_LAGS: tuple[str, ...] = tuple(f"sdf_{k}_l{lag}" for k in _LAG_KEYS for lag in _LAGS)
_GROUP_S: tuple[str, ...] = _GROUP_S_CORRIDOR + _GROUP_S_LINES + _GROUP_S_LAGS
_GROUP_X: tuple[str, ...] = (
    "cs_rank_ret_5", "cs_rank_ret_20", "cs_rank_sdf_don20", "cs_rank_vol_z20",
    "mkt_ret_5", "mkt_ret_20", "mkt_rv_20", "mkt_sdf_don55", "vix_close", "vix_chg_5",
)
_GROUP_C: tuple[str, ...] = ("sector_id", "dow", "month")

FEATURE_COLUMNS: tuple[str, ...] = _GROUP_P + _GROUP_S + _GROUP_X + _GROUP_C
assert len(FEATURE_COLUMNS) == 82 and len(set(FEATURE_COLUMNS)) == 82

_INT8_COLUMNS: frozenset[str] = frozenset({"sdf_nearest_k", "sector_id", "dow", "month"})

# Ablation column sets (spec 11.4).
SDF_DEPENDENT_X: tuple[str, ...] = ("cs_rank_sdf_don20", "mkt_sdf_don55")
ABLATION_B_NO_SDF: tuple[str, ...] = (
    _GROUP_P + tuple(c for c in _GROUP_X if c not in SDF_DEPENDENT_X) + _GROUP_C
)  # 27 columns
ABLATION_C_SDF_ONLY: tuple[str, ...] = _GROUP_S + _GROUP_C  # 56 columns

_REQUIRED = {"symbol", "session_date", "open", "high", "low", "close", "volume"}

# SDF / tradable constants (spec sections 3, 4, 12).
_CLIP = 10.0
_VA_WINDOW, _VA_BINS, _VA_COVERAGE = 60, 40, 0.70
_SWING_K, _SWING_LOOKBACK = 3, 120
_MIN_PRICE, _MIN_ADV20 = 5.0, 20_000_000.0
_VA_CHUNK = 2048


def feature_hash(columns: tuple[str, ...] = FEATURE_COLUMNS) -> str:
    """SHA-256 of the ordered column list (column order is API)."""
    return hashlib.sha256(",".join(columns).encode()).hexdigest()


# --------------------------------------------------------------------------- small kernels


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """Wilder ATR, seeded with the simple mean of the first ``window`` true ranges.

    Causal: ``atr[i]`` depends only on bars ``0..i``. Floored at ``1e-8 * close``.
    """
    n = len(close)
    atr = np.full(n, np.nan)
    if n < window:
        return atr
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr[window - 1] = tr[:window].mean()
    for i in range(window, n):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return np.maximum(atr, 1e-8 * close)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """``out[t] = x[t-k]`` with NaN fill (strictly backward)."""
    out = np.full(len(x), np.nan)
    if 0 < k < len(x):
        out[k:] = x[:-k]
    elif k == 0:
        out[:] = x
    return out


def _rmin(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w).min().to_numpy()


def _rmax(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w).max().to_numpy()


def _clean(a: np.ndarray) -> np.ndarray:
    """float64 copy with +-inf mapped to NaN."""
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, np.nan)


def _wilder_smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Wilder smoothing seeded by the SMA of the first ``window`` values (causal)."""
    out = np.full(len(x), np.nan)
    if len(x) < window:
        return out
    seed = x[:window].mean()
    out[window - 1] = seed
    if len(x) > window:
        a = 1.0 / window
        out[window:] = lfilter([a], [1.0, -(1.0 - a)], x[window:], zi=[(1.0 - a) * seed])[0]
    return out


def _rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    """Wilder RSI(14), range 0-100 (100 if no losses, 50 if flat)."""
    n = len(close)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    d = np.diff(close)
    ag = _wilder_smooth(np.maximum(d, 0.0), window)
    al = _wilder_smooth(np.maximum(-d, 0.0), window)
    with np.errstate(all="ignore"):
        rsi = np.where(al > 0, 100.0 - 100.0 / (1.0 + ag / al), np.where(ag > 0, 100.0, 50.0))
    out[1:] = np.where(np.isnan(ag) | np.isnan(al), np.nan, rsi)
    return out


# --------------------------------------------------------------------------- SDF geometry


def _corridor_block(c: np.ndarray, atr: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    """Signed distance field ``phi``, ``pos`` and ``width`` of one corridor (spec 4.1).

    ``phi = max(L' - c, c - U') / ATR`` clipped to [-10, 10]; ``pos`` clipped to [-1, 2] and NaN
    when the corridor is degenerate; ``width = (U' - L') / ATR``. Uses walls known at ``t``.
    """
    with np.errstate(all="ignore"):
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)
        phi = np.clip(np.maximum(lo - c, c - hi) / atr, -_CLIP, _CLIP)
        span = hi - lo
        pos = np.where(span >= 1e-8 * c, (c - lo) / span, np.nan)
        pos = np.clip(pos, -1.0, 2.0)
        width = span / atr
    return phi, pos, width


def _line_sdf(c: np.ndarray, line: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """Plain signed ATR distance to a single line, clipped to [-10, 10]."""
    with np.errstate(all="ignore"):
        return np.clip((c - line) / atr, -_CLIP, _CLIP)


def _donchian(h: np.ndarray, l: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Walls over ``[t-window .. t-1]`` (excludes ``t``)."""
    return _rmin(_shift(l, 1), window), _rmax(_shift(h, 1), window)


def _bollinger(c: np.ndarray, window: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """SMA +- k * population std over ``[t-window+1 .. t]`` (includes ``t``)."""
    s = pd.Series(c)
    mid = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (mid - k * sd).to_numpy(), (mid + k * sd).to_numpy()


def _va_chunk(H: np.ndarray, Lw: np.ndarray, T: np.ndarray, V: np.ndarray, bins: int, coverage: float):
    """Vectorised value-area algorithm over a block of windows (spec 4.2)."""
    m = H.shape[0]
    valid = (np.isfinite(H).all(1) & np.isfinite(Lw).all(1) & np.isfinite(T).all(1) & np.isfinite(V).all(1))
    with np.errstate(all="ignore"):
        pmax, pmin = H.max(1), Lw.min(1)
        total = V.sum(1)
        ok = valid & (pmax > pmin) & (total > 0)
        pmin = np.where(ok, pmin, 0.0)
        rng = np.where(ok, pmax - pmin, 1.0)
        T0 = np.where(np.isfinite(T), T, 0.0)
        V0 = np.where(np.isfinite(V), V, 0.0)
        idx = np.floor((T0 - pmin[:, None]) / rng[:, None] * bins).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)  # upper edge goes to the top bin
    flat = idx + bins * np.arange(m)[:, None]
    vol = np.bincount(flat.ravel(), weights=V0.ravel(), minlength=m * bins).reshape(m, bins)

    poc_bin = vol.argmax(1)  # ties -> lowest bin
    rows = np.arange(m)
    thr = coverage * vol.sum(1)
    padded = np.full((m, bins + 2), -1.0)  # sentinels: an exhausted side is never preferred
    padded[:, 1:-1] = vol
    lo, hi = poc_bin.copy(), poc_bin.copy()
    acc = vol[rows, poc_bin].copy()
    for _ in range(bins - 1):
        active = ok & (acc < thr)
        if not active.any():
            break
        up = padded[rows, hi + 2]
        dn = padded[rows, lo]
        take_up = up >= dn  # ties -> upper bin
        hi = np.where(active & take_up, hi + 1, hi)
        lo = np.where(active & ~take_up, lo - 1, lo)
        acc = np.where(active, acc + np.where(take_up, up, dn), acc)

    binw = rng / bins
    val = np.where(ok, pmin + lo * binw, np.nan)
    vah = np.where(ok, pmin + (hi + 1) * binw, np.nan)
    poc = np.where(ok, pmin + (poc_bin + 0.5) * binw, np.nan)
    return val, vah, poc


def _value_area(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray,
                window: int = _VA_WINDOW, bins: int = _VA_BINS, coverage: float = _VA_COVERAGE):
    """Value Area Low/High and POC from bars ``[t-window .. t-1]`` (excludes ``t``)."""
    n = len(c)
    val, vah, poc = (np.full(n, np.nan) for _ in range(3))
    if n <= window:
        return val, vah, poc
    tp = (h + l + c) / 3.0
    nwin = n - window  # window k ends at k+window-1 and is used at t = k+window
    H = sliding_window_view(h, window)[:nwin]
    Lw = sliding_window_view(l, window)[:nwin]
    T = sliding_window_view(tp, window)[:nwin]
    V = sliding_window_view(v, window)[:nwin]
    for s in range(0, nwin, _VA_CHUNK):
        e = min(nwin, s + _VA_CHUNK)
        a, b, p = _va_chunk(H[s:e], Lw[s:e], T[s:e], V[s:e], bins, coverage)
        val[window + s: window + e] = a
        vah[window + s: window + e] = b
        poc[window + s: window + e] = p
    return val, vah, poc


def _swing_bounds(h: np.ndarray, l: np.ndarray, k: int = _SWING_K,
                  lookback: int = _SWING_LOOKBACK) -> tuple[np.ndarray, np.ndarray]:
    """Lowest confirmed swing low / highest confirmed swing high of the last ``lookback`` sessions.

    A swing at ``j`` (strict extreme versus ``j +- 1..k``) is confirmed at ``j + k`` and used
    from then on, with ``j`` in ``[t - lookback + 1, t - k]``. Uses data ``<= t`` only.
    """
    n = len(h)
    is_sh = np.ones(n, dtype=bool)
    is_sl = np.ones(n, dtype=bool)
    with np.errstate(all="ignore"):
        for i in range(1, k + 1):
            h_left, h_right = np.full(n, np.nan), np.full(n, np.nan)
            l_left, l_right = np.full(n, np.nan), np.full(n, np.nan)
            if n > i:
                h_left[i:], h_right[:-i] = h[:-i], h[i:]
                l_left[i:], l_right[:-i] = l[:-i], l[i:]
            is_sh &= (h > h_left) & (h > h_right)  # NaN comparisons are False
            is_sl &= (l < l_left) & (l < l_right)
    known_high = _shift(np.where(is_sh, h, np.nan), k)  # swing at j visible at j + k
    known_low = _shift(np.where(is_sl, l, np.nan), k)
    w = lookback - k
    upper = pd.Series(known_high).rolling(w, min_periods=1).max().to_numpy()
    lower = pd.Series(known_low).rolling(w, min_periods=1).min().to_numpy()
    return lower, upper


# --------------------------------------------------------------------------- market context


def _equal_weight_proxy(ordered: pd.DataFrame) -> pd.DataFrame:
    """Causal equal-weight OHLC index from the universe's own bars (SPY fallback).

    Each date's index return is the mean over symbols of ``x_t / close_{t-1}``, so only the
    bars of dates ``<= t`` are involved and composition changes do not create jumps.
    """
    prev_close = ordered.groupby("symbol")["close"].shift(1).astype(float)
    ratios = pd.DataFrame({"session_date": ordered["session_date"].to_numpy()})
    for k in ("open", "high", "low", "close"):
        ratios[k] = (ordered[k].astype(float) / prev_close).to_numpy()
    with np.errstate(all="ignore"):
        m = ratios.groupby("session_date", sort=True)[["open", "high", "low", "close"]].mean()
    m = m.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    close = 100.0 * m["close"].cumprod()
    prev = close.shift(1)
    prev.iloc[0] = close.iloc[0]
    out = pd.DataFrame({
        "session_date": m.index,
        "open": (prev * m["open"]).to_numpy(),
        "high": (prev * m["high"]).to_numpy(),
        "low": (prev * m["low"]).to_numpy(),
        "close": close.to_numpy(),
    })
    return out


def _market_context(ordered: pd.DataFrame, vix_rows: pd.DataFrame, context: dict | None) -> pd.DataFrame:
    """Per-date market features (indexed by ``session_date``), computed from data ``<= t``."""
    context = context or {}
    spy = context.get("SPY")
    if spy is None:
        sub = ordered[ordered["symbol"] == "SPY"]
        spy = sub if len(sub) else _equal_weight_proxy(ordered)
    spy = (spy.sort_values("session_date").drop_duplicates("session_date", keep="last")
           .reset_index(drop=True))
    idx = pd.Index(spy["session_date"])
    h, l, c = (spy[k].to_numpy(dtype=float) for k in ("high", "low", "close"))

    with np.errstate(all="ignore"):
        logc = pd.Series(np.log(c))
        ret1 = logc.diff()
        atr = wilder_atr(h, l, c)
        lo55, up55 = _donchian(h, l, 55)
        phi55, _, _ = _corridor_block(c, atr, lo55, up55)
        mkt_rv_20 = ret1.rolling(20).std(ddof=1).to_numpy()

    vix_src = context.get("^VIX")
    if vix_src is None and len(vix_rows):
        vix_src = vix_rows
    if vix_src is not None:
        vs = (vix_src.sort_values("session_date").drop_duplicates("session_date", keep="last"))
        vclose = pd.Series(vs["close"].to_numpy(dtype=float), index=pd.Index(vs["session_date"]))
        with np.errstate(all="ignore"):
            vchg = np.log(vclose / vclose.shift(5))
        vix_close = vclose.reindex(idx).to_numpy()
        vix_chg_5 = vchg.reindex(idx).to_numpy()
    else:  # realised-volatility proxy in VIX points (annualised %, same causal window as mkt_rv_20)
        vix_close = 100.0 * np.sqrt(252.0) * mkt_rv_20
        with np.errstate(all="ignore"):
            vix_chg_5 = np.log(vix_close / _shift(vix_close, 5))

    return pd.DataFrame(
        {
            "mkt_ret_1": ret1.to_numpy(),
            "mkt_ret_5": (logc - logc.shift(5)).to_numpy(),
            "mkt_ret_20": (logc - logc.shift(20)).to_numpy(),
            "mkt_rv_20": mkt_rv_20,
            "mkt_sdf_don55": phi55,
            "vix_close": vix_close,
            "vix_chg_5": vix_chg_5,
        },
        index=idx,
    )


# --------------------------------------------------------------------------- per-symbol features


def _features_one(g: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    g = g.reset_index(drop=True)
    n = len(g)
    o, h, l, c, v = (g[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    c_raw = g["close_raw"].to_numpy(dtype=float) if "close_raw" in g.columns else c
    v_raw = g["volume_raw"].to_numpy(dtype=float) if "volume_raw" in g.columns else v
    ctx = mk.reindex(pd.Index(g["session_date"]))

    f: dict[str, np.ndarray] = {}
    with np.errstate(all="ignore"):
        cs = pd.Series(c)
        logc = pd.Series(np.log(c))
        ret1 = logc.diff()
        atr = wilder_atr(h, l, c)

        # ---- Group P
        for k in (1, 5, 10, 20, 60):
            f[f"ret_{k}"] = (logc - logc.shift(k)).to_numpy()
        f["rv_20"] = ret1.rolling(20).std(ddof=1).to_numpy()
        f["rv_60"] = ret1.rolling(60).std(ddof=1).to_numpy()
        f["atr14_pct"] = atr / c
        f["gap_pct"] = o / _shift(c, 1) - 1.0
        f["range_pct"] = (h - l) / c
        f["close_loc"] = np.where(h > l, (c - l) / (h - l), 0.5)
        lv = pd.Series(np.log1p(v))
        sd = lv.rolling(20).std(ddof=1)
        z = (lv - lv.rolling(20).mean()) / sd
        f["vol_z20"] = z.where(sd > 0, 0.0).where(sd.notna()).to_numpy()
        dollar = pd.Series(c_raw * v_raw)
        adv20 = dollar.rolling(20).mean().to_numpy()
        f["dollar_vol_20_log"] = np.log1p(adv20)
        f["rsi14"] = _rsi(c)
        macd = cs.ewm(span=12, adjust=False).mean() - cs.ewm(span=26, adjust=False).mean()
        hist = (macd - macd.ewm(span=9, adjust=False).mean()).to_numpy().copy()
        hist[:33] = np.nan  # EMA warm-up (26 + 9 - 2)
        f["macd_hist_atr"] = hist / atr
        mret = pd.Series(ctx["mkt_ret_1"].to_numpy())
        mvar = mret.rolling(60).var()
        beta = ret1.rolling(60).cov(mret) / mvar
        f["beta_60"] = beta.where(mvar > 0).to_numpy()

        # ---- Group S: corridors
        lo20, up20 = _donchian(h, l, 20)
        lo55, up55 = _donchian(h, l, 55)
        lo_bb, up_bb = _bollinger(c)
        va_lo, va_hi, poc = _value_area(h, l, c, v)
        sw_lo, sw_hi = _swing_bounds(h, l)
        walls = {"don20": (lo20, up20), "don55": (lo55, up55), "bb20": (lo_bb, up_bb),
                 "va60": (va_lo, va_hi), "swg": (sw_lo, sw_hi)}
        phis: dict[str, np.ndarray] = {}
        poss: dict[str, np.ndarray] = {}
        for key in _SDF_KEYS:
            phi, pos, width = _corridor_block(c, atr, *walls[key])
            phis[key], poss[key] = phi, pos
            f[f"sdf_{key}"] = phi
            f[f"sdf_{key}_pos"] = pos
            f[f"sdf_{key}_width"] = width
            f[f"sdf_{key}_grad5"] = (phi - _shift(phi, 5)) / 5.0
            f[f"sdf_{key}_curv"] = phi - 2.0 * _shift(phi, 1) + _shift(phi, 2)

        # ---- Group S: single lines
        tp = (h + l + c) / 3.0
        vsum = pd.Series(v).rolling(20).sum()
        vwap = (pd.Series(tp * v).rolling(20).sum() / vsum).where(vsum > 0).to_numpy()
        f["sdf_vwap20"] = _line_sdf(c, vwap, atr)
        f["sdf_pdh"] = _line_sdf(c, _shift(h, 1), atr)
        f["sdf_pdl"] = _line_sdf(c, _shift(l, 1), atr)
        f["sdf_poc60"] = _line_sdf(c, poc, atr)

        # ---- Group S: composites
        phi_m = np.column_stack([phis[k] for k in _SDF_KEYS])
        pos_m = np.column_stack([poss[k] for k in _SDF_KEYS])
        abs_m = np.abs(phi_m)
        all_nan = np.isnan(abs_m).all(axis=1)
        filled = np.where(np.isnan(abs_m), np.inf, abs_m)
        f["sdf_min_abs"] = np.where(all_nan, np.nan, filled.min(axis=1))
        f["sdf_n_up_breach"] = ((phi_m > 0) & (pos_m > 1)).sum(axis=1).astype(float)
        f["sdf_n_dn_breach"] = ((phi_m > 0) & (pos_m < 0)).sum(axis=1).astype(float)
        f["sdf_nearest_k"] = np.where(all_nan, -1, filled.argmin(axis=1)).astype(np.int8)

        # ---- Group S: lagged memory (phi as known at t - lag)
        for key in _LAG_KEYS:
            for lag in _LAGS:
                f[f"sdf_{key}_l{lag}"] = _shift(phis[key], lag)

        # ---- Group X: market context (cross-sectional ranks are filled after concat)
        for col in ("cs_rank_ret_5", "cs_rank_ret_20", "cs_rank_sdf_don20", "cs_rank_vol_z20"):
            f[col] = np.full(n, np.nan)
        for col in ("mkt_ret_5", "mkt_ret_20", "mkt_rv_20", "mkt_sdf_don55", "vix_close", "vix_chg_5"):
            f[col] = ctx[col].to_numpy(dtype=float)

        # ---- Group C
        if "sector_id" in g.columns:
            sector = pd.to_numeric(g["sector_id"], errors="coerce").fillna(-1).clip(-1, 10)
            f["sector_id"] = sector.to_numpy()
        else:
            f["sector_id"] = np.full(n, -1)
        dates = pd.DatetimeIndex(pd.to_datetime(g["session_date"]))
        f["dow"] = dates.dayofweek.to_numpy()
        f["month"] = dates.month.to_numpy()

        # ---- tradable mask (spec 3): raw close >= 5 and ADV20 dollar >= 20M, data through t
        if "tradable" in g.columns:
            tradable = g["tradable"].fillna(False).to_numpy(dtype=bool)
        else:
            tradable = (c_raw >= _MIN_PRICE) & (adv20 >= _MIN_ADV20)

    for k in f:
        if k not in _INT8_COLUMNS:
            f[k] = _clean(f[k])
    out = pd.DataFrame(f)
    out.insert(0, "session_date", g["session_date"].to_numpy())
    out.insert(0, "symbol", g["symbol"].to_numpy())
    out["_tradable"] = tradable
    return out


def build_features(bars: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Build the ``sdf_fs_v1`` feature matrix, sorted by (symbol, session_date).

    Causal window: row ``t`` uses bars ``<= t`` of the same symbol, same-date cross-sectional
    information, and market context (SPY / VIX) of dates ``<= t`` only. ``context`` may supply
    ``"SPY"`` (OHLC) and ``"^VIX"`` (close) frames; see the module docstring for fallbacks.
    Output columns: ``KEYS`` followed by ``FEATURE_COLUMNS``; float32 except the int8 categoricals.
    """
    missing = _REQUIRED - set(bars.columns)
    if missing:
        raise DataContractError(f"bars missing columns {sorted(missing)}")
    cols = list(KEYS) + list(FEATURE_COLUMNS)
    ordered = bars.sort_values(list(KEYS)).reset_index(drop=True)
    is_vix = ordered["symbol"] == "^VIX"
    vix_rows = ordered[is_vix]
    ordered = ordered[~is_vix].reset_index(drop=True)
    if ordered.empty:
        return pd.DataFrame(columns=cols)

    mk = _market_context(ordered, vix_rows, context)
    parts = [_features_one(g, mk) for _, g in ordered.groupby("symbol", sort=True)]
    out = pd.concat(parts, ignore_index=True)

    # Cross-sectional percentile ranks on each date over the tradable universe, in (0, 1].
    tradable = out["_tradable"].to_numpy(dtype=bool)
    for col, src in (("cs_rank_ret_5", "ret_5"), ("cs_rank_ret_20", "ret_20"),
                     ("cs_rank_sdf_don20", "sdf_don20"), ("cs_rank_vol_z20", "vol_z20")):
        vals = out[src].where(tradable)
        out[col] = vals.groupby(out["session_date"]).rank(method="average", pct=True)

    out = out[cols].copy()
    for col in FEATURE_COLUMNS:
        out[col] = out[col].astype(np.int8 if col in _INT8_COLUMNS else np.float32)
    return out
