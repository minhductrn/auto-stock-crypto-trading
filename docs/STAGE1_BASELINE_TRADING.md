# STAGE1_BASELINE_TRADING.md

**Status:** NORMATIVE. This is the single source of truth for the Stage 1 US-equity trading engine.
**Spec version:** `stage1-v1.0.0`
**Feature set version:** `sdf_fs_v1` (82 model-input columns, fixed order, see §5)
**Label set version:** `tb_v1`
**Scope:** US-listed common stocks only. Daily bars. Long-only. Research, backtest and paper-trading signal generation.
**Out of scope (forbidden in Stage 1):** every crypto asset, exchange, data feed, library or 24/7 market assumption. See §13.

The keywords MUST, MUST NOT, SHOULD and MAY follow RFC 2119. Any deviation needs a version bump of this document and a changelog entry in §15.

---

## 0. Repository State and Required Actions

Inspection of `/home/dtran/projects/auto-stock-crypto-trading` showed only:

```
README.md        # one line: "ai-sdf auto-trading project - Stage 1 (US Stock)"
.gitignore       # .venv/, __pycache__/, *.pyc, .env, node_modules/
```

Nothing is implemented yet. All directories in §2 are created from scratch. Required `.gitignore` additions: `data/`, `artifacts/`, `*.parquet`, `*.ubj`, `*.lgb`, `.pytest_cache/`, `.mypy_cache/`. The `data/` and `artifacts/` roots are never committed. Their `.gitkeep` files are the exception.

---

## 1. Objectives and Non-Goals

### 1.1 Objectives
1. Build a reproducible, leakage-free pipeline: US equity daily bars → geometric (SDF) feature matrix → XGBoost and LightGBM → probability signals → cost-aware backtest.
2. Define the **AI-SDF geometry layer**. Each stock's price is placed inside time-varying corridors of market-structure boundaries. Signed, ATR-normalised distances to those boundaries, and their temporal derivatives, become the model features.
3. Fix strict, versioned contracts (schemas, interfaces, directory layout) so independent developers can work in parallel without drift.
4. Prove the value of the SDF layer with a mandatory ablation (§11.4).

### 1.2 Non-goals for Stage 1
Intraday or HFT execution, shorting, leverage, options, live order routing, deep learning, reinforcement learning, and any non-equity asset class.

---

## 2. Immutable Directory Structure

Paths are relative to the repo root. Developers MUST NOT add top-level directories without amending this document.

```
auto-stock-crypto-trading/
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/
│   └── STAGE1_BASELINE_TRADING.md          # this file
├── configs/
│   └── stage1.yaml                         # the only runtime config (see §12)
├── data/                                   # git-ignored
│   ├── raw/                                # L0: vendor payloads, immutable, append-only
│   │   ├── bars_1d/                        #   date=YYYY-MM-DD/part-*.parquet
│   │   ├── corporate_actions/
│   │   ├── universe_membership/
│   │   └── reference/                      #   sector map, VIX, calendar
│   ├── silver/                             # L1: validated, split/dividend-adjusted
│   │   └── bars_1d_adj/                    #   year=YYYY/part-*.parquet
│   ├── gold/                               # L2: model-ready
│   │   ├── features/sdf_fs_v1/             #   year=YYYY/part-*.parquet
│   │   └── labels/tb_v1/                   #   year=YYYY/part-*.parquet
│   └── quarantine/                         # rows that failed validation
├── artifacts/                              # git-ignored
│   ├── models/{run_id}/                    #   xgb.ubj, lgbm.txt, calibrators, manifest.json
│   ├── signals/{run_id}/                   #   signals.parquet
│   └── backtests/{run_id}/                 #   trades.parquet, equity.parquet, metrics.json
├── src/
│   └── sdftrade/
│       ├── __init__.py
│       ├── contracts/                      # dataclasses, Protocols, pandera schemas (§8, §9)
│       │   ├── schemas.py
│       │   ├── interfaces.py
│       │   └── errors.py
│       ├── ingestion/                      # DataProvider adapters, L0 writers
│       ├── calendar/                       # NYSE session calendar wrapper
│       ├── universe/                       # point-in-time universe builder
│       ├── adjust/                         # corporate-action adjustment (L0→L1)
│       ├── validation/                     # data-quality gates
│       ├── sdf/                            # boundary builders + signed distance kernels (§4)
│       │   ├── boundaries.py
│       │   ├── kernels.py
│       │   └── registry.py                 # corridor registry (frozen list)
│       ├── features/                       # feature assembly in the §5 column order
│       ├── labels/                         # triple-barrier labeller
│       ├── models/                         # xgb_model.py, lgbm_model.py, calibrate.py, ensemble.py
│       ├── cv/                             # purged walk-forward splitter
│       ├── signals/                        # probability → target weights
│       ├── backtest/                       # daily event-driven simulator
│       └── cli.py                          # entry points, see §10
├── tests/
│   ├── unit/
│   ├── contract/                           # schema, interface and column-order tests
│   ├── leakage/                            # look-ahead and truncation tests
│   └── golden/                             # frozen small fixtures with expected outputs
└── scripts/
    └── check_no_crypto.sh                  # CI guard (§13)
```

Language and toolchain: Python ≥ 3.11, `pandas ≥ 2.1`, `numpy`, `pyarrow`, `pandera`, `xgboost ≥ 2.0`, `lightgbm ≥ 4.0`, `scikit-learn`, `exchange_calendars`, `pyyaml`, `pytest`, `mypy --strict` on `contracts/`.

---

## 3. Market and Data Specification

| Item | Specification |
|---|---|
| Asset class | US-listed common stocks (NYSE, Nasdaq, NYSE American). No ETFs, ADRs, preferreds, warrants or OTC names in the **tradable** universe. |
| Benchmarks and context instruments | `SPY` (market proxy) and `^VIX` (volatility index). Read-only context, never tradable. |
| Universe | Point-in-time S&P 500 constituents. Membership is by effective date and never uses today's list for past dates. |
| Bar frequency | 1 session = 1 daily bar, keyed by `session_date`, with the timestamp semantic "session close, `America/New_York`". |
| Calendar | NYSE regular sessions from `exchange_calendars` (`XNYS`). Half-days are valid sessions. |
| History | `2010-01-01` → present. A symbol's first model-usable feature row needs ≥ 252 prior sessions. |
| Raw price basis | Raw OHLCV is stored **unadjusted**, with split and dividend events stored separately. Adjustment is done by us in L0→L1, never by the vendor. |
| Adjusted basis | Split- and dividend-adjusted (total-return style) OHLC. Volume is split-adjusted. |
| Liquidity filter (tradable mask) | `close_raw ≥ 5.00` and `ADV20_dollar ≥ 20,000,000`, evaluated at `t` using data through `t`. `ADV20_dollar = mean(close_raw·volume_raw over [t-19..t])`. |
| Timezone | Timestamps are stored as UTC `datetime64[ns, UTC]`. `session_date` is an exchange-local `date32`. |

### 3.1 Time conventions (the anti-leakage backbone)
- **Decision time `t`:** after the close of session `t`. Every feature at row `(symbol, t)` MUST be computable from bars with `session_date ≤ t`.
- **Execution time:** the open of session `t+1` (`entry_px = open[t+1]`, adjusted basis). Fills at the close of `t` are forbidden.
- **Label window:** starts at `t+1` and extends through at most `t+H`.
- A feature or label that uses information beyond its allowed window is a **contract violation**. The tests in `tests/leakage/` MUST fail the build when this happens (§11.5).

### 3.2 Adjustment-invariance rule
The adjustment factors are anchored at the latest date, so the adjusted history is rescaled whenever a new corporate action arrives. To keep features causal, every feature in §5 MUST be invariant to a constant positive rescaling of the whole price history of a symbol (ratios, log-returns and ATR-normalised distances all satisfy this). The only exceptions are `dollar_vol_20_log` and the tradable mask, which MUST use **raw** close and **raw** volume. Feature tests rescale the history by a random constant and assert that outputs are unchanged within `1e-6` (except those two).

---

## 4. AI-SDF Financial Geometry (Normative Mathematics)

### 4.1 Concept
For a symbol at decision time `t`, the price axis is a one-dimensional space. A **corridor** `k` is a pair of boundaries `(L_k(t), U_k(t))` built only from data up to `t` (or `t-1` where stated). The **signed distance field** of corridor `k` evaluated at the close `c_t` is:

```
L'_k = min(L_k, U_k)
U'_k = max(L_k, U_k)

φ_k(t) = max( L'_k - c_t ,  c_t - U'_k ) / ATR14_t
```

Sign convention (SDF standard):
- `φ_k < 0`: price is **inside** the corridor. The magnitude is the distance to the nearest wall, in ATR units.
- `φ_k = 0`: price is exactly on a boundary.
- `φ_k > 0`: price is **outside** the corridor. The magnitude is the distance beyond the nearest wall. Whether it broke up or down is encoded by `pos_k`.

Companion geometry per corridor:

```
pos_k(t)   = (c_t - L'_k) / (U'_k - L'_k)     # 0 = lower wall, 1 = upper wall; <0 or >1 outside
                                              # clipped to [-1, 2]; if U'_k - L'_k < 1e-8*c_t → NaN
width_k(t) = (U'_k - L'_k) / ATR14_t          # corridor thickness in ATR units
grad5_k(t) = (φ_k(t) - φ_k(t-5)) / 5          # field velocity
curv_k(t)  =  φ_k(t) - 2·φ_k(t-1) + φ_k(t-2) # field acceleration
```

`ATR14_t` is the Wilder ATR over 14 sessions using adjusted OHLC, with `TR = max(h-l, |h-c_{t-1}|, |l-c_{t-1}|)` and the first value seeded by the simple mean of the first 14 TRs. If `ATR14_t < 1e-8 · c_t`, set it to `1e-8 · c_t`. All `φ_k` values are clipped to `[-10, 10]`.

Single-line boundaries (degenerate corridors) use the plain signed ATR distance `(c_t - line_t) / ATR14_t`, also clipped to `[-10, 10]`.

### 4.2 Frozen corridor registry (`sdf/registry.py`)
Exactly five corridors exist in `sdf_fs_v1`. The order and IDs are frozen. Windows are in sessions and all values are on the adjusted basis.

| ID | Key | Lower wall `L` | Upper wall `U` | Lookback |
|---|---|---|---|---|
| 0 | `don20` | `min(low[t-20 … t-1])` | `max(high[t-20 … t-1])` | 20 (excludes `t`, so a breakout gives `φ > 0`) |
| 1 | `don55` | `min(low[t-55 … t-1])` | `max(high[t-55 … t-1])` | 55 (excludes `t`) |
| 2 | `bb20` | `SMA20(c) - 2·σ20(c)` | `SMA20(c) + 2·σ20(c)` | 20 (includes `t`, `σ` with `ddof=0`) |
| 3 | `va60` | Value Area Low (VAL) | Value Area High (VAH) | 60, window `[t-60 … t-1]` |
| 4 | `swg` | most recent confirmed swing low | most recent confirmed swing high | last 120 sessions |

**Value Area (`va60`) algorithm (deterministic):**
1. `P_min = min(low)` and `P_max = max(high)` over the window. Create 40 equal-width bins on `[P_min, P_max]`.
2. For each bar, add its volume to the bin containing the typical price `(h+l+c)/3`. A value on the upper edge goes to the top bin.
3. POC is the center of the bin with the greatest volume. Ties go to the lowest bin.
4. Starting from the POC bin, repeatedly add the adjacent bin (above or below) with the greater volume until the accumulated volume ≥ 70% of the total. Ties go to the upper bin. When one side is exhausted, take the other.
5. `VAL` is the lower edge of the lowest included bin. `VAH` is the upper edge of the highest included bin. `POC` is the POC price, exposed as the point line `poc60`.
6. If `P_max == P_min` or total volume is 0, all `va60` outputs are NaN.

**Swing (`swg`) algorithm:** a bar `j` is a swing high when `high[j] > high[j±i]` for all `i = 1..3` (strict). A swing low is the mirror case. A swing at `j` becomes **known only at session `j+3`**, so at decision time `t` only swings with `j ≤ t-3` may be used. `U` is the highest confirmed swing high in the last 120 sessions and `L` is the lowest confirmed swing low in the last 120 sessions. If either is missing, all `swg` outputs are `NaN` (allowed).

### 4.3 Single-line boundaries

| Feature | Line |
|---|---|
| `sdf_vwap20` | `Σ_{i=t-19..t} tp_i·v_i / Σ v_i`, with `tp=(h+l+c)/3` |
| `sdf_pdh` | `high[t-1]` |
| `sdf_pdl` | `low[t-1]` |
| `sdf_poc60` | POC from the `va60` algorithm |

### 4.4 Composite geometry
Let `K = {don20, don55, bb20, va60, swg}`, ignoring corridors where `φ` is `NaN`.

```
sdf_min_abs      = min_k |φ_k|
sdf_n_up_breach  = #{ k : φ_k > 0 and pos_k > 1 }
sdf_n_dn_breach  = #{ k : φ_k > 0 and pos_k < 0 }
sdf_nearest_k    = argmin_k |φ_k|   (registry ID 0..4; ties → lowest ID; no valid k → -1)   dtype int8
```

### 4.5 Geometric time-series structure
The feature matrix is a **sampled, flattened view of the geometry field over time**. For every row, the SDF state is described at three temporal scales:
1. **Level** at `t`: `φ_k(t)`, `pos_k`, `width_k`.
2. **Local dynamics**: `grad5_k`, `curv_k`.
3. **Memory**: lagged levels `φ_k(t-ℓ)` for `ℓ ∈ {1, 2, 3, 5, 10}` for the corridors `don20`, `bb20`, `va60`, `swg` (columns `sdf_{key}_l{ℓ}`). `don55` has no lag columns by design.

Lagged values are recomputed from the corridor at the lagged time, and each uses the **ATR and walls as known at `t-ℓ`**. They are never recomputed with `t`-time walls.

---

## 5. Feature Matrix Contract: `sdf_fs_v1` (82 columns)

**Grain:** one row per `(symbol, session_date)`, restricted to the point-in-time universe on `session_date`. The tradable mask (§3) is a separate boolean column. Illiquid rows are kept but excluded from training and trading.

**Column order is immutable.** The `features` module MUST emit columns in exactly this order. `tests/contract/test_feature_columns.py` hashes the ordered list (`sha256`), and any change breaks the build. All feature columns are `float32`, except those marked `int8`.

### 5.0 Keys and metadata (not model inputs)

| Column | dtype | Meaning |
|---|---|---|
| `symbol` | string | Ticker (a stable `security_id` is also carried in `universe_membership`) |
| `session_date` | date32 | Decision session `t` |
| `tradable` | bool | Liquidity and price filter at `t` |
| `fs_version` | string | `"sdf_fs_v1"` |

### 5.1 Group P: Price and Volume (16 columns, positions 1–16)

| # | Column | Definition (adjusted basis unless stated) |
|---|---|---|
| 1 | `ret_1` | `ln(c_t / c_{t-1})` |
| 2 | `ret_5` | `ln(c_t / c_{t-5})` |
| 3 | `ret_10` | `ln(c_t / c_{t-10})` |
| 4 | `ret_20` | `ln(c_t / c_{t-20})` |
| 5 | `ret_60` | `ln(c_t / c_{t-60})` |
| 6 | `rv_20` | `std(ret_1[t-19..t], ddof=1)` |
| 7 | `rv_60` | `std(ret_1[t-59..t], ddof=1)` |
| 8 | `atr14_pct` | `ATR14_t / c_t` |
| 9 | `gap_pct` | `o_t / c_{t-1} - 1` |
| 10 | `range_pct` | `(h_t - l_t) / c_t` |
| 11 | `close_loc` | `(c_t - l_t) / (h_t - l_t)`, 0.5 if `h=l` |
| 12 | `vol_z20` | z-score of `ln(1+v_t)` over `[t-19..t]` (`ddof=1`; 0 if std = 0), split-adjusted volume |
| 13 | `dollar_vol_20_log` | `ln(1 + mean(close_raw·volume_raw over [t-19..t]))` (raw basis, see §3.2) |
| 14 | `rsi14` | Wilder RSI(14), range 0–100 |
| 15 | `macd_hist_atr` | `(MACD(12,26) - EMA9(MACD)) / ATR14_t` |
| 16 | `beta_60` | OLS slope of `ret_1` on `SPY ret_1` over 60 sessions |

### 5.2 Group S: SDF Geometry (53 columns, positions 17–69)

**Corridor blocks (25 columns, positions 17–41).** For each key in the order `don20, don55, bb20, va60, swg`, five columns in this order: `sdf_{key}`, `sdf_{key}_pos`, `sdf_{key}_width`, `sdf_{key}_grad5`, `sdf_{key}_curv`.

```
17 sdf_don20          18 sdf_don20_pos     19 sdf_don20_width   20 sdf_don20_grad5   21 sdf_don20_curv
22 sdf_don55          23 sdf_don55_pos     24 sdf_don55_width   25 sdf_don55_grad5   26 sdf_don55_curv
27 sdf_bb20           28 sdf_bb20_pos      29 sdf_bb20_width    30 sdf_bb20_grad5    31 sdf_bb20_curv
32 sdf_va60           33 sdf_va60_pos      34 sdf_va60_width    35 sdf_va60_grad5    36 sdf_va60_curv
37 sdf_swg            38 sdf_swg_pos       39 sdf_swg_width     40 sdf_swg_grad5     41 sdf_swg_curv
```

**Single lines and composites (8 columns, positions 42–49):**

```
42 sdf_vwap20         43 sdf_pdh           44 sdf_pdl           45 sdf_poc60
46 sdf_min_abs        47 sdf_n_up_breach   48 sdf_n_dn_breach   49 sdf_nearest_k (int8)
```

**Lagged memory (20 columns, positions 50–69):**

```
50 sdf_don20_l1   51 sdf_don20_l2   52 sdf_don20_l3   53 sdf_don20_l5   54 sdf_don20_l10
55 sdf_bb20_l1    56 sdf_bb20_l2    57 sdf_bb20_l3    58 sdf_bb20_l5    59 sdf_bb20_l10
60 sdf_va60_l1    61 sdf_va60_l2    62 sdf_va60_l3    63 sdf_va60_l5    64 sdf_va60_l10
65 sdf_swg_l1     66 sdf_swg_l2     67 sdf_swg_l3     68 sdf_swg_l5     69 sdf_swg_l10
```

Group S total = 25 + 8 + 20 = **53 columns**.

**NaN policy:** all `*_swg*` columns, `sdf_swg_l*` and `sdf_nearest_k = -1` may be NaN or -1 when no valid swing exists. `va60` columns and `sdf_poc60` may be NaN only when the window is degenerate (§4.2 item 6). `sdf_min_abs` is NaN only if all corridors are NaN. All other S columns MUST be finite after warm-up.

### 5.3 Group X: Cross-Sectional and Market Context (10 columns, positions 70–79)

| # | Column | Definition |
|---|---|---|
| 70 | `cs_rank_ret_5` | Percentile rank of `ret_5` across the **tradable** universe on date `t`, in (0,1] |
| 71 | `cs_rank_ret_20` | Same for `ret_20` |
| 72 | `cs_rank_sdf_don20` | Same for `sdf_don20` |
| 73 | `cs_rank_vol_z20` | Same for `vol_z20` |
| 74 | `mkt_ret_5` | `SPY ret_5` |
| 75 | `mkt_ret_20` | `SPY ret_20` |
| 76 | `mkt_rv_20` | `SPY rv_20` |
| 77 | `mkt_sdf_don55` | `sdf_don55` computed on SPY |
| 78 | `vix_close` | `^VIX` close at `t` |
| 79 | `vix_chg_5` | `ln(VIX_t / VIX_{t-5})` |

Ranks use `method="average"`, are normalised by the count of valid names, and use only same-date information. Rows with `tradable == False` get NaN ranks.

### 5.4 Group C: Static and Calendar (3 columns, positions 80–82)

| # | Column | Definition |
|---|---|---|
| 80 | `sector_id` (int8) | GICS sector 0..10 at `t` (point-in-time). Unknown = -1. |
| 81 | `dow` (int8) | Day of week of `t`, 0 = Monday … 4 = Friday |
| 82 | `month` (int8) | Calendar month 1..12 |

Totals: P 16 + S 53 + X 10 + C 3 = **82 model input columns**. Normalisation is not applied, since tree models are scale-invariant. Winsorisation applies only where §4 mandates a clip.

### 5.5 Warm-up and row eligibility
- A row is **eligible** iff the symbol has ≥ 252 prior sessions of validated data and `tradable == True`.
- Eligible rows in the training matrix MUST NOT contain `±inf`. NaN is permitted only in the columns named in §5.2 and in the SPY/VIX-derived columns on dates where SPY or VIX is itself in warm-up (such dates are not eligible).
- Rows that fail these checks are written to `data/quarantine/` with a `reason` column. They are never silently dropped.

---

## 6. Label Contract: `tb_v1` (Triple Barrier)

For each eligible `(symbol, t)`:

```
entry_px = open_adj[t+1]
ATR_ref  = ATR14_t
upper    = entry_px + 1.5 · ATR_ref
lower    = entry_px - 1.5 · ATR_ref
H        = 10 sessions   (the window is t+1 … t+10)
```

Scan sessions `t+1 … t+H` in order. For the first session `s` where any barrier is touched:
- If only `high_s ≥ upper`: label `+1`, exit price `upper`.
- If only `low_s ≤ lower`: label `-1`, exit price `lower`.
- If **both** are touched in the same bar: label `-1`, exit price `lower` (conservative, pessimistic tie-break).
- **Gaps:** for `s ≥ t+2`, if `open_s` is already beyond a barrier (`open_s ≥ upper` or `open_s ≤ lower`), that barrier is hit at the open and the exit price is `open_s`. On `t+1` the barriers are defined relative to `open[t+1]`, so no open gap applies.
- If no barrier is touched by `t+H`: label `0`, exit price `close_adj[t+H]`.

Output columns:

| Column | dtype | Meaning |
|---|---|---|
| `symbol`, `session_date` | | Keys, `t` |
| `y_tb` | int8 | −1, 0, +1 |
| `y_long` | int8 | **Model target:** `1` iff `y_tb == +1`, else `0` |
| `t_exit` | date32 | Session on which the barrier was hit, or `t+H` |
| `ret_fwd` | float32 | `ln(exit_px / entry_px)` |
| `ret_fwd_net` | float32 | `ret_fwd - 2·cost_bps/1e4` (round-trip cost from the config) |
| `sample_weight` | float32 | Average-uniqueness weight (López de Prado), computed from the label spans `[t+1, t_exit]` and normalised to mean 1 within the training fold |
| `label_version` | string | `"tb_v1"` |

Rows whose window `t+1…t+H` is not fully inside the data are **not labelled** (no partial labels).

---

## 7. Model Contract (XGBoost and LightGBM)

### 7.1 Inputs
- **X:** the 82 columns of §5 in the order given. Metadata columns (`symbol`, `session_date`, `tradable`, `fs_version`) MUST NOT be passed to the model.
- **y:** `y_long`.
- **w:** `sample_weight`.
- Categorical columns: `sector_id`, `dow`, `month`, `sdf_nearest_k`. LightGBM uses native categorical handling. XGBoost uses `enable_categorical=True` with `category` dtype. Negative codes are mapped to a dedicated "unknown" category.
- Both models MUST see the same 82 columns, verified by comparing `feature_names` in the manifest.

### 7.2 Baseline hyperparameters (fixed for Stage 1; changes need a spec version bump)

**XGBoost** (`xgboost.XGBClassifier`):

| Param | Value |
|---|---|
| `objective` | `binary:logistic` |
| `eval_metric` | `logloss` |
| `tree_method` | `hist` |
| `max_depth` | 4 |
| `learning_rate` | 0.03 |
| `n_estimators` | 1000 (early stopping, 50 rounds) |
| `min_child_weight` | 50 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.6 |
| `reg_lambda` | 10.0 |
| `reg_alpha` | 1.0 |
| `gamma` | 0.0 |
| `max_cat_to_onehot` | 4 |
| `random_state` | 42 |
| `n_jobs` | from config |

**LightGBM** (`lightgbm.LGBMClassifier`):

| Param | Value |
|---|---|
| `objective` | `binary` |
| `metric` | `binary_logloss` |
| `boosting_type` | `gbdt` |
| `num_leaves` | 15 |
| `max_depth` | -1 |
| `learning_rate` | 0.03 |
| `n_estimators` | 1000 (early stopping, 50 rounds) |
| `min_child_samples` | 200 |
| `subsample` | 0.7, `subsample_freq` = 1 |
| `colsample_bytree` | 0.6 |
| `reg_lambda` | 10.0 |
| `reg_alpha` | 1.0 |
| `cat_smooth` | 20 |
| `deterministic` | True, `force_row_wise` = True |
| `random_state` | 42 |

### 7.3 Training protocol
1. **Splits:** purged walk-forward only (§7.4). Random K-fold is forbidden.
2. **Early-stopping set:** the last 10% of the training dates in each fold, **purged** against the remaining training rows by `H` sessions (rows whose label span overlaps are removed from the training remainder). It is never the test block.
3. **Calibration:** isotonic regression fit on the early-stopping set predictions of each model, applied to test predictions. Calibrators are stored next to the model.
4. **Ensemble:** `p_long = 0.5·p_xgb_cal + 0.5·p_lgbm_cal`. Single-model results are also reported.
5. **Seeds:** all seeds are fixed to 42. Training twice on identical input MUST give bit-identical predictions on the same hardware (tolerance `1e-6` across hardware).

### 7.4 Purged walk-forward cross-validation (`cv/`)

| Parameter | Value |
|---|---|
| Scheme | Expanding window |
| Minimum train span | 5 years (from 2010-01-01) |
| Test block | 6 months |
| Step | 6 months |
| Purge | Remove every training row whose label span `[t+1, t_exit]` reaches `test_start` or later (i.e. `t_exit ≥ test_start`) |
| Embargo | 10 sessions after each test block before data re-enters any later training set |
| Final holdout | The most recent **12 months** are locked and touched only once, for the final report (`--holdout` flag) |

### 7.5 Model manifest (`artifacts/models/{run_id}/manifest.json`)
MUST contain: `spec_version`, `fs_version`, `label_version`, `feature_names` (ordered), `feature_hash_sha256`, `train_start/end`, per-fold date ranges, hyperparameters, library versions, `git_commit`, `seed`, and metrics per fold.

---

## 8. Data Pipeline Contract: Layers and Schemas

Pipeline: `L0 raw → L1 silver → L2 gold (features, labels) → models → signals → backtest`. Each stage reads only from the previous layer. Writing back to an earlier layer is forbidden. Every write is atomic (write to a temp path, then rename) and idempotent for the same inputs. Partition columns are as in §2.

### 8.1 L0 `raw/bars_1d` (unadjusted)

| Column | dtype | Constraint |
|---|---|---|
| `symbol` | string | non-null, uppercase |
| `session_date` | date32 | must be an XNYS session |
| `open`, `high`, `low`, `close` | float64 | > 0 |
| `volume` | int64 | ≥ 0 |
| `vendor` | string | |
| `ingested_at` | timestamp[ns, UTC] | |

### 8.2 L0 `raw/corporate_actions`

| Column | dtype | Constraint |
|---|---|---|
| `symbol` | string | |
| `ex_date` | date32 | |
| `action_type` | string | `SPLIT` or `CASH_DIVIDEND` |
| `split_ratio` | float64 | new shares per old share (2-for-1 → 2.0); null if dividend |
| `cash_amount` | float64 | per share, unadjusted; null if split |

### 8.3 L0 `raw/universe_membership`

| Column | dtype | Constraint |
|---|---|---|
| `security_id` | string | stable across ticker changes |
| `symbol` | string | ticker valid for that interval |
| `valid_from` | date32 | inclusive |
| `valid_to` | date32 | inclusive, null if still a member |
| `sector_id` | int8 | GICS 0..10 |

### 8.4 L1 `silver/bars_1d_adj`

All columns from 8.1, with the raw prices renamed `open_raw`, `high_raw`, `low_raw`, `close_raw` and `volume` renamed `volume_raw`, plus:

| Column | dtype | Meaning |
|---|---|---|
| `adj_factor` | float64 | cumulative multiplicative price factor as of that date (latest = 1.0) |
| `open`, `high`, `low`, `close` | float64 | adjusted: `raw · adj_factor` |
| `volume` | float64 | split-adjusted volume |
| `dq_flags` | int32 | bitmask of non-fatal warnings |

Dividend adjustment follows the standard backward method: `factor = (prev_close - cash) / prev_close` for sessions before `ex_date`. Split factors are `1/split_ratio`. Factors multiply cumulatively.

### 8.5 Data-quality gates (`validation/`)
Fatal violations send the row to `quarantine/`. `DataContractError` is raised for the batch if the quarantined share of any date exceeds 2%.

| Rule | Fatal? |
|---|---|
| `high ≥ max(open, close, low)` and `low ≤ min(open, close, high)` | Yes |
| Price ≤ 0, or volume < 0 | Yes |
| Duplicate `(symbol, session_date)` | Yes |
| `session_date` not an XNYS session | Yes |
| Missing session inside a symbol's active interval | Yes (the symbol is flagged for that date) |
| Absolute one-day adjusted return > 40% without a corporate action | Warning (`dq_flags` bit 0) |
| Zero volume on a session | Warning (`dq_flags` bit 1) |

### 8.6 L2 outputs
`gold/features/sdf_fs_v1` uses the schema in §5, including keys. `gold/labels/tb_v1` uses the schema in §6. Both are partitioned by `year` of `session_date`. Features are generated for all eligible dates. Labels are generated only where the full window exists.

### 8.7 Signals (`artifacts/signals/{run_id}/signals.parquet`)

| Column | dtype | Meaning |
|---|---|---|
| `session_date` | date32 | Decision session `t` |
| `symbol` | string | |
| `p_xgb`, `p_lgbm`, `p_long` | float32 | Calibrated probabilities |
| `rank` | int16 | Rank by `p_long` on `t` among tradable names (1 = best) |
| `target_weight` | float32 | Portfolio weight to hold from the `t+1` open, ≥ 0 |
| `run_id` | string | |

---

## 9. Interface Contracts (`src/sdftrade/contracts/interfaces.py`)

These Protocols are mandatory. Implementations MUST NOT alter signatures. The `Frame` type is `pandas.DataFrame`.

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence, Mapping, runtime_checkable
import pandas as pd

Frame = pd.DataFrame

@dataclass(frozen=True)
class DateRange:
    start: date          # inclusive
    end: date            # inclusive

@runtime_checkable
class DataProvider(Protocol):
    """Vendor adapter. MUST return unadjusted data conforming to §8.1/§8.2."""
    def fetch_bars_1d(self, symbols: Sequence[str], rng: DateRange) -> Frame: ...
    def fetch_corporate_actions(self, symbols: Sequence[str], rng: DateRange) -> Frame: ...
    def fetch_index_levels(self, symbol: str, rng: DateRange) -> Frame: ...   # e.g. "^VIX"

@runtime_checkable
class UniverseProvider(Protocol):
    def members_on(self, session_date: date) -> Frame: ...                    # §8.3 rows valid that date
    def membership(self, rng: DateRange) -> Frame: ...

@runtime_checkable
class SessionCalendar(Protocol):
    def sessions(self, rng: DateRange) -> Sequence[date]: ...
    def next_session(self, d: date) -> date: ...
    def prev_session(self, d: date) -> date: ...
    def is_session(self, d: date) -> bool: ...

@runtime_checkable
class BarAdjuster(Protocol):
    def adjust(self, raw_bars: Frame, actions: Frame) -> Frame: ...           # L0 → L1 (§8.4)

@runtime_checkable
class BarValidator(Protocol):
    def validate(self, bars: Frame) -> tuple[Frame, Frame]: ...               # (valid, quarantined)

@runtime_checkable
class BoundaryBuilder(Protocol):
    """Builds (L, U) series for one corridor from bars with session_date ≤ t only."""
    key: str                                                                  # "don20", ...
    registry_id: int
    def build(self, bars: Frame) -> Frame: ...                                # cols: symbol, session_date, L, U

@runtime_checkable
class FeatureBuilder(Protocol):
    fs_version: str                                                           # "sdf_fs_v1"
    columns: tuple[str, ...]                                                  # the 82 names, §5 order
    def build(self, bars_adj: Frame, context: Mapping[str, Frame],
              membership: Frame) -> Frame: ...                                # §5 schema

@runtime_checkable
class Labeller(Protocol):
    label_version: str                                                        # "tb_v1"
    def label(self, bars_adj: Frame, features_keys: Frame) -> Frame: ...      # §6 schema

@runtime_checkable
class SplitGenerator(Protocol):
    def splits(self, keys: Frame, labels: Frame) -> Sequence[tuple[Frame, Frame]]: ...  # (train_keys, test_keys)

@runtime_checkable
class StageModel(Protocol):
    name: str                                                                 # "xgb" | "lgbm"
    def fit(self, X: Frame, y: pd.Series, w: pd.Series,
            X_val: Frame, y_val: pd.Series, w_val: pd.Series) -> None: ...
    def predict_proba(self, X: Frame) -> pd.Series: ...                       # P(y_long = 1), calibrated
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "StageModel": ...

@runtime_checkable
class SignalGenerator(Protocol):
    def generate(self, probs: Frame, features_meta: Frame) -> Frame: ...      # §8.7 schema

@runtime_checkable
class Backtester(Protocol):
    def run(self, signals: Frame, bars_adj: Frame) -> Mapping[str, Frame]: ...  # trades, equity
```

**Interface rules**
1. Pure functions of inputs. No hidden global state, and no network I/O except in `DataProvider` implementations.
2. Every `build`, `label` and `generate` MUST verify its output against the pandera schema in `contracts/schemas.py` before returning. A violation raises `SchemaContractError`.
3. `FeatureBuilder.build` MUST be **causally truncation-invariant**: the result for date `t` must be identical (within `1e-6`) whether the input contains data after `t` or not (tested in §11.5).
4. Errors derive from `SdfTradeError` (`DataContractError`, `SchemaContractError`, `LeakageError`, `ConfigError`). Raising a bare `Exception` is forbidden.

---

## 10. CLI Entry Points (`sdftrade.cli`)

| Command | Reads | Writes |
|---|---|---|
| `ingest --start --end` | vendor | `raw/*` |
| `build-silver` | `raw/*` | `silver/bars_1d_adj`, `quarantine/` |
| `build-features` | `silver`, `raw/reference`, `raw/universe_membership` | `gold/features/sdf_fs_v1` |
| `build-labels` | `silver`, `gold/features` keys | `gold/labels/tb_v1` |
| `train --run-id` | `gold/*` | `artifacts/models/{run_id}` |
| `signal --run-id` | `gold/features`, models | `artifacts/signals/{run_id}` |
| `backtest --run-id [--holdout]` | `signals`, `silver` | `artifacts/backtests/{run_id}` |
| `ablate --run-id` | `gold/*` | ablation report (§11.4) |

All commands read only `configs/stage1.yaml` and CLI flags, and log the effective config hash.

---

## 11. Signal, Portfolio, Backtest, and Acceptance

### 11.1 Signal → portfolio rules (long-only)
1. On decision date `t`, take tradable names with `p_long ≥ 0.55`.
2. Keep the top `K = 20` by `p_long`. If fewer than 5 qualify, hold cash.
3. Weights are `w_i ∝ 1/atr14_pct_i` (inverse volatility), capped at **5%** per name, and renormalised under the cap. Gross exposure ≤ 100%. The residual is cash (zero return).
4. Positions are held for the label horizon: exit at the first of the barrier touch (stop `lower` or target `upper` from §6) or the `t+H` close. A name already held is not re-bought until flat. Max concurrent positions is 20.
5. Per-day turnover cap: 30% one-way. Excess orders are scaled proportionally.
6. No shorting, no leverage, no margin interest in Stage 1.

### 11.2 Execution and cost model
- Entry at the `open[t+1]` price. Barrier exits fill at the barrier price (or at `open` if gapped through). Time exits fill at `close[t+H]`.
- Cost per side: `cost_bps = 5` (slippage and fees combined) plus an impact penalty of `0.1 · (order_dollars / ADV20_dollar) · 1e4` bps, with participation capped at 2% of ADV20 (excess is not filled).
- Cash earns 0%.

### 11.3 Metrics (`metrics.json`)
Per fold and aggregate out-of-sample: AUC, log-loss, Brier score, precision at top-K, hit rate by `y_tb`, annualised return, annualised volatility, Sharpe (rf = 0), Sortino, max drawdown, Calmar, turnover, average holding period, exposure, net and gross alpha vs SPY (beta-adjusted), and cost drag in bps/yr.

### 11.4 Mandatory ablations
Same splits, same hyperparameters, different `X`:

| Run | Columns |
|---|---|
| `A_full` | all 82 |
| `B_no_sdf` | Group P (16) + Group X without `cs_rank_sdf_don20` and `mkt_sdf_don55` (8) + Group C (3) = **27 columns**; the manifest records the exact list |
| `C_sdf_only` | Group S (53) + Group C (3) = 56 columns |
| `D_benchmark` | Not a model: SPY buy-and-hold, and an equal-weight top-K by `ret_20` momentum baseline |

### 11.5 Acceptance gates for Stage 1 exit
All of the following MUST hold, or Stage 1 is not complete.

**Engineering gates (hard)**
1. `tests/contract`: the feature column hash matches and all schemas pass.
2. `tests/leakage`:
   (a) **Truncation invariance:** features for date `t` computed on data truncated at `t` equal those from the full history.
   (b) **Future-poison test:** replacing all bars after `t` with random noise leaves every feature at `t` unchanged.
   (c) **Label isolation:** no feature column equals or is a deterministic function of `ret_fwd` (checked by a correlation threshold of |ρ| < 0.5 on the pooled training set and by the (a)/(b) tests).
   (d) **Rescaling invariance** per §3.2.
3. `tests/golden`: the golden fixture reproduces the expected `φ_k` values to `1e-6`.
4. `scripts/check_no_crypto.sh` passes.
5. The full pipeline runs end to end via CLI with a fixed config and reproduces an identical `metrics.json` on a repeated run.

**Research gates (hard, on out-of-sample walk-forward folds, before the holdout)**
1. Ensemble out-of-sample AUC ≥ 0.52 in the pooled folds, and > 0.50 in at least 70% of folds.
2. Net Sharpe (after costs) ≥ 0.7 and max drawdown ≤ 25%.
3. `A_full` beats `B_no_sdf` on pooled AUC **and** net Sharpe. If it does not, the SDF layer is declared non-additive, and a written post-mortem is required before Stage 2.
4. The holdout is run once. Its net Sharpe must be ≥ 50% of the pooled walk-forward net Sharpe.

---

## 12. Configuration Contract (`configs/stage1.yaml`)

```yaml
spec_version: "stage1-v1.0.0"
market:
  exchange_calendar: "XNYS"
  timezone: "America/New_York"
  universe: "SP500_PIT"
  benchmark: "SPY"
  volatility_index: "^VIX"
  start_date: "2010-01-01"
  min_price: 5.0
  min_adv20_dollar: 20000000
  warmup_sessions: 252
sdf:
  atr_window: 14
  clip: 10.0
  corridors: [don20, don55, bb20, va60, swg]   # frozen order
  lag_set: [1, 2, 3, 5, 10]
  lag_corridors: [don20, bb20, va60, swg]
  va_bins: 40
  va_coverage: 0.70
  swing_k: 3
  swing_lookback: 120
label:
  version: "tb_v1"
  horizon: 10
  atr_mult: 1.5
  tie_break: "pessimistic"
cv:
  min_train_years: 5
  test_months: 6
  step_months: 6
  embargo_sessions: 10
  holdout_months: 12
model:
  seed: 42
  early_stopping_rounds: 50
  val_fraction: 0.10
  ensemble_weights: {xgb: 0.5, lgbm: 0.5}
portfolio:
  p_threshold: 0.55
  top_k: 20
  min_names: 5
  max_weight: 0.05
  turnover_cap: 0.30
  long_only: true
costs:
  cost_bps_per_side: 5
  impact_coeff: 0.1
  max_adv_participation: 0.02
paths:
  data_root: "data"
  artifacts_root: "artifacts"
```

Unknown keys or missing required keys raise `ConfigError`. Implicit defaults are not allowed.

---

## 13. Exclusion Policy: US Stocks Only

Stage 1 MUST contain **no** crypto components. The repository folder name is historical and does not extend the scope.

Forbidden in `src/`, `configs/`, `tests/` and `pyproject.toml`:
- Any crypto exchange SDK or aggregator library (for example `ccxt`, `python-binance`).
- Any crypto ticker, pair, or symbol format (for example `BTC`, `ETH`, `*-USD`, `*/USDT`).
- 24/7 calendars, funding rates, perpetual-swap concepts, on-chain data, and tick/lot rules specific to crypto.
- Any schema column, config key or module name suggesting a non-equity asset class.

`scripts/check_no_crypto.sh` runs a case-insensitive `grep -RniE` for those tokens over the paths above and exits non-zero on any hit. It runs in CI and pre-commit. `docs/` and the repo folder name are excluded from the check.

---

## 14. Developer Rules (Enforced)

1. **No look-ahead.** See §3.1 and the tests in §11.5. A PR that disables a leakage test is rejected.
2. **Adjusted basis only** for any feature or label. Raw columns are used only for the price filter and dollar-volume features (§3.2).
3. **Point-in-time everywhere:** universe, sector, and corporate-action knowledge.
4. **Column order, names, dtypes and version strings are API.** Changing them needs a new `fs_version` (e.g. `sdf_fs_v2`) and a new row in §15.
5. **Determinism:** fixed seeds, sorted inputs (`symbol`, `session_date`), no reliance on dict or set ordering.
6. **No hyperparameter tuning on the holdout.** Tuning, if any, happens inside the walk-forward folds and is recorded in the manifest.
7. **Typing:** `mypy --strict` passes for `contracts/`. Every public function has a docstring that states its causal window.
8. **Vectorised kernels:** the `sdf/kernels.py` implementation may use numpy or numba, but must match the golden fixtures exactly.

---

## 15. Changelog

| Version | Date | Change |
|---|---|---|
| stage1-v1.0.0 | initial | First baseline: daily US equities, 5 SDF corridors, 82-column `sdf_fs_v1`, triple-barrier `tb_v1`, XGBoost and LightGBM ensemble, purged walk-forward, long-only portfolio |

---

## 16. Appendix: Worked Example (golden-fixture style)

Symbol `XYZ`, decision date `t`: `c_t = 105.0`, `ATR14_t = 2.0`.
- `don20`: `L = 96.0`, `U = 103.0`. Price is above the upper wall:
  `φ = max(96 - 105, 105 - 103) / 2 = max(-9, 2) / 2 = 1.0` → outside, a breakout of 1 ATR.
  `pos = (105 - 96) / (103 - 96) = 1.2857` → above the upper wall (`> 1`).
  `width = 7 / 2 = 3.5`.
- `bb20`: `L = 99.0`, `U = 109.0`: `φ = max(-6, -4) / 2 = -2.0` → inside, 2 ATR from the nearest wall (the upper one). `pos = 0.6`, `width = 5.0`.
- `sdf_n_up_breach` counts only `don20` in this example (assuming the other corridors are inside): `1`.
- `sdf_nearest_k`: `|φ_don20| = 1.0` and `|φ_bb20| = 2.0`, so the nearest is `don20` → `0` (assuming the other corridors have larger magnitudes).
- Label example: `open[t+1] = 105.5`, `upper = 105.5 + 3.0 = 108.5`, `lower = 102.5`. If session `t+3` has `high = 109` and `low = 104` (and no earlier session touched a barrier), then `y_tb = +1`, `exit_px = 108.5`, `ret_fwd = ln(108.5/105.5) ≈ 0.0280`.
