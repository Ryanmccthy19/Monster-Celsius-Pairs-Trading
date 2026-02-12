import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import os

# =========================
# CONFIG
# =========================
TICKERS = ["MNST", "CELH"]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
FILE = "mnstr_celh_prices.csv"

BETA_WINDOW = 120   # rolling window to estimate hedge ratio
Z_WINDOW = 60       # rolling window for z-score stats

ENTRY_Z = 2.0
EXIT_Z = 0.5

# Simple transaction cost model
# Example: 5 bps per leg per "turn" (entry/exit/flip), 2 legs => 10 bps per turn
COST_PER_LEG = 0.0005
COST_PER_TURN = 2 * COST_PER_LEG

ANN_FACTOR = 252

# =========================
# DATA LOAD
# =========================
if os.path.exists(FILE):
    data = pd.read_csv(FILE, index_col=0, parse_dates=True)
else:
    data = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )["Close"]
    data.to_csv(FILE)

data = data.sort_index().dropna()

# Use log prices for plotting & cointegration tests
log_prices = np.log(data)

plt.figure(figsize=(10, 5))
plt.plot(log_prices["MNST"], label="MNST")
plt.plot(log_prices["CELH"], label="CELH")
plt.legend()
plt.title("Log Prices")
plt.show()

# =========================
# RETURNS (for hedge ratio + P&L consistency)
# =========================
r_mnst = log_prices["MNST"].diff()
r_celh = log_prices["CELH"].diff()

# Align / drop initial NaNs
rets = pd.DataFrame({"MNST": r_mnst, "CELH": r_celh}).dropna()
r_mnst = rets["MNST"]
r_celh = rets["CELH"]

# =========================
# ROLLING HEDGE RATIO (beta) ON RETURNS
# =========================
rolling_beta = pd.Series(index=rets.index, dtype=float)

for i in range(BETA_WINDOW, len(rets)):
    y_win = r_mnst.iloc[i - BETA_WINDOW:i]
    x_win = r_celh.iloc[i - BETA_WINDOW:i]

    X_win = sm.add_constant(x_win)
    model_win = sm.OLS(y_win, X_win).fit()

    # x_win.name should be "STZ"
    rolling_beta.iat[i] = model_win.params[x_win.name]

# Use yesterday's beta for today's trading
beta_used = rolling_beta.shift(1)

print(f"Rolling beta mean (used): {beta_used.mean():.4f}")

plt.figure(figsize=(10, 4))
plt.plot(beta_used, label="Rolling beta (t-1)", alpha=0.8)
plt.axhline(beta_used.mean(), color="black", linestyle="--", label="Mean")
plt.legend()
plt.title("Rolling Hedge Ratio (Beta) — estimated on returns")
plt.show()

# =========================
# SPREAD RETURN + SYNTHETIC SPREAD LEVEL
# =========================
spread_ret = r_mnst - beta_used * r_celh
spread_ret = spread_ret.dropna()

# Build a synthetic "spread level" by cumulating spread returns
spread = spread_ret.cumsum()

plt.figure(figsize=(10, 4))
plt.plot(spread, label="Synthetic Spread (cumulated spread returns)", alpha=0.7)
plt.axhline(spread.mean(), color="black", linestyle="--", label="Mean")
plt.legend()
plt.title("Synthetic Spread Level (consistent with returns-beta)")
plt.show()

# =========================
# STATIONARITY / COINTEGRATION CHECKS
# =========================
def print_adf(label, series):
    series = series.dropna()
    stat, pval, *_ = adfuller(series)
    print(f"{label}")
    print(f"  ADF statistic: {stat:.4f}")
    print(f"  p-value: {pval:.4f}\n")

print_adf("ADF (Synthetic spread level)", spread)

# Engle–Granger cointegration test on log price levels (informational)
# (If p-value is high, pair may not be cointegrated in levels.)
try:
    score, pval, _ = coint(log_prices["MNST"].dropna(), log_prices["CELH"].dropna())
    print("Engle–Granger cointegration test (log price levels)")
    print(f"  p-value: {pval:.4f}\n")
except Exception as e:
    print("Cointegration test failed:", e, "\n")

# =========================
# Z-SCORE (ROLLING) + SHIFTED SIGNALS (NO LOOKAHEAD)
# =========================
spread_mean = spread.rolling(Z_WINDOW).mean()
spread_std = spread.rolling(Z_WINDOW).std()
z = (spread - spread_mean) / spread_std

# Use only information known at close of t-1 to decide t actions
z_signal = z.shift(1)

plt.figure(figsize=(10, 4))
plt.plot(z, label="z")
plt.axhline(ENTRY_Z, color="red", linestyle="--")
plt.axhline(-ENTRY_Z, color="red", linestyle="--")
plt.axhline(0, color="black")
plt.title("Spread Z-Score")
plt.legend()
plt.show()

signals = pd.DataFrame(index=z.index)
signals["long"] = z_signal < -ENTRY_Z
signals["short"] = z_signal > ENTRY_Z
signals["exit"] = z_signal.abs() < EXIT_Z

# =========================
# POSITION STATE MACHINE (ENTER/EXIT/FLIP)
# =========================
pos = pd.Series(0, index=signals.index, dtype=int)

for i in range(1, len(pos)):
    prev = pos.iat[i - 1]
    pos.iat[i] = prev  # carry forward by default

    # exit if in a position and exit condition hits
    if prev != 0 and signals["exit"].iat[i]:
        pos.iat[i] = 0

    # only enter if flat
    elif prev == 0:
        if signals["long"].iat[i]:
            pos.iat[i] = 1
        elif signals["short"].iat[i]:
            pos.iat[i] = -1

signals["position"] = pos

# We execute P&L on day t using position decided at end of day t-1
pos_trade = pos.shift(1).fillna(0)
signals["position_trade"] = pos_trade

# =========================
# TRADE COUNTS (FIXED)
# =========================
entries = ((pos != 0) & (pos.shift(1) == 0)).sum()
exits = ((pos == 0) & (pos.shift(1) != 0)).sum()
flips = ((pos != 0) & (pos.shift(1) != 0) & (pos != pos.shift(1))).sum()
changes = (pos != pos.shift(1)).sum()

print("Number of long signals:", int(signals["long"].sum()))
print("Number of short signals:", int(signals["short"].sum()))
print("Number of exit signals:", int(signals["exit"].sum()))
print("Days long spread:", int((signals["position"] == 1).sum()))
print("Days short spread:", int((signals["position"] == -1).sum()))
print(f"Entries: {int(entries)} | Exits: {int(exits)} | Flips: {int(flips)} | Position changes: {int(changes)}")

# =========================
# STRATEGY RETURNS (GROSS + NET)
# =========================
strategy_ret_gross = pos_trade * spread_ret
strategy_ret_gross = strategy_ret_gross.dropna()

# Turnover: any day position changes (entry/exit/flip) => apply one "turn" cost
turnover = (pos_trade.diff().abs().fillna(0) > 0).astype(int)
turnover = turnover.reindex(strategy_ret_gross.index).fillna(0).astype(int)

strategy_ret_net = strategy_ret_gross - turnover * COST_PER_TURN

# =========================
# EQUITY CURVES
# =========================
equity_gross = np.exp(strategy_ret_gross.cumsum())
equity_net = np.exp(strategy_ret_net.cumsum())

plt.figure(figsize=(10, 4))
plt.plot(equity_gross, label="Gross")
plt.plot(equity_net, label="Net")
plt.axhline(1, color="black", linestyle="--", linewidth=1)
plt.title("Pairs Strategy Equity Curve (Rolling Beta on Returns)")
plt.legend()
plt.show()

# =========================
# PERFORMANCE STATS
# =========================
def perf_stats(r, label):
    r = r.dropna()
    ann_return = np.exp(r.mean() * ANN_FACTOR) - 1
    ann_vol = r.std() * np.sqrt(ANN_FACTOR)
    sharpe = (r.mean() / r.std()) * np.sqrt(ANN_FACTOR) if r.std() > 0 else np.nan

    eq = np.exp(r.cumsum())
    roll_max = eq.cummax()
    drawdown = eq / roll_max - 1
    max_dd = drawdown.min()

    print(label)
    print(f"  Ann return: {ann_return:.2%}")
    print(f"  Ann vol:    {ann_vol:.2%}")
    print(f"  Sharpe:     {sharpe:.2f}")
    print(f"  Max DD:     {max_dd:.2%}\n")

perf_stats(strategy_ret_gross, "GROSS")
perf_stats(strategy_ret_net, "NET (with simple costs)")

# =========================
# Z-SCORE + POSITION VISUAL
# =========================
plt.figure(figsize=(10, 4))
plt.plot(z, label="z")
plt.axhline(ENTRY_Z, linestyle="--")
plt.axhline(-ENTRY_Z, linestyle="--")
plt.axhline(0, linestyle="-")
plt.plot(signals["position"] * ENTRY_Z, alpha=0.3, label=f"position x{ENTRY_Z}")
plt.legend()
plt.title("Z-score with Positions (signals shifted, no lookahead)")
plt.show()
