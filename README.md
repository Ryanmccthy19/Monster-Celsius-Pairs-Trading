# MNST–CELH Pairs Trading Strategy

A quantitative pairs-trading backtest exploring the relationship between **Monster Beverage (MNST)** and **Celsius Holdings (CELH)** using rolling regression, mean-reversion signals, and statistical testing.

## Overview

This project tests whether temporary differences in the movement of MNST and CELH can be used to create a mean-reversion trading strategy.

Historical stock prices are downloaded using `yfinance` and converted into log prices and returns. A rolling regression is then used to estimate the relationship between the two stocks and construct a hedged spread.

The strategy uses:

* A **120-day rolling OLS regression** to estimate the hedge ratio
* A **60-day rolling z-score** to identify unusual movements in the spread
* **ADF and Engle-Granger tests** to examine stationarity and cointegration
* Shifted signals to prevent **look-ahead bias**
* Estimated transaction costs to produce both gross and net performance

## Strategy

The rolling hedge ratio is estimated using:

$$
r_{MNST} = \alpha + \beta r_{CELH} + \epsilon
$$

The spread return is then calculated as:

$$
r_{spread} = r_{MNST} - \beta r_{CELH}
$$

The strategy calculates a rolling z-score of the spread to determine when the relationship has moved unusually far from its recent average.

### Trading Rules

* **Long spread:** Z-score < -2
* **Short spread:** Z-score > 2
* **Exit:** |Z-score| < 0.5

A long spread represents long MNST and short beta-adjusted CELH exposure. A short spread represents the opposite.

## Statistical Results

MNST and CELH had a historical correlation of approximately **0.86**, indicating that they frequently moved in the same direction.

However, the statistical tests did not find strong evidence of a stable mean-reverting relationship:

* **ADF p-value:** 0.5615
* **Engle-Granger cointegration p-value:** 0.8403

These results highlight an important distinction: **high correlation does not necessarily imply cointegration or mean reversion.**

## Backtest Results

The strategy completed **25 entries and 25 exits** during the backtest.

| Metric                |   Gross |     Net |
| --------------------- | ------: | ------: |
| Annualized Return     |   1.93% |   1.03% |
| Annualized Volatility |  15.55% |  15.55% |
| Sharpe Ratio          |    0.12 |    0.07 |
| Maximum Drawdown      | -27.77% | -28.63% |

Net performance includes a simple transaction-cost model of 5 basis points per leg.

## Key Takeaways

Although the strategy generated a positive return, its low Sharpe ratio and relatively large drawdown indicate weak risk-adjusted performance. Transaction costs further reduced the strategy's returns.

The project demonstrates the importance of:

* Distinguishing **correlation from cointegration**
* Testing assumptions behind mean-reversion strategies
* Avoiding look-ahead bias in backtesting
* Accounting for transaction costs
* Evaluating strategies using risk-adjusted performance

## Future Improvements

Potential improvements include testing additional stock pairs, adding out-of-sample testing, experimenting with different rolling windows and trading thresholds, and comparing the return-based hedge ratio with a traditional cointegration-based price spread.

## Technologies

**Python:** Pandas, NumPy, Matplotlib, Statsmodels, yfinance

## Disclaimer

This project is for educational and research purposes only and does not constitute financial advice.
