# optimal-portfolio-allocator

A portfolio-optimization project for Applied Mathematics showcasing **Markowitz portfolio theory**, convex optimization, and risk-return tradeoff analysis.

## Project Structure

```
optimal-portfolio-allocator/
├── notebooks/
│   └── Portfolio_Optimization.ipynb
├── src/
│   └── portfolio.py
├── data/
├── outputs/
├── README.md
└── requirements.txt
```

## Markowitz Portfolio Theory

Markowitz mean-variance optimization models a portfolio with weights vector \(w\) over assets with expected returns vector \(\mu\) and covariance matrix \(\Sigma\).

- Portfolio expected return: \(\mathbb{E}[R_p] = w^T\mu\)
- Portfolio variance: \(\sigma_p^2 = w^T\Sigma w\)
- Portfolio volatility: \(\sigma_p = \sqrt{w^T\Sigma w}\)

### Optimization problems used

1. **Global minimum variance portfolio**

\[
\min_w\; w^T\Sigma w
\]
subject to
\[
\sum_i w_i = 1,\quad w_i \ge 0
\]

2. **Target-return minimum-risk portfolio**

\[
\min_w\; w^T\Sigma w
\]
subject to
\[
\sum_i w_i = 1,\quad w^T\mu = r_{target},\quad w_i \ge 0
\]

3. **Maximum Sharpe ratio portfolio**

\[
\max_w\; \frac{w^T\mu - r_f}{\sqrt{w^T\Sigma w}}
\]
subject to
\[
\sum_i w_i = 1,\quad w_i \ge 0
\]

## What the Notebook Demonstrates

`notebooks/Portfolio_Optimization.ipynb`:

- Downloads 5 years of daily data for 8 stocks: `AAPL, GOOGL, MSFT, TSLA, JNJ, XOM, PG, V`
- Computes daily returns, expected returns vector \(\mu\), covariance matrix \(\Sigma\)
- Solves:
  - Global minimum variance portfolio
  - Maximum Sharpe ratio portfolio
  - Target-return risk-minimizing portfolio
- Generates 1000 random feasible portfolios for efficient frontier visualization
- Produces plots for:
  - Efficient frontier (risk vs return)
  - Highlighted minimum-variance and max-Sharpe points
  - Optimal allocation bar chart
  - Individual stock risk-vs-return map
- Compares optimized portfolios vs equal-weight (1/N) strategy
- Prints summary statistics including correlation matrix, top performers, and risk comparison

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run notebook:

```bash
cd notebooks
jupyter notebook Portfolio_Optimization.ipynb
```

3. Optional module use in scripts:

```python
from src.portfolio import Portfolio
```

## Interpreting Results

- **Higher return with lower risk** is generally preferred.
- **Efficient frontier** represents feasible portfolios under long-only constraints.
- **Max Sharpe** portfolio optimizes risk-adjusted return for a chosen risk-free rate.
- **Target-return** optimization helps choose the least-risk portfolio that meets a return objective.

## Limitations

- Mean and covariance estimates are historically based and unstable over regime shifts.
- Transaction costs, taxes, slippage, and liquidity are ignored.
- Long-only assumption excludes short-selling and leverage.
- Risk is modeled only by variance; downside-risk alternatives are not included.
- Expected returns are sample averages, which can be noisy.

## Future Improvements

- Black-Litterman prior-adjusted expected returns
- Robust optimization under parameter uncertainty
- Rolling-window backtests and walk-forward validation
- Cardinality/turnover constraints for practical implementation
- Alternative risk measures (CVaR, downside semivariance)

## References

- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Luenberger, D. G. (1997). *Investment Science*. Oxford University Press.

## Disclaimer

This project is for educational purposes and should not be treated as financial advice.
