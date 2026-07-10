"""Portfolio optimization utilities based on Markowitz portfolio theory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    """Container for optimization outputs."""

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    success: bool
    message: str


class Portfolio:
    """Portfolio optimizer implementing Markowitz mean-variance methods.

    Attributes:
        prices: Adjusted close prices indexed by date.
        returns: Daily arithmetic returns.
        expected_returns: Annualized expected returns vector.
        covariance_matrix: Annualized covariance matrix.
    """

    TRADING_DAYS = 252

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.expected_returns: Optional[pd.Series] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None

    def load_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical adjusted close prices with yfinance.

        Args:
            tickers: List of ticker symbols.
            start_date: Inclusive start date (YYYY-MM-DD).
            end_date: Inclusive end date (YYYY-MM-DD).

        Returns:
            DataFrame with adjusted close prices.

        Raises:
            ValueError: If no valid price data is returned.
        """
        if not tickers:
            raise ValueError("Tickers list must not be empty.")

        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            actions=False,
            group_by="ticker",
        )

        if raw.empty:
            raise ValueError("No data returned from yfinance. Check tickers and date range.")

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw.xs("Close", axis=1, level=-1)
        else:
            prices = raw.rename(columns={"Close": tickers[0]})[[tickers[0]]]

        prices = prices.sort_index().dropna(how="all")
        prices = prices.dropna(axis=1, how="all")

        missing = sorted(set(tickers) - set(prices.columns))
        if missing:
            raise ValueError(f"Missing ticker data: {', '.join(missing)}")

        self.prices = prices[tickers].copy()
        return self.prices

    def compute_statistics(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Compute returns, annualized expected returns, and annualized covariance matrix."""
        if self.prices is None:
            raise ValueError("Price data not loaded. Call load_data first.")

        returns = self.prices.pct_change().dropna(how="all")
        returns = returns.dropna(axis=1, how="any")

        if returns.empty:
            raise ValueError("Not enough data to compute returns.")

        expected_returns = returns.mean() * self.TRADING_DAYS
        covariance_matrix = returns.cov() * self.TRADING_DAYS

        self.returns = returns
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix

        return returns, expected_returns, covariance_matrix

    def _ensure_statistics(self) -> None:
        if self.expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Statistics not computed. Call compute_statistics first.")

    def _portfolio_metrics(self, weights: np.ndarray) -> tuple[float, float]:
        self._ensure_statistics()
        port_return = float(np.dot(weights, self.expected_returns.values))
        port_var = float(weights.T @ self.covariance_matrix.values @ weights)
        port_volatility = float(np.sqrt(max(port_var, 0.0)))
        return port_return, port_volatility

    def sharpe_ratio(self, weights: np.ndarray, risk_free_rate: float = 0.03) -> float:
        """Calculate annualized Sharpe ratio for provided weights."""
        ret, vol = self._portfolio_metrics(weights)
        if vol <= 1e-12:
            return -np.inf
        return (ret - risk_free_rate) / vol

    def optimize_portfolio(
        self,
        target_return: Optional[float] = None,
        method: str = "min_variance",
        risk_free_rate: float = 0.03,
    ) -> OptimizationResult:
        """Solve a constrained portfolio optimization problem.

        Args:
            target_return: Desired annualized return for risk minimization.
            method: "min_variance" or "max_sharpe".
            risk_free_rate: Annual risk-free rate for Sharpe optimization.

        Returns:
            OptimizationResult with weights and metrics.
        """
        self._ensure_statistics()

        n_assets = len(self.expected_returns)
        x0 = np.repeat(1.0 / n_assets, n_assets)
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if method == "min_variance":
            objective = lambda w: float(w.T @ self.covariance_matrix.values @ w)
            if target_return is not None:
                constraints.append(
                    {
                        "type": "eq",
                        "fun": lambda w: float(np.dot(w, self.expected_returns.values) - target_return),
                    }
                )
        elif method == "max_sharpe":
            objective = lambda w: -self.sharpe_ratio(w, risk_free_rate=risk_free_rate)
        else:
            raise ValueError("method must be 'min_variance' or 'max_sharpe'.")

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        weights = np.asarray(result.x, dtype=float)
        weights = np.clip(weights, 0.0, 1.0)
        weights = weights / weights.sum()

        exp_return, volatility = self._portfolio_metrics(weights)
        sharpe = self.sharpe_ratio(weights, risk_free_rate=risk_free_rate)

        return OptimizationResult(
            weights=weights,
            expected_return=exp_return,
            volatility=volatility,
            sharpe=sharpe,
            success=bool(result.success),
            message=str(result.message),
        )

    def compute_efficient_frontier(self, num_portfolios: int = 1000) -> pd.DataFrame:
        """Generate random feasible portfolios and compute risk/return metrics."""
        self._ensure_statistics()
        if num_portfolios <= 0:
            raise ValueError("num_portfolios must be positive.")

        rng = np.random.default_rng(self.random_seed)
        n_assets = len(self.expected_returns)

        data = []
        for _ in range(num_portfolios):
            weights = rng.dirichlet(np.ones(n_assets))
            expected_return, volatility = self._portfolio_metrics(weights)
            sharpe = self.sharpe_ratio(weights)
            data.append(
                {
                    "return": expected_return,
                    "risk": volatility,
                    "sharpe": sharpe,
                    **{f"w_{asset}": w for asset, w in zip(self.expected_returns.index, weights)},
                }
            )

        return pd.DataFrame(data)
