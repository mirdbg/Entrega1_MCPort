# mcport/__init__.py

# --- Núcleo del paquete ---
from .providers import YahooProvider, AlphaVantageProvider
from .models import PriceSeries, Portfolio     # ajusta el módulo si tus clases están en otro archivo
from .montecarlo import MonteCarloSimulation
from .providers import DataProvider, YahooProvider, AlphaVantageProvider

# --- Utils (como ya hacías) ---
from .utils import (
    to_business_days, clean_price_frame, log_returns, annualize_stats,
    sharpe_ratio, drawdowns, var_cvar
)

# --- Report---
from .reports import PortfolioReporter, PDFReportConfig

# --- Plots  ---
from .plot import (
    # PriceSeries
    plot_priceseries_history,
    plot_priceseries_drawdown,
    plot_priceseries_rolling_vol,
    plot_priceseries_rolling_return,
    plot_priceseries_returns_hist,
    # Portfolio
    plot_portfolio_equity,
    plot_portfolio_drawdown,
    plot_portfolio_weights,
    plot_portfolio_corr_heatmap,
    # Monte Carlo
    plot_mc_paths,
    plot_mc_fan,
    plot_mc_overlay_with_history,
    plot_mc_terminal_hist,
    plot_mc_terminal_cdf,
)

__all__ = [
    # Core
    "PriceSeries", "Portfolio", "MonteCarloSimulation",
    "DataProvider", "YahooProvider", "AlphaVantageProvider",

    # Utils
    "to_business_days", "clean_price_frame", "log_returns", "annualize_stats",
    "sharpe_ratio", "drawdowns", "var_cvar",

    # Report
    "PortfolioReporter", "PDFReportConfig",

    # Plots
    "plot_priceseries_history",
    "plot_priceseries_drawdown",
    "plot_priceseries_rolling_vol",
    "plot_priceseries_rolling_return",
    "plot_priceseries_returns_hist",
    "plot_portfolio_equity",
    "plot_portfolio_drawdown",
    "plot_portfolio_weights",
    "plot_portfolio_corr_heatmap",
    "plot_mc_paths",
    "plot_mc_fan",
    "plot_mc_overlay_with_history",
    "plot_mc_terminal_hist",
    "plot_mc_terminal_cdf",
]
