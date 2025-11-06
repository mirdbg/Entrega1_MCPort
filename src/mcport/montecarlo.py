from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

# -------------------------
# Helpers de normalización
# -------------------------

def _as_price_df(obj) -> pd.DataFrame:
    """
    Normaliza cualquier input a DataFrame con UNA columna 'price' e índice DatetimeIndex.
    Soporta:
      - objetos con .to_price_dataframe() -> DataFrame('price')
      - objetos estilo PriceSeries con .data (DataFrame/Series)
      - pandas DataFrame/Series
    """
    # 1) Protocolo general
    if hasattr(obj, "to_price_dataframe"):
        df = obj.to_price_dataframe()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("to_price_dataframe() debe devolver un pandas.DataFrame.")
        if "price" not in df.columns:
            if "close" in df.columns:
                df = df.rename(columns={"close": "price"})
            else:
                raise ValueError("to_price_dataframe(): falta columna 'price'.")
        return df[["price"]]

    # 2) Estilo PriceSeries (.data -> DF/Series)
    if hasattr(obj, "data"):
        data = getattr(obj, "data")
        if isinstance(data, pd.DataFrame):
            if "price" in data.columns:
                return data[["price"]]
            if "close" in data.columns:
                return data.rename(columns={"close": "price"})[["price"]]
            if data.shape[1] == 1:
                return data.rename(columns={data.columns[0]: "price"})
            raise ValueError("PriceSeries.data debe tener 'price'/'close' o 1 columna.")
        if isinstance(data, pd.Series):
            return data.to_frame(name="price")
        raise TypeError("PriceSeries.data debe ser DataFrame o Series.")

    # 3) Pandas puros
    if isinstance(obj, pd.DataFrame):
        if "price" in obj.columns:
            return obj[["price"]]
        if "close" in obj.columns:
            return obj.rename(columns={"close": "price"})[["price"]]
        if obj.shape[1] == 1:
            return obj.rename(columns={obj.columns[0]: "price"})
        raise ValueError("DataFrame debe tener 'price'/'close' o 1 columna.")
    if isinstance(obj, pd.Series):
        return obj.to_frame(name="price")

    raise TypeError("Objeto no compatible: implemente to_price_dataframe() o pase DataFrame/Series/PriceSeries-like.")


def _label_of(obj, fallback: str = "DESCONOCIDO") -> str:
    for attr in ("label", "symbol", "name"):
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                if val:
                    return str(val)
            except Exception:
                pass
    return fallback or obj.__class__.__name__


# -------------------------
# MonteCarlo compatible
# -------------------------


@dataclass
class MonteCarloSimulation:
    """
    Monte Carlo (GBM discreto) para un activo o una cartera.

    Acepta:
      - PriceSeries (con .data y .symbol)
      - Portfolio (con .positions y .weights)
      - pandas DataFrame/Series
    """
    price_series: object                     # PriceSeries o Portfolio
    symbol: Optional[str] = None
    simulate_individual: bool = False        # Solo relevante si es Portfolio
    _last_sim: np.ndarray | None = None
    _price_df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        """Normaliza el input a DataFrame con columna 'price'."""
        if self.symbol is None:
            self.symbol = _label_of(self.price_series, fallback="DESCONOCIDO")

        # Si es Portfolio y no se quiere simular individualmente:
        if hasattr(self.price_series, "positions") and not self.simulate_individual:
            # Convertimos el Portfolio en una serie única de valor total
            df = self.price_series.value_series(initial_capital=1.0).to_frame(name="price")
        else:
            # Para PriceSeries o simulación individual
            df = _as_price_df(self.price_series)

        if df.empty:
            raise ValueError("❌ La serie está vacía. Proporcione precios.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("❌ El índice debe ser pandas.DatetimeIndex.")
        if df["price"].isna().all():
            raise ValueError("❌ Todos los valores de 'price' son NaN.")

        self._price_df = df.sort_index()

    # ---------- Retornos ----------
    def _log_returns(self) -> pd.Series:
        """Usa log_returns() si el objeto lo tiene (PriceSeries o Portfolio)."""
        if hasattr(self.price_series, "log_returns"):
            try:
                r = self.price_series.log_returns()
                if isinstance(r, pd.Series) and not r.empty:
                    return r.dropna()
            except Exception:
                pass
        s = self._price_df["price"].astype("float64")
        return np.log(s).diff().dropna()

    # ---------- Simulación ----------
    def monte_carlo(
        self,
        days: int = 252,
        n_sims: int = 2000,
        seed: Optional[int] = 123,
        start_price: Optional[float] = None,
    ) -> np.ndarray:

        rng = np.random.default_rng(seed)

        # CASO 1: Simulación individual por activo (solo para Portfolio)
        if hasattr(self.price_series, "positions") and self.simulate_individual:
            assets = self.price_series.positions
            weights = np.array(self.price_series.weights)
            sims = []

            for ps, w in zip(assets, weights):
                r = ps.log_returns()
                mu, sigma = r.mean(), r.std(ddof=1)
                price0 = float(ps.data["price"].iloc[-1])
                z = rng.standard_normal((days, n_sims))
                increments = (mu - 0.5 * sigma**2) + sigma * z
                paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(increments, axis=0)])
                prices = price0 * np.exp(paths)
                sims.append(w * prices)  # Pondera por peso

            portfolio_prices = np.sum(sims, axis=0)
            self._last_sim = portfolio_prices
            return portfolio_prices

        # CASO 2: Simulación agregada (activo único o cartera convertida)
        price0 = float(self._price_df["price"].iloc[-1]) if start_price is None else float(start_price)
        r = self._log_returns()
        if r.empty:
            raise ValueError("❌ Historial insuficiente para calcular retornos.")
        mu = r.mean()
        sigma = r.std(ddof=1)
        if np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
            raise ValueError("❌ Deriva/volatilidad no válidas.")

        z = rng.standard_normal((days, n_sims))
        increments = (mu - 0.5 * sigma**2) + sigma * z
        paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(increments, axis=0)])
        prices = price0 * np.exp(paths)
        self._last_sim = prices
        return prices

    # ---------- Final y resumen ----------
    def final_values(self, prices: np.ndarray) -> np.ndarray:
        if prices.ndim != 2:
            raise ValueError("❌ Se esperaba array 2D (days+1, n_sims).")
        return prices[-1, :]

    def simulate_and_summarize(
        self,
        days: int = 252,
        n_sims: int = 2000,
        seed: Optional[int] = 123,
        start_price: Optional[float] = None,
        percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
    ) -> dict:
        prices = self.monte_carlo(days=days, n_sims=n_sims, seed=seed, start_price=start_price)
        finals = self.final_values(prices)
        summary = {f"p{p}": float(np.percentile(finals, p)) for p in percentiles}
        return {"prices": prices, "finals": finals, "summary": summary}
