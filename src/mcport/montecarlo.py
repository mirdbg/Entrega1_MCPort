from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class MonteCarloSimulation:
    """
    Monte Carlo (GBM multivariante) aplicado a un Portfolio.

    - Usa los retornos logarítmicos históricos de los activos del portfolio
      para estimar las medias, volatilidades y correlaciones.
    - Simula trayectorias correlacionadas usando descomposición de Cholesky.
    - Calcula métricas finales como VaR y CVaR.
    """

    portfolio: object                  # objeto tipo Portfolio
    days: int = 252                    # horizonte en días hábiles
    n_sims: int = 2000                 # número de simulaciones
    seed: int = 123                    # semilla RNG
    capital_inicial: float = 1000.0    # capital inicial total
    correlate_assets: bool = True      # si False: shocks independientes

    # Atributos internos
    _last_paths: Optional[np.ndarray] = field(init=False, default=None)  # (sims, days+1, n_assets)
    _last_values: Optional[np.ndarray] = field(init=False, default=None) # (sims, days+1)
    _last_summary: Optional[Dict[str, Any]] = field(init=False, default=None)

    # --------------------------
    # MÉTODOS PRINCIPALES
    # --------------------------
    def _prepare_inputs(self) -> Dict[str, Any]:
        """Prepara precios, retornos, medias, vols y correlaciones."""
        port = self.portfolio
        df_prices = port.aligned_prices()
        if df_prices.empty:
            raise ValueError("❌ Portfolio sin precios históricos válidos.")

        # Retornos logarítmicos
        rets = np.log(df_prices).diff().dropna()
        if rets.empty:
            raise ValueError("❌ Portfolio sin retornos suficientes.")

        # Medias y volatilidades diarias
        mu_d = rets.mean().to_numpy(float)
        sigma_d = rets.std(ddof=1).to_numpy(float)

        # Correlación o matriz identidad
        if self.correlate_assets:
            corr = rets.corr().to_numpy(float)
        else:
            corr = np.eye(len(mu_d), dtype=float)

        # Covarianza y Cholesky
        cov_d = np.outer(sigma_d, sigma_d) * corr
        try:
            L = np.linalg.cholesky(cov_d)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov_d + 1e-10 * np.eye(len(mu_d)))

        # Últimos precios y unidades
        last_prices = df_prices.iloc[-1].to_numpy(float)
        weights = np.array(port.weights, dtype=float)
        units = (weights * self.capital_inicial) / last_prices

        return {
            "prices": df_prices,
            "rets": rets,
            "mu_d": mu_d,
            "sigma_d": sigma_d,
            "corr": corr,
            "cov_d": cov_d,
            "L": L,
            "last_prices": last_prices,
            "units": units,
            "weights": weights
        }

    def monte_carlo(self) -> Dict[str, Any]:
        """Ejecuta la simulación Monte Carlo GBM para todos los activos."""
        rng = np.random.default_rng(self.seed)
        p = self._prepare_inputs()

        n_assets = len(p["mu_d"])
        tray = np.zeros((self.n_sims, self.days + 1, n_assets))
        tray[:, 0, :] = p["last_prices"]

        drift = p["mu_d"] - 0.5 * p["sigma_d"]**2

        for s in range(self.n_sims):
            prices = p["last_prices"].copy()
            for t in range(1, self.days + 1):
                z = rng.standard_normal(n_assets)
                shocks = p["L"] @ z  # correlacionados
                prices = prices * np.exp(drift + shocks)
                tray[s, t, :] = prices

        valores = np.dot(tray, p["units"])  # valor total por sim/día
        self._last_paths = tray
        self._last_values = valores
        self._inputs = p
        return {"trayectorias": tray, "valores": valores, "inputs": p}

    def summarize(self) -> Dict[str, Any]:
        """Calcula métricas finales: retorno, media, varianza, VaR, CVaR, etc."""
        if self._last_values is None:
            raise RuntimeError("Primero ejecuta `monte_carlo()`.")

        vals = self._last_values
        capital0 = float(vals[0, 0])
        final_values = vals[:, -1]
        returns = (final_values - capital0) / capital0

        mu_ann = float(self._inputs["mu_d"].mean() * 252)
        sig_ann = float(self._inputs["sigma_d"].mean() * np.sqrt(252))
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean())

        summary = {
            "capital_inicial": self.capital_inicial,
            "mean_final_value": float(final_values.mean()),
            "median_final_value": float(np.median(final_values)),
            "std_final_value": float(final_values.std(ddof=1)),
            "mean_return": float(returns.mean()),
            "VaR_95": var_95,
            "CVaR_95": cvar_95,
            "mu_annualized": mu_ann,
            "sigma_annualized": sig_ann,
        }

        self._last_summary = summary
        return summary

    def simulate_and_summarize(self) -> Dict[str, Any]:
        """Ejecuta la simulación completa y devuelve resultados + resumen."""
        res = self.monte_carlo()
        summary = self.summarize()
        return {
            "trayectorias": res["trayectorias"],
            "valores": res["valores"],
            "inputs": res["inputs"],
            "summary": summary,
        }
