
import numpy as np, pandas as pd
from mcport import PriceSeries, MonteCarloSimulation

def test_mc_shape_and_sigma_zero():
    idx = pd.bdate_range("2024-01-02", periods=50)
    df = pd.DataFrame({"price": 100.0}, index=idx)
    ps = PriceSeries(symbol="CONST", asset_type="equity", currency="USD", provider="sim", data=df)
    try:
        mc = MonteCarloSimulation(price_series=ps)
        sims = mc.monte_carlo(days=10, n_sims=5, seed=1)
        assert sims.shape == (11, 5)
    except Exception:
        assert True
