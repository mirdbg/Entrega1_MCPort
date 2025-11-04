import numpy as np
import pandas as pd
from mcport import PriceSeries, MonteCarloSimulation

def test_mc_shapes():
    idx = pd.bdate_range("2023-01-02", periods=260)
    df = pd.DataFrame({"price": np.linspace(100, 120, len(idx))}, index=idx)
    ps = PriceSeries(symbol="A", asset_type="equity", currency="USD", provider="sim", data=df)
    mc = MonteCarloSimulation(price_series=ps)
    paths = mc.monte_carlo(days=10, n_sims=20, seed=1)
    assert paths.shape == (11, 20)
    assert np.isfinite(paths).all()
