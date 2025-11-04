
import numpy as np, pandas as pd
from mcport import PriceSeries, Portfolio

def _series(symbol="A", n=250):
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"price": np.linspace(100, 120, n)}, index=idx)
    return PriceSeries(symbol=symbol, asset_type="equity", currency="USD", provider="sim", data=df)

def test_priceseries_structure():
    ps = _series()
    assert "price" in ps.data.columns
    assert isinstance(ps.data.index, pd.DatetimeIndex)

def test_portfolio_value_series():
    a, b = _series("A"), _series("B")
    port = Portfolio(positions=[a, b], weights=[0.5, 0.5], name="Test", currency="USD")
    eq = port.value_series(1.0)
    assert not eq.empty
    assert eq.index.equals(a.data.index.intersection(b.data.index))
