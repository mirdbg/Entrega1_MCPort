
import numpy as np, pandas as pd
from mcport import drawdowns, var_cvar

def test_drawdowns_non_positive():
    s = pd.Series([1,1.1,1.05,1.2,1.15])
    dd = drawdowns(s)
    assert (dd <= 0).all()

def test_var_cvar_returns_float():
    r = pd.Series(np.random.normal(0, 0.01, size=1000))
    v, c = var_cvar(r, alpha=0.05)
    assert isinstance(v, float) and isinstance(c, float)
