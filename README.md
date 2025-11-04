# mcport — Monte Carlo & Portfolio Toolkit

Librería en **layout `src/`** para:
- **PriceSeries** y **Portfolio**
- **Monte Carlo** sobre series y carteras
- **Gráficas** seaborn/matplotlib + **PDFs** de informe

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Uso rápido
```python
import pandas as pd, numpy as np
from mcport import PriceSeries, Portfolio, MonteCarloSimulation, MonteCarloPlots, MonteCarloReport
# ... ver notebook en notebooks/Demo_MonteCarlo_Portfolio.ipynb
```
