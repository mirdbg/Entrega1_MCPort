#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
from mcport import PriceSeries, MonteCarloSimulation, MonteCarloReport

def main():
    ap = argparse.ArgumentParser(description="Generar informe PDF de Monte Carlo desde CSV")
    ap.add_argument("--infile", required=True, help="CSV con columnas: date, price (o close)")
    ap.add_argument("--symbol", default="ASSET")
    ap.add_argument("--out", default="reports/pdf/report.pdf")
    ap.add_argument("--days", type=int, default=252)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.infile, parse_dates=["date"]).set_index("date")
    if "price" not in df and "close" in df:
        df = df.rename(columns={"close":"price"})
    ps = PriceSeries(symbol=args.symbol, asset_type="equity", currency="USD", provider="csv", data=df[["price"]])
    mc = MonteCarloSimulation(price_series=ps)
    summ = mc.simulate_and_summarize(days=args.days, n_sims=args.sims, seed=args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rep = MonteCarloReport(mc)
    rep.to_pdf(summ["prices"], str(out))
    print("Reporte guardado en", out.resolve())

if __name__ == "__main__":
    main()
