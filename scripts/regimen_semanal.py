"""Escribe local-data/m5/regimen_semanal-h2.csv: régimen semanal con peso macro 0,5 sobre etf-m5-1d-2026-10 (M5-12)."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

from quantagent.data.snapshot import load_manifest, read_snapshot, snapshot_root  # noqa: E402
from quantagent.fabrica.regimen import regimen_semanal  # noqa: E402
from quantagent.fabrica.transversal import PRIORS_M5  # noqa: E402

SNAPSHOT = "etf-m5-1d-2026-10"


def main():
    cierres = pd.DataFrame({s: read_snapshot(SNAPSHOT, s)["close"] for s in load_manifest(SNAPSHOT)["symbols"]})
    tabla = regimen_semanal(cierres, PRIORS_M5, 0.5)
    destino = snapshot_root() / "m5" / "regimen_semanal-h2.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False)
    print(f"{destino}: {len(tabla)} filas")
    for activo in ("XLRE", "XLC"):
        print(f"primera fecha con score {activo}: {tabla.loc[tabla[f'score_{activo}'].notna(), 'fecha'].min().date()}")
    print(tabla["estado"].value_counts().to_string())
    print(tabla[["fecha", "habilitados", "estado"]].tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
