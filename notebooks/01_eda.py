# EDA rápido del dataset de scouting
# Ejecutar tras: python -m src.pipeline

import pandas as pd
from pathlib import Path

players = pd.read_parquet(Path("data/processed/players.parquet"))
print(players.shape)
print(players["league"].value_counts())
print(players["pos_main"].value_counts())
players[["gls_p90","ast_p90","int_p90","tklw_p90","crs_p90"]].describe()
