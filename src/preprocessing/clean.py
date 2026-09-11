"""Limpieza y features per-90."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DB_PATH, FEATURE_COLS, MIN_MINUTES, PLAYERS_PARQUET


def _per90(series: pd.Series, nineties: pd.Series) -> pd.Series:
    n90 = nineties.replace(0, np.nan)
    return (series.fillna(0) / n90).fillna(0.0)


def clean_players(df: pd.DataFrame, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    out = df.copy()

    # Unificar columnas
    rename = {
        "Player": "player",
        "Squad": "team",
        "Pos": "pos",
        "Age": "age",
        "Nation": "nation",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    # Eliminar columnas duplicadas (p. ej. Age + age tras merge)
    out = out.loc[:, ~out.columns.duplicated()].copy()

    for col in [
        "minutes", "mp", "nineties", "gls", "ast", "ga", "sh", "sot",
        "crs", "int", "tklw", "fls", "fld", "age",
    ]:
        if col in out.columns:
            series = out[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            out[col] = pd.to_numeric(series, errors="coerce")

    if "nineties" not in out.columns or out["nineties"].isna().all():
        out["nineties"] = (out.get("minutes", pd.Series(0)).fillna(0) / 90.0).clip(lower=0.1)
    else:
        out["nineties"] = out["nineties"].fillna(out.get("minutes", 0) / 90.0).clip(lower=0.1)

    if "ga" not in out.columns:
        out["ga"] = out.get("gls", 0).fillna(0) + out.get("ast", 0).fillna(0)

    for c in ["gls", "ast", "ga", "sh", "sot", "crs", "int", "tklw", "fls", "fld", "minutes"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)

    out["gls_p90"] = _per90(out["gls"], out["nineties"])
    out["ast_p90"] = _per90(out["ast"], out["nineties"])
    out["ga_p90"] = _per90(out["ga"], out["nineties"])
    out["sh_p90"] = _per90(out["sh"], out["nineties"])
    out["sot_p90"] = _per90(out["sot"], out["nineties"])
    out["crs_p90"] = _per90(out["crs"], out["nineties"])
    out["int_p90"] = _per90(out["int"], out["nineties"])
    out["tklw_p90"] = _per90(out["tklw"], out["nineties"])
    out["fls_p90"] = _per90(out["fls"], out["nineties"])
    out["fld_p90"] = _per90(out["fld"], out["nineties"])

    out["pos_main"] = out["pos"].astype(str).str.split(",").str[0].str.strip().str.upper()
    out["budget_est"] = (
        out["age"].fillna(25).rsub(40).clip(lower=5)
        * (1 + out["ga_p90"].clip(0, 1.5))
        * np.where(out["league"].astype(str).str.contains("La Liga", case=False, na=False), 1.8, 1.0)
        * 0.5
    ).round(2)

    out = out[out["minutes"] >= min_minutes].copy()
    out = out.drop_duplicates(subset=["player", "team", "season", "league"], keep="first")
    out = out.reset_index(drop=True)
    out["player_id"] = out.index.astype(int)
    return out


def save_processed(df: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PLAYERS_PARQUET, index=False)
    df.to_csv(DATA_PROCESSED / "players.csv", index=False)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("players", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_players_pos ON players(pos_main)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_players_league ON players(league)"
    )
    conn.commit()
    conn.close()
    print(f"[preprocessing] Guardado {PLAYERS_PARQUET} y {DB_PATH} ({len(df)} jugadores)")


def load_players() -> pd.DataFrame:
    if not PLAYERS_PARQUET.exists():
        raise FileNotFoundError(
            f"No existe {PLAYERS_PARQUET}. Ejecuta: python -m src.pipeline"
        )
    return pd.read_parquet(PLAYERS_PARQUET)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].fillna(0.0)
