"""Aplicación Streamlit de scouting."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FEATURE_COLS, MODELS_DIR, PLAY_STYLES
from src.features.styles import list_styles
from src.models.recommend import PlayerRecommender
from src.preprocessing.clean import load_players


@st.cache_resource
def load_model() -> PlayerRecommender:
    path = MODELS_DIR / "recommender.joblib"
    if path.exists():
        return PlayerRecommender.load(path)
    players = load_players()
    rec = PlayerRecommender()
    rec.fit(players)
    rec.save()
    return rec


def radar_figure(player_row: pd.Series, feature_cols: list[str], title: str) -> go.Figure:
    values = [float(player_row.get(c, 0) or 0) for c in feature_cols]
    # Escala relativa simple para visualización
    labels = [c.replace("_p90", "").upper() for c in feature_cols]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            name=str(player_row.get("player", "Jugador")),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        title=title,
        margin=dict(l=40, r=40, t=50, b=40),
        height=420,
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="TFM Scouting", page_icon="⚽", layout="wide")
    st.title("Scouting de futbolistas")
    st.caption("Recomendación por afinidad con estilos de juego · TFM")

    try:
        rec = load_model()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "No hay modelo/datos. Ejecuta primero:\n\n"
            "`python -m src.pipeline`"
        )
        st.exception(exc)
        return

    styles = list_styles()
    style_labels = {k: v["label"] for k, v in styles.items()}

    with st.sidebar:
        st.header("Filtros de búsqueda")
        style_key = st.selectbox(
            "Estilo de juego",
            options=list(style_labels),
            format_func=lambda k: style_labels[k],
        )
        st.info(PLAY_STYLES[style_key]["description"])
        positions = ["Todas", "FW", "MF", "DF", "GK"]
        position = st.selectbox("Posición", positions)
        leagues = ["Todas"] + sorted(rec.players["league"].dropna().unique().tolist())
        league = st.selectbox("Liga", leagues)
        max_age = st.slider("Edad máxima", 18, 40, 32)
        max_budget = st.slider("Presupuesto estimado máx. (M€)", 1.0, 80.0, 40.0, 1.0)
        top_n = st.slider("Top N", 5, 30, 10)
        run = st.button("Buscar jugadores", type="primary")

    tab1, tab2, tab3 = st.tabs(["Recomendaciones", "Similares", "Explorar clusters"])

    with tab1:
        if run or "last_result" not in st.session_state:
            result = rec.recommend(
                style_key=style_key,
                position=None if position == "Todas" else position,
                max_age=max_age,
                max_budget=max_budget,
                league=None if league == "Todas" else league,
                top_n=top_n,
            )
            st.session_state["last_result"] = result
        else:
            result = st.session_state["last_result"]

        st.subheader(f"Ranking · {style_labels[style_key]}")
        if result.empty:
            st.warning("Sin resultados con esos filtros.")
        else:
            show = result.copy()
            show["affinity"] = (show["affinity"] * 100).round(1)
            st.dataframe(
                show.rename(
                    columns={
                        "player": "Jugador",
                        "team": "Equipo",
                        "league": "Liga",
                        "season": "Temp.",
                        "pos": "Pos",
                        "age": "Edad",
                        "affinity": "Afinidad %",
                        "budget_est": "Presupuesto est. M€",
                        "minutes": "Minutos",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            c1, c2 = st.columns(2)
            top = result.iloc[0]
            feats = [c for c in FEATURE_COLS if c in rec.players.columns]
            with c1:
                st.plotly_chart(
                    radar_figure(top, feats, f"Radar · {top['player']}"),
                    use_container_width=True,
                )
            with c2:
                fig = px.bar(
                    result.head(10),
                    x="affinity",
                    y="player",
                    orientation="h",
                    title="Top afinidad",
                    labels={"affinity": "Afinidad", "player": "Jugador"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Jugadores similares (KNN coseno)")
        options = rec.players.sort_values("player")[["player_id", "player", "team"]]
        label_map = {
            int(r.player_id): f"{r.player} ({r.team})"
            for r in options.itertuples()
        }
        selected = st.selectbox(
            "Jugador de referencia",
            options=list(label_map),
            format_func=lambda i: label_map[i],
        )
        if st.button("Buscar similares"):
            sim = rec.similar_players(selected, top_n=10)
            st.dataframe(sim[["player", "team", "pos", "age", "league", "similarity"]], hide_index=True)

    with tab3:
        st.subheader("Mapa PCA + K-Means")
        df = rec.players
        if "pc1" in df.columns and "pc2" in df.columns:
            sample = df.sample(min(800, len(df)), random_state=42)
            fig = px.scatter(
                sample,
                x="pc1",
                y="pc2",
                color="cluster",
                hover_data=["player", "team", "pos_main", "league"],
                title="Proyección PCA de perfiles",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Los clusters agrupan perfiles estadísticos similares (no posiciones puras).")
        else:
            st.info("PCA no disponible en el modelo cargado.")


if __name__ == "__main__":
    main()
