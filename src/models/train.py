"""Entrenamiento con tracking MLflow."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import silhouette_score

from src.config import (
    FEATURE_COLS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PLAY_STYLES,
)
from src.models.recommend import PlayerRecommender
from src.preprocessing.clean import load_players


def evaluate_style_separation(rec: PlayerRecommender) -> dict[str, float]:
    """Métrica simple: separación media entre top-10 de estilos distintos."""
    assert rec.players is not None
    tops = {}
    for key in PLAY_STYLES:
        tops[key] = set(rec.recommend(key, top_n=10)["player_id"].tolist())
    keys = list(tops)
    overlaps = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = tops[keys[i]], tops[keys[j]]
            overlaps.append(len(a & b) / max(len(a | b), 1))
    return {
        "mean_top10_jaccard_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "style_diversity": float(1.0 - np.mean(overlaps)) if overlaps else 0.0,
    }


def train_and_log(
    n_neighbors: int = 15,
    n_clusters: int = 6,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> Path:
    players = load_players()
    feature_cols = [c for c in FEATURE_COLS if c in players.columns]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="recommender_cosine_knn_kmeans"):
        mlflow.log_param("n_players", len(players))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("n_styles", len(PLAY_STYLES))
        mlflow.log_param("algoritmo", "cosine_affinity+knn+kmeans+pca")

        rec = PlayerRecommender(n_neighbors=n_neighbors, n_clusters=n_clusters)
        rec.fit(players, feature_cols=feature_cols)

        sil = float(silhouette_score(rec.X, rec.players["cluster"])) if rec.X is not None else 0.0
        metrics = evaluate_style_separation(rec)
        metrics["silhouette_kmeans"] = sil
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        path = rec.save()
        mlflow.log_artifact(str(path))

        meta = {
            "feature_cols": feature_cols,
            "styles": list(PLAY_STYLES.keys()),
            "metrics": metrics,
            "model_path": str(path),
        }
        meta_path = MODELS_DIR / "metrics.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(meta_path))

        print("[train] Modelo guardado en", path)
        print("[train] Métricas:", metrics)
        return path


if __name__ == "__main__":
    train_and_log()
