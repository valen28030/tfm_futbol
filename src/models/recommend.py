"""Motor de recomendación: cosine, KNN y clustering."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLS, MODELS_DIR, RANDOM_SEED
from src.features.styles import affinity_scores


@dataclass
class RecommenderArtifacts:
    scaler: StandardScaler
    knn: NearestNeighbors
    kmeans: KMeans
    pca: PCA
    feature_cols: list[str]
    player_ids: np.ndarray


class PlayerRecommender:
    def __init__(self, n_neighbors: int = 15, n_clusters: int = 6, n_pca: int = 4):
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.n_pca = n_pca
        self.artifacts: RecommenderArtifacts | None = None
        self.players: pd.DataFrame | None = None
        self.X: np.ndarray | None = None

    def fit(self, players: pd.DataFrame, feature_cols: list[str] | None = None) -> "PlayerRecommender":
        feature_cols = feature_cols or [c for c in FEATURE_COLS if c in players.columns]
        self.players = players.reset_index(drop=True).copy()
        feats = self.players[feature_cols].fillna(0.0)

        scaler = StandardScaler()
        X = scaler.fit_transform(feats.values)

        knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X)), metric="cosine")
        knn.fit(X)

        kmeans = KMeans(n_clusters=min(self.n_clusters, len(X)), random_state=RANDOM_SEED, n_init=10)
        clusters = kmeans.fit_predict(X)
        self.players["cluster"] = clusters

        pca = PCA(n_components=min(self.n_pca, X.shape[1]), random_state=RANDOM_SEED)
        pcs = pca.fit_transform(X)
        for i in range(pcs.shape[1]):
            self.players[f"pc{i+1}"] = pcs[:, i]

        self.X = X
        self.artifacts = RecommenderArtifacts(
            scaler=scaler,
            knn=knn,
            kmeans=kmeans,
            pca=pca,
            feature_cols=feature_cols,
            player_ids=self.players["player_id"].to_numpy(),
        )
        return self

    def recommend(
        self,
        style_key: str,
        position: str | None = None,
        max_age: int | None = None,
        max_budget: float | None = None,
        league: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        if self.players is None or self.X is None or self.artifacts is None:
            raise RuntimeError("Modelo no entrenado")

        scores = affinity_scores(self.X, style_key, self.artifacts.feature_cols)
        result = self.players.copy()
        result["affinity"] = scores

        if position and position != "Todas":
            result = result[result["pos_main"].str.contains(position, case=False, na=False)]
        if max_age is not None:
            result = result[result["age"].fillna(99) <= max_age]
        if max_budget is not None:
            result = result[result["budget_est"].fillna(0) <= max_budget]
        if league and league != "Todas":
            result = result[result["league"] == league]

        cols = [
            "player_id", "player", "team", "league", "season", "pos", "pos_main",
            "age", "minutes", "affinity", "cluster", "budget_est",
            "gls_p90", "ast_p90", "ga_p90", "sh_p90", "int_p90", "tklw_p90", "crs_p90",
        ]
        cols = [c for c in cols if c in result.columns]
        return result.sort_values("affinity", ascending=False).head(top_n)[cols]

    def similar_players(self, player_id: int, top_n: int = 10) -> pd.DataFrame:
        assert self.artifacts and self.players is not None and self.X is not None
        idx = self.players.index[self.players["player_id"] == player_id]
        if len(idx) == 0:
            raise ValueError(f"player_id {player_id} no encontrado")
        i = int(idx[0])
        distances, indices = self.artifacts.knn.kneighbors(
            self.X[i].reshape(1, -1), n_neighbors=min(top_n + 1, len(self.X))
        )
        rows = []
        for dist, j in zip(distances[0], indices[0]):
            if j == i:
                continue
            row = self.players.iloc[j].copy()
            row["similarity"] = 1.0 - float(dist)
            rows.append(row)
        return pd.DataFrame(rows).head(top_n)

    def save(self, path: Path | None = None) -> Path:
        path = path or (MODELS_DIR / "recommender.joblib")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"recommender": self, "players": self.players}, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "PlayerRecommender":
        path = path or (MODELS_DIR / "recommender.joblib")
        payload = joblib.load(path)
        return payload["recommender"]
