"""Tests básicos del motor de recomendación."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PLAY_STYLES
from src.models.recommend import PlayerRecommender
from src.preprocessing.clean import load_players


def test_recommend_returns_rows():
    players = load_players()
    rec = PlayerRecommender(n_neighbors=10, n_clusters=4)
    rec.fit(players)
    style = next(iter(PLAY_STYLES))
    out = rec.recommend(style, top_n=5)
    assert len(out) > 0
    assert "affinity" in out.columns


def test_similar_players():
    players = load_players()
    rec = PlayerRecommender()
    rec.fit(players)
    pid = int(players.iloc[0]["player_id"])
    sim = rec.similar_players(pid, top_n=5)
    assert len(sim) > 0
