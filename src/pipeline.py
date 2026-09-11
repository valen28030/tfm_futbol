"""Pipeline end-to-end: datos -> limpieza -> entrenamiento."""
from __future__ import annotations

from src.models.train import train_and_log
from src.preprocessing.clean import clean_players, save_processed
from src.scraping.fbref import build_raw_dataset


def run_all() -> None:
    raw = build_raw_dataset(use_demo_if_needed=True)
    clean = clean_players(raw)
    save_processed(clean)
    train_and_log()
    print("[pipeline] Completado.")


if __name__ == "__main__":
    run_all()
