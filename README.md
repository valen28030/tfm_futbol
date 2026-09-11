# TFM — Análisis y scouting de futbolistas

Sistema de recomendación de jugadores por **afinidad con estilos de juego** (vectores de métricas ponderadas), con datos FBref (caché) + dataset ampliado, modelos scikit-learn, MLflow y app Streamlit.

**Autores:** Iván Galán López, Basma Marso, Carlos Valencia Sánchez

## Qué incluye

- Pipeline de datos → SQLite / Parquet
- 6 estilos de juego (posesión, presión alta, vertical, contraataque, solidez, desborde)
- Modelos: afinidad coseno, KNN, K-Means, PCA
- Tracking MLflow
- App interactiva Streamlit (ranking, radar, similares, clusters)
- Tests básicos

## Quick start

```powershell
git clone https://github.com/valen28030/tfm_futbol.git
cd tfm_futbol
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Opción A — Usar datos/modelo ya generados (rápido)

```powershell
streamlit run src/app/streamlit_app.py
```

### Opción B — Regenerar todo el pipeline

```powershell
python -m src.pipeline
streamlit run src/app/streamlit_app.py
```

### MLflow UI

```powershell
python -m src.models.train
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre http://127.0.0.1:5000 → **Model training** → Experiments.

## Estructura

```text
tfm_futbol/
├── data/raw/              # extracción
├── data/processed/        # players.parquet, scouting.db
├── docs/                  # caso de uso, process mining, cloud, anteproyecto
├── models/                # recommender.joblib + metrics.json
├── notebooks/             # EDA
├── src/
│   ├── scraping/          # FBref caché + demo
│   ├── preprocessing/     # limpieza y per-90
│   ├── features/          # estilos de juego
│   ├── models/            # recomendación + train MLflow
│   ├── app/               # Streamlit
│   └── pipeline.py
└── tests/
```

## Estilos de juego

| Clave | Nombre |
|-------|--------|
| `posesion` | Posesión / construcción |
| `presion_alta` | Presión alta |
| `vertical` | Juego vertical / directo |
| `contraataque` | Contraataque |
| `solidez_defensiva` | Solidez defensiva |
| `desborde_banda` | Desborde / banda |

## Datos

- **FBref (caché soccerdata):** La Liga 2023-24 (standard, shooting, misc) cuando está disponible en `~/soccerdata/data/FBref`
- **Demo sintético:** completa Segunda y otras temporadas para alcanzar el volumen del anteproyecto (~1.2k jugadores tras filtro de minutos)
- FBref a veces muestra CAPTCHA; el pipeline no se bloquea gracias al modo demo

## Documentación

| Doc | Contenido |
|-----|-----------|
| [docs/modulo2_caso_uso.md](docs/modulo2_caso_uso.md) | Caso de uso |
| [docs/modulo9_process_mining.md](docs/modulo9_process_mining.md) | Process mining |
| [docs/modulo9_cloud_estimacion.md](docs/modulo9_cloud_estimacion.md) | Cloud (~35 €/mes VM) |
| [docs/modulo9_evidencias.md](docs/modulo9_evidencias.md) | Checklist evidencias |
| [docs/memoria_resumen.md](docs/memoria_resumen.md) | Resumen técnico del sistema |

## Stack

Python, pandas, scikit-learn, MLflow, Streamlit, Plotly, SQLite, soccerdata (opcional).
