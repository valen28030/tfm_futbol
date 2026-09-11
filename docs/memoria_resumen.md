# Resumen técnico del sistema de scouting

## Problema
Identificar jugadores cuyo perfil estadístico encaje con un estilo de juego definido por el cuerpo técnico, de forma reproducible.

## Enfoque
1. Integrar métricas ofensivas, creativas y defensivas (per 90).
2. Definir estilos como vectores de pesos.
3. Rankear por similitud coseno perfil↔estilo.
4. Complementar con KNN (similares), K-Means (perfiles latentes) y PCA (visualización).

## Pipeline
`scraping/fbref.py` → `preprocessing/clean.py` → `models/train.py` (MLflow) → `app/streamlit_app.py`

## Evaluación interna
- Diversidad entre tops de estilos (baja solapación Jaccard).
- Silhouette del clustering K-Means.
- Validación cualitativa en la app (radars + ranking).

## Limitaciones
- FBref limita scraping (CAPTCHA); parte del volumen es sintético ampliado.
- El “presupuesto” es una proxy heurística, no valor de mercado real.
- Los estilos son simplificaciones cuantitativas del juego.

## Extensiones futuras
- Wyscout/InStat, valor de mercado, más ligas, feedback del scouting en MLflow.
