# Módulo 9 — Evidencias para la entrega

Checklist de capturas / enlaces a aportar al máster.

## 1. Repositorio
- [ ] URL del repo: https://github.com/valen28030/tfm_futbol
- [ ] Captura de la estructura de carpetas en GitHub
- [ ] Captura del README

## 2. Process mining
- [ ] Captura o PDF/export de `docs/modulo9_process_mining.md` (AS-IS / TO-BE / KPIs)

## 3. Cloud
- [ ] Documento `docs/modulo9_cloud_estimacion.md` con costes rellenados
- [ ] (Si lo piden) Captura de la calculadora oficial del máster con t3.medium / B2s / 50 GB

## 4. MLflow
Cómo reproducir y capturar:

```powershell
cd "...\TFM\tfm_futbol"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe src\models\train_mlflow_demo.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Capturas recomendadas:
- [ ] Lista de experimentos (`tfm-scouting-recomendacion`)
- [ ] Detalle del run (`knn_cosine_demo_v1`) con parámetros
- [ ] Métrica `mean_neighbor_distance_sample`

## 5. Recordatorio
No subir `mlflow.db` ni `.venv` a GitHub (ya están en `.gitignore`).
