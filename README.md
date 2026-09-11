# TFM — Análisis y scouting de futbolistas

Sistema de recomendación de jugadores basado en similitud entre perfiles estadísticos y estilos de juego definidos por vectores de métricas ponderadas.

**Autores:** Iván Galán López, Basma Marso, Carlos Valencia Sánchez

## Objetivo

Ayudar a direcciones deportivas y scouting a identificar candidatos alineados con el estilo táctico del club, de forma más objetiva y reproducible que la observación puramente subjetiva.

La aplicación final (Streamlit) permitirá filtrar por posición, estilo de juego, edad y presupuesto estimado, y devolverá un ranking con visualizaciones tipo radar.

## Stack tecnológico

| Área | Herramientas |
|------|----------------|
| Lenguaje | Python |
| Datos | pandas, soccerdata / scraping FBref, SQLite |
| ML | scikit-learn (KNN, similitud coseno, PCA, K-Means) |
| Tracking | MLflow |
| Visualización / app | Plotly, mplsoccer, Streamlit |
| Control de versiones | Git / GitHub |

## Estructura del repositorio

```text
tfm_futbol/
├── data/
│   ├── raw/           # datos brutos
│   ├── processed/     # datos limpios
│   └── external/      # fuentes auxiliares
├── docs/              # documentación del TFM y módulos
├── models/            # artefactos entrenados
├── notebooks/         # EDA y prototipos
├── src/
│   ├── scraping/      # extracción de datos
│   ├── preprocessing/ # limpieza y normalización
│   ├── features/      # vectores de estilo de juego
│   ├── models/        # entrenamiento y MLflow
│   └── app/           # aplicación Streamlit
├── tests/
├── requirements.txt
└── README.md
```

## Puesta en marcha

### 1. Clonar el repositorio

```powershell
git clone https://github.com/valen28030/tfm_futbol.git
cd tfm_futbol
```

### 2. Crear y activar el entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## MLflow (demo Módulo 9)

Hay un experimento mínimo con datos sintéticos para validar el tracking:

```powershell
.\.venv\Scripts\python.exe src\models\train_mlflow_demo.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre http://127.0.0.1:5000 → **Model training** → **Experiments** → `tfm-scouting-recomendacion`.

> `mlflow.db` y `mlruns/` están en `.gitignore` (no se suben al remoto).

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [docs/modulo2_caso_uso.md](docs/modulo2_caso_uso.md) | Caso de uso de ciencia de datos |
| [docs/modulo9_process_mining.md](docs/modulo9_process_mining.md) | Problema de negocio con visión de proceso |
| [docs/modulo9_cloud_estimacion.md](docs/modulo9_cloud_estimacion.md) | Estimación de recursos cloud |

## Estado del proyecto

- [x] Repositorio y estructura de carpetas
- [x] Entorno de desarrollo y `requirements.txt`
- [x] Documentación Módulo 2 y Módulo 9
- [x] Demo MLflow
- [ ] Pipeline de scraping FBref
- [ ] Feature engineering de estilos de juego
- [ ] Modelos de recomendación con datos reales
- [ ] App Streamlit
- [ ] Evaluación con casos de fichajes conocidos

## Licencia / uso académico

Proyecto académico de Trabajo de Fin de Máster. Los datos de FBref están sujetos a los términos de uso de Sports Reference.
