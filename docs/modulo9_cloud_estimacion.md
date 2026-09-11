# Módulo 9 — Estimación de recursos cloud

## 1. Premisas de carga
- Dataset pequeño: ~1.500–2.500 jugadores, >40 métricas
- Modelos ligeros: scikit-learn (similitud coseno, KNN, PCA, K-Means)
- App: Streamlit (pocos usuarios concurrentes en fase TFM/demo)
- Experiment tracking: MLflow local o en la misma VM
- No se requiere Big Data (Spark) ni GPU

## 2. Arquitectura propuesta (fase TFM)
1 máquina virtual cloud (o local) que concentre:
- Pipeline de datos / entrenamiento
- MLflow Tracking Server (file store)
- App Streamlit

Opcional futuro: almacenamiento de objetos (S3/Blob) para datos raw y artefactos.

## 3. Recursos estimados (usar en la calculadora del máster)

| Recurso | Especificación orientativa | Justificación |
|---------|----------------------------|---------------|
| Compute (VM) | 2–4 vCPU, 8 GB RAM | Suficiente para pandas + sklearn + Streamlit |
| Disco | 30–50 GB SSD | SO + venv + datos + mlruns + modelos |
| Red / egress | Baja (< 10 GB/mes) | Demo académica, poco tráfico |
| GPU | No | No hay deep learning |
| BD gestionada | No necesaria | SQLite basta en esta fase |
| Orquestación | No (sin Kubernetes) | Un solo servicio |

## 4. Coste orientativo (rellenar con la calculadora oficial)
- Proveedor elegido: _________________ (AWS / Azure / GCP)
- Región: _________________
- Coste mensual estimado VM: _______ €
- Coste almacenamiento: _______ €
- Coste total mensual estimado: _______ €
- Coste anual estimado: _______ €

> Instrucción: abrir la calculadora del módulo, seleccionar una VM equivalente (p. ej. AWS t3.medium / Azure B2s / GCP e2-standard-2) y anotar aquí los números reales.

## 5. Escenarios
### A) Desarrollo TFM (recomendado ahora)
- Local + GitHub
- Coste cloud ≈ 0 €
- MLflow en `./mlruns`

### B) Demo en cloud
- 1 VM pequeña 24/7 o encendida solo en demos
- Coste bajo (orden de magnitud: pocos euros/decenas al mes según proveedor y horas)

### C) Producto escalable (futuro)
- Contenedor Streamlit (Cloud Run / App Service)
- Object storage para datos
- MLflow en servidor dedicado o managed
- Posible BD (PostgreSQL) si crece el volumen

## 6. Conclusión
Para el volumen y algoritmos del TFM, **no hace falta un cluster**. Una VM pequeña (o entorno local) cubre entrenamiento, MLflow y la app. La calculadora sirve para justificar el escenario B ante el máster y planificar coste si se despliega la demo.
