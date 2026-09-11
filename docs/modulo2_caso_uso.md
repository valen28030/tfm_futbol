# Caso de uso de ciencia de datos — TFM Scouting de futbolistas

## 1. Contexto y problema de negocio
Los departamentos de scouting de clubes profesionales y semiprofesionales basan gran parte de la identificación de talento en observación subjetiva. Eso dificulta comparar jugadores de forma objetiva, reproducir criterios entre analistas y alinear fichajes con el estilo de juego del equipo.

## 2. Objetivo del caso de uso
Desarrollar un sistema de recomendación que, dado un estilo de juego (vector de métricas ponderadas), una posición y filtros (edad, presupuesto estimado), proponga un ranking de jugadores cuyo perfil estadístico sea afín a ese estilo.

## 3. Usuario final
- Director deportivo / responsable de scouting
- Analista de datos deportivos del club

## 4. Pregunta de negocio
¿Qué jugadores disponibles se ajustan mejor al estilo de juego X, en la posición Y, dentro de un rango de edad y presupuesto Z?

## 5. Datos
- Fuente: FBref (y soccerdata), ligas La Liga y Segunda División (temporadas 2022-23 a 2024-25)
- Volumen estimado: 1.500–2.500 jugadores, >40 métricas por jugador
- Tipos de métricas: ofensivas (xG, xAG, progresión), presión, creación (GCA/SCA), defensivas y de posesión

## 6. Pipeline de ciencia de datos
1. Extracción (scraping / soccerdata)
2. Limpieza y estructuración (SQLite / tablas procesadas)
3. Feature engineering: normalización y vectores de estilo de juego
4. Modelado: similitud coseno, KNN, PCA y K-Means
5. Evaluación con casos de fichajes conocidos
6. Despliegue en app Streamlit (ranking + radars)

## 7. Output esperado
- Ranking de jugadores por afinidad táctica
- Visualizaciones (radar, comparativas)
- Trazabilidad del criterio (métricas y pesos del estilo)

## 8. Criterios de éxito
- Coherencia con fichajes/casos reales conocidos
- Ranking interpretable por scouting
- Prototipo usable en Streamlit
- Pipeline reproducible (código + dependencias en GitHub)

## 9. Riesgos y limitaciones
- Sesgo por liga/minutos jugados
- Fragilidad del scraping
- Falta de valor de mercado preciso en fuentes públicas
- Estilos de juego son simplificaciones cuantitativas del juego real

## 10. Relación con el TFM
Este caso de uso materializa el anteproyecto "Análisis y scouting de futbolistas para la toma de decisiones en scouting deportivo" (Iván Galán, Basma Marso, Carlos Valencia).
