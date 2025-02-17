# NBA Points Prediction App

Este proyecto utiliza inteligencia artificial para predecir los puntos que un jugador de la NBA anotará en un partido específico contra un equipo determinado. La aplicación incluye un frontend interactivo desarrollado con **Streamlit**, que permite a los usuarios seleccionar un jugador, un equipo rival y obtener predicciones basadas en datos históricos.

## 🚀 Características principales

- **Predicción personalizada**: Selecciona uno de los cinco jugadores disponibles, el equipo rival, días de descanso, día de la semana y si juegea en casa para obtener la predicción de puntos para ese partido.
- **Frontend interactivo**: Interfaz sencilla y accesible gracias a **Streamlit**.
- **Descarga y enriquecimiento de datos**: Descarga actualizada de los partidos de las últimas tres temporadas e exctracción de parámetros.
- **Análisis basado en datos**: Utiliza los CSV generados con datos históricos de rendimiento de los jugadores.

---

## 📂 Estructura del proyecto

```
ia_project_2/
├── .streamlit/
│   └── config.toml           # Configuración Streamlit
├── data/
│   └── player-name.csv       # Datos históricos de puntos
├── models/
│   └── model-lstm-player-name.h5   # ModeloLSTM
│   └── model-mlp-player-name.h5    # ModeloMLP
├── functions.py               # Funciones de descarga y extracción
├── get_games.py               # Pipeline de descarga de datos
├── model-hybrid.py            # Modelo híbrido LSTM y MLP
├── prediction.py              # Pipeline de predicción híbrida
├── README.md                  # Descripción del proyecto
├── requirements.txt           # Dependencias del proyecto
├── streamlit_app.py           # Código de la aplicación Streamlit
├── style.css                  # Hoja de estilos de Streamlit
├── teams.py                   # Equipos de la NBA, nombres e IDs.
```

---

## 📊 Dataset

El dataset debe contener al menos las siguientes columnas:

- `OPPONENT_ID`: ID del rival.
- `WEEK_DAY`: Día de la semana.
- `REST_DAYS`: Días de descanso.
- `HOME`: Local o visitante.
- `PPG`: Puntos anotados.

### Ejemplo de datos:
| OPPONENT_ID  | WEEK_DAY | REST_DAYS | HOME | PPG |
|--------------|----------|-----------|------|-----|
| 42           | 2        | 1         | 1    | 35 |
| 37           | 5        | 3         | 0    | 24 |

---

## 🛠️ Tecnologías utilizadas

- **Lenguaje**: Python, HTML, CSS
- **Framework**: Streamlit
- **Librerías principales**:
  - Pandas
  - Scikit-learn
  - Matplotlib / Seaborn (para visualización)
  - Streamlit (para el frontend)

---

## 📐 Arquitectura

<img width="1131" alt="streamlit-architecture" src="https://github.com/user-attachments/assets/542f3bfc-b25b-4283-b67d-ef263e6564a4" />

## 🔗 App desplegada
[Streamlit App](https://nba-predictions-mia.streamlit.app/)

¡Gracias por visitar el proyecto! 🏀
