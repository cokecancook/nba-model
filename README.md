# NBA Points Prediction App

Este proyecto utiliza inteligencia artificial para predecir los puntos que un jugador de la NBA anotará en un partido específico contra un equipo determinado. La aplicación incluye un frontend interactivo desarrollado con **Streamlit**, que permite a los usuarios seleccionar un jugador, un equipo rival y obtener predicciones basadas en datos históricos.

## 🚀 Características principales

- **Predicción personalizada**: Introduce el nombre del jugador y el equipo rival para obtener la predicción de puntos para ese partido.
- **Frontend interactivo**: Una interfaz sencilla y accesible gracias a **Streamlit**.
- **Análisis basado en datos**: Utiliza un archivo CSV con datos históricos de rendimiento de los jugadores.

---

## 📂 Estructura del proyecto

```
ia_project_2/
├── data/
│   └── player_stats.csv       # Datos históricos de puntos de los jugadores
├── models/
│   └── trained_model.pkl      # Modelo entrenado para la predicción
├── app/
│   └── app.py                 # Código de la aplicación Streamlit
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Documentación del proyecto
└── notebooks/
    └── analysis.ipynb         # Exploración y entrenamiento del modelo
```

---

## 📊 Dataset

El dataset debe contener al menos las siguientes columnas:

- `player_name`: Nombre del jugador.
- `team_name`: Equipo del jugador.
- `opponent_team`: Equipo rival.
- `points`: Puntos anotados en el partido.
- `game_date`: Fecha del partido.

### Ejemplo de datos:
| player_name  | team_name  | opponent_team | points | game_date  |
|--------------|------------|---------------|--------|------------|
| LeBron James | Lakers     | Warriors      | 30     | 2024-01-01 |
| Kevin Durant | Suns       | Lakers        | 25     | 2024-01-02 |

---

## 🛠️ Tecnologías utilizadas

- **Lenguaje**: Python
- **Framework**: Streamlit
- **Librerías principales**:
  - Pandas
  - Scikit-learn
  - Matplotlib / Seaborn (para visualización)
  - Streamlit (para el frontend)

---

¡Gracias por visitar el proyecto! 🏀
