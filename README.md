# 🏀 NBA Performance Model

A Streamlit app for exploring and predicting NBA player performance using historical data and machine learning.

## 🚀 Main Features

- **Personalized Prediction:** Select from available players, choose the opponent team, rest days, weekday, and home/away status to get a tailored points prediction for a specific game.
- **Interactive Frontend:** Simple, accessible interface built with **Streamlit**.
- **Data Download & Enrichment:** Updated download of games from the last three seasons and extraction of relevant features.
- **Data-Driven Analysis:** Uses generated CSVs with historical player performance data.
- **Exploratory Data Analysis (EDA):**
  - Visualize player stats for the last 10 games.
  - Line chart of points scored per game (last 100 games).
  - Explore trends and patterns in player performance.
- **Machine Learning Predictions:**
  - Predict player points for upcoming games using models trained on historical NBA data (LSTM, MLP, hybrid models).
- **Caching:**
  - Fast data loading with Streamlit's caching.

---

## 📂 Project Structure

```
nba-model/
├── data/
│   └── {player}.csv               # Player data
├── images/                        # Streamlit images
│   └── architecture.svg           # Architecture diagram
│   └── logo-nba.svg               # NBA logo
│   └── nba-stars.svg              # Players picture
├── models/                        # Trained models
│   └── model-lstm-{player}.h5     # LSTM model
│   └── model-mlp-{player}.h5      # MLP model
├── .gitignore                     # Gitignore
├── functions.py                   # Preprocessing functions
├── get_games.py                   # Database generator
├── model_hybrid.py                # Training pipeline
├── README.md                      # Project description
├── requirements.txt               # Project dependencies
├── streamlit_app.py               # Streamlit app code
├── style.css                      # Custom Streamlit styles
├── teams.py                       # List of teams and abbreviations
```

---

## 📊 Dataset

The dataset should contain at least the following columns:

- `OPPONENT_ID`: Opponent team ID
- `WEEK_DAY`: Day of the week
- `REST_DAYS`: Days of rest
- `HOME`: Home (1) or Away (0)
- `PPG`: Points scored

**Example:**
| OPPONENT_ID  | WEEK_DAY | REST_DAYS | HOME | PPG |
|--------------|----------|-----------|------|-----|
| 42           | 2        | 1         | 1    | 35  |
| 37           | 5        | 3         | 0    | 24  |

---

## 🛠️ Technologies Used

- **Language:** Python, HTML, CSS
- **Framework:** Streamlit
- **Main Libraries:**
  - Pandas
  - Scikit-learn
  - Matplotlib / Seaborn (for visualization)
  - Streamlit (for frontend)

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the app:**
   ```bash
   streamlit run streamlit_app.py
   ```
3. **Open in your browser:**
   Streamlit will provide a local URL (usually [http://localhost:8501](http://localhost:8501)).

---

## 🕹️ How to Use

- Use the **sidebar** to navigate between:
  - **🏀 Introduction:** Overview and project description.
  - **🔍 EDA:** Explore player statistics, trends, and patterns. View the last 10 games and season performance for players (default: LeBron James).
  - **🔮 Prediction:** Predict future player performance using machine learning models.
- The app loads data from the `data/` directory (e.g., `data/lebron-james.csv`).
- Custom CSS provides a modern, NBA-themed interface.

---

## 📐 Architecture

<img width="1131" alt="streamlit-architecture" src="https://github.com/user-attachments/assets/542f3bfc-b25b-4283-b67d-ef263e6564a4" />

---

## 🔗 Deployed App
[Streamlit App](https://nba-predictions-mia.streamlit.app/)
---

Feel free to contribute or suggest improvements!
