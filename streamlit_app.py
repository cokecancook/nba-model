import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict_points
import get_games as gg

# ========== Page Config ========== #
st.set_page_config(page_title="NBA Points Predictor", layout="wide")

# ========== CSS ========== #
@st.cache_data
def load_css(file_name):
    with open(file_name) as f:
        return f.read()  # Return CSS content

# Load CSS once and apply
css_content = load_css("style.css")
st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

dataframes = []

@st.cache_data
def load_dataframes():
    player_ids = [2544, 1628369, 1629029, 201939, 203507]
    for player_id in player_ids:
        dataframes.append(gg.getGames(player_id).enrich_stats())
    
    return dataframes

# ========== Data ========== #
def load_data(player='lebron-james'):
    try:
        df = pd.read_csv(f"data/{player}.csv")      
        return df
    except FileNotFoundError:
        st.error("CSV file not found!")
        return pd.DataFrame()  # Return empty DataFrame as fallback

# Initialize session state for selected player
@st.cache_data
def initialize_selected_player(player='lebron-james'):
    st.session_state.current_df = load_data(player)
    st.session_state.selected_player = player
    st.session_state.player_name = 'LeBron'

# Function to update the DataFrame based on the selected player
def update_dataframe(player):
    st.session_state.current_df = load_data(player)
    st.session_state.selected_player = player

# ========== Pages ========== #
def introduction():
    initialize_selected_player()
    st.markdown("""
    <div class="hero">
        <img src="https://cdn.nba.com/logos/leagues/logo-nba.svg" alt="NBA Logo" title="NBA Logo">
        <h1 class="gradient-text">Performance Model</h1>
        <p class="subheading">Advanced analytics for player performance forecasting</p>
    </div>

    <div class="content-container">
        <div class="card">
            <p>Create player databases on demand including <span style="color: var(--secondary);"><strong>up-to-date statistics</strong></span> and explore the emerging patterns.</p>
            <div class="shine"></div>
        </div>
        <div class="card">
            <p>Predict player performance using <span style="color: var(--secondary);"><strong>machine learning</strong></span> models trained on historical NBA data.</p>
            <div class="shine"></div>
        </div>
    </div>

    <div class="image-container">
        <img src="https://a57.foxsports.com/statics.foxsports.com/www.foxsports.com/content/uploads/2024/06/1294/728/2024-06-14_2024-2025-NBA-Championship-Futures_16x9-2.jpg" alt="NBA Top Players" title="NBA Top Players">
    </div>
    """, unsafe_allow_html=True)

def eda():
    st.markdown("""
    <div class="hero">
        <img src="https://cdn.nba.com/logos/leagues/logo-nba.svg" alt="NBA Logo" title="NBA Logo">
        <h1 class="gradient-text">Data Analysis</h1>
        <p class="subheading">Select player to explore statistics, patterns, and trends.</p>
    </div>    
    """, unsafe_allow_html=True)
        
    col1, col2, col3 = st.columns([0.21, 7, 0.21])
    
    with col2:
        player1, player2, player3, player4, player5 = st.columns(5)
        if player1.button("Curry", use_container_width=True, key="curry"):
            update_dataframe('stephen-curry')
            st.session_state.player_name = 'Curry'
        if player2.button("Giannis", use_container_width=True, key="giannis"):
            update_dataframe('giannis-antetokounmpo')
            st.session_state.player_name = 'Giannis'
        if player3.button("Dončić", use_container_width=True):
            update_dataframe('luka-doncic')
            st.session_state.player_name = 'Dončić'
        if player4.button("Tatum", use_container_width=True, key="tatum"):
            update_dataframe('jayson-tatum')
            st.session_state.player_name = 'Tatum'
        if player5.button("LeBron", use_container_width=True, key="lebron"):
            update_dataframe('lebron-james')
            st.session_state.player_name = 'LeBron'

    if st.session_state.selected_player is not None:
        
        with st.container():
            col1b, col2b, col3b, col4b = st.columns([0.21, 3.5, 3.5, 0.21])  # Adjust middle column width as needed

            with col2b:  # Middle column controls the width
                st.subheader(f"{st.session_state.player_name}'s PPG (Last 100 Games)")
                last_100_values = st.session_state.current_df.tail(100)

                st.line_chart(
                    last_100_values["PTS"],  # Use the PTS column
                    use_container_width=True,
                    height=240
                )
            with col3b:  # Middle column controls the width
                st.subheader(f"Stats")
                st.dataframe(
                    st.session_state.current_df
                    .sort_index(ascending=False)
                    .style.background_gradient(cmap="Blues"),
                    use_container_width=True,  # Fits within the column width
                    height=240,
                )
    else:
        st.info("Select a player to explore statistics.")
    
    col1, col2, col3 = st.columns([0.21, 7, 0.21])

    with col2:
        st.button("Download latest data (for all 5 players)", on_click=load_dataframes, key="download", use_container_width=True)
        
def prediction():
    st.markdown("""
    <div class="hero">
        <img src="https://cdn.nba.com/logos/leagues/logo-nba.svg" alt="NBA Logo" title="NBA Logo">
        <h1 class="gradient-text">Points Prediction</h1>
        <p class="subheading">Predict future performance for upcomming game.</p>
    </div>    
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.21, 7, 0.21])

    with col2:
        player1, player2, player3, player4, player5 = st.columns(5)
        if player1.button("Curry", use_container_width=True, key="curry"):
            update_dataframe('stephen-curry')
            st.session_state.player_name = 'Curry'
        if player2.button("Giannis", use_container_width=True, key="giannis"):
            update_dataframe('giannis-antetokounmpo')
            st.session_state.player_name = 'Giannis'
        if player3.button("Dončić", use_container_width=True):
            update_dataframe('luka-doncic')
            st.session_state.player_name = 'Dončić'
        if player4.button("Tatum", use_container_width=True, key="tatum"):
            update_dataframe('jayson-tatum')
            st.session_state.player_name = 'Tatum'
        if player5.button("LeBron", use_container_width=True, key="lebron"):
            update_dataframe('lebron-james')
            st.session_state.player_name = 'LeBron'
    
    if st.session_state.selected_player is not None:
        with st.container():

            col1c, col2c, col3c, col4c = st.columns([0.21, 3.5, 3.5, 0.21])  # Adjust middle column width as needed

            with col2c:
                st.subheader(f"{st.session_state.player_name}'s PPG (Last 100 Games)")
                last_100_values = st.session_state.current_df.tail(100)

                st.line_chart(
                    last_100_values["PTS"],  # Use the PTS column
                    use_container_width=True,
                    height=240,
            
                )

            with col3c:
                # Inputs del usuario para los últimos 5 partidos
                st.subheader("Upcoming Game")

                model_files = {
                    "Curry": "stephen-curry",
                    "Giannis": "giannis-antetokounmpo",
                    "Dončić": "luka-doncic",
                    "Tatum": "jayson-tatum",
                    "LeBron": "lebron-james",
                }

                # Obtener el nombre de archivo correspondiente al jugador seleccionado
                player_key = model_files.get(st.session_state.player_name)

                # Botón para predecir
                if st.button("Predict Next Game Points", key="predict", use_container_width=True):
                    # Llamar al modelo correcto
                    predicted_pts = predict_points(player_key)

                    # Mostrar resultado
                    st.success(f"🎯 PPG Prediction: {predicted_pts}")
        
    
    

# ========== Main App ========== #
def main():
    page = st.sidebar.radio("Select a Page", ["🏀 Introduction", "🔍 EDA", "🔮 Prediction"], key="menu", label_visibility="hidden")
    
    st.sidebar.markdown("""
        <div class="sidebar-footer">
            <p class="sidebar-footer-subheading">Developed by:</p>
            <p><a href="https://github.com/Maurobalas" target="_blank">Mauro Balaguer</a></p>
            <p><a href="https://github.com/anpiboi" target="_blank">Andreu Picornell</a></p>
            <p><a href="https://github.com/cokecancook" target="_blank">Coke Stuyck</a></p>
        </div>
    """, unsafe_allow_html=True)

    if page == "🏀 Introduction":
        introduction()
    elif page == "🔍 EDA":
        eda()
    elif page == "🔮 Prediction":
        prediction()

if __name__ == "__main__":
    main()