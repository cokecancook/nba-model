import streamlit as st
import pandas as pd
import numpy as np

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


# ========== Data ========== #
def load_data(player='lebron-james'):
    try:
        df = pd.read_csv(f"data/{player}.csv")      
        return df
    except FileNotFoundError:
        st.error("CSV file not found!")
        return pd.DataFrame()  # Return empty DataFrame as fallback


# Initialize session state for selected player
if "selected_player" not in st.session_state:
    st.session_state.current_df = load_data("lebron-james")
    
# Function to update the DataFrame based on the selected player
def update_dataframe(player):
    st.session_state.current_df = load_data(player)


# ========== Pages ========== #
def introduction():
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
        if player1.button("Stephen Curry", use_container_width=True):
            update_dataframe('stephen-curry')
        if player2.button("Giannis Antetokounmpo", use_container_width=True):
            update_dataframe('giannis-antetokounmpo')
        if player3.button("Luka Dončić", use_container_width=True):
            update_dataframe('luka-dončić')
        if player4.button("Jayson Tatum", use_container_width=True):
            update_dataframe('jayson-tatum')
        if player5.button("LeBron James", use_container_width=True):
            update_dataframe('lebron-james')

    with st.container():
        st.subheader("Player Stats (Last 10 Games)")
        col1b, col2b, col3b = st.columns([0.21, 7, 0.21])  # Adjust middle column width as needed

        with col2b:  # Middle column controls the width
            st.dataframe(
                st.session_state.current_df.tail(10)
                .sort_index(ascending=False)
                .style.background_gradient(cmap="Blues"),
                use_container_width=True,  # Fits within the column width
            )
                
    with st.container():
        st.subheader(f"PPG Scored (Current Season)")
        col1, col2, col3 = st.columns([0.21, 7, 0.21])  # Adjust middle column width as needed

        with col2:  # Middle column controls the width
            last_100_values = st.session_state.current_df.tail(100)

            st.line_chart(
                last_100_values["PTS"],  # Use the PTS column
                use_container_width=True
            )
            # Optional: Add a title


def prediction():
    st.markdown("""
    <div class="hero">
        <img src="https://cdn.nba.com/logos/leagues/logo-nba.svg" alt="NBA Logo" title="NBA Logo">
        <h1 class="gradient-text">Points Prediction</h1>
        <p class="subheading">Predict future performance for upcomming game.</p>
    </div>    
    """, unsafe_allow_html=True)
    
    

# ========== Main App ========== #
def main():
    page = st.sidebar.radio("", ["🏀 Introduction", "🔍 EDA", "🔮 Prediction"])
    
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