import streamlit as st
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# ========== Page Config ========== #
st.set_page_config(page_title="NBA Points Predictor", layout="wide")

# ========== CSS ========== #
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

app = Flask(__name__)

@app.route('/_stcore/streamlit-components/update-player', methods=['POST'])
def update_player():
    data = request.get_json()
    player = data.get('player')
    if player:
        st.session_state.selected_player = player
        return jsonify(success=True)
    return jsonify(success=False), 400

# ========== Mock Data ========== #
@st.cache_data
def load_data(player='lebron-james'):
    try:
        df = pd.read_csv(f"data/{player}.csv")
        
        return df
    except FileNotFoundError:
        st.error("CSV file not found!")
        return pd.DataFrame()  # Return empty DataFrame as fallback

# Function to update the DataFrame based on the selected player
def update_df(player):
    if player == 'Stephen Curry':
        return load_data('stephen-curry')
    elif player == 'Giannis Antetokounmpo':
        return load_data('giannis-antetokounmpo')
    elif player == 'Luka Dončić':
        return load_data('luca-doncic')
    elif player == 'Jayson Tatum':
        return load_data('jayson-tatum')
    elif player == 'Lebron James':
        return load_data('lebron-james')
    else:
        return pd.DataFrame()  # Fallback


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

    st.markdown("""
    <div class="content-container-no-animation">
        <div class="player-selection">
            <button onclick="selectPlayer('Stephen Curry')">Stephen Curry</button>
            <button onclick="selectPlayer('Giannis Antetokounmpo')">Giannis Antetokounmpo</button>
            <button onclick="selectPlayer('Luka Dončić')">Luka Dončić</button>
            <button onclick="selectPlayer('Jayson Tatum')">Jayson Tatum</button>
            <button onclick="selectPlayer('Lebron James')">Lebron James</button>
        </div>    
    </div>
    """, unsafe_allow_html=True)
    
    # JavaScript to handle button clicks and update session state
    st.markdown("""
    <script>
    function selectPlayer(player) {
        fetch('/_stcore/streamlit-components/update-player', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ player: player })
        }).then(response => {
            if (response.ok) {
                window.location.reload();  // Reload the page to update the app
            }
        });
    }
    </script>
    """, unsafe_allow_html=True)
    
        # Initialize session state for the selected player
    # if 'selected_player' not in st.session_state:
    #     st.session_state.selected_player = 'Lebron James'  # Default player
    
    if 'selected_player' not in st.session_state:
        st.session_state.selected_player = None

    df = update_df(st.session_state.selected_player)
    
    # df = load_data()

    with st.container():
        st.subheader("Player Stats (Last 10 Games)")
        col1, col2, col3 = st.columns([0.21, 7, 0.21])  # Adjust middle column width as needed

        with col2:  # Middle column controls the width
            st.dataframe(
                df.tail(10)
                .sort_index(ascending=False)
                .style.background_gradient(cmap="Blues"),
                use_container_width=True,  # Fits within the column width
            )
                
    with st.container():
        st.write(f"Points Scored per Game in the current Season")
        col1, col2, col3 = st.columns([0.21, 7, 0.21])  # Adjust middle column width as needed

        with col2:  # Middle column controls the width
            last_100_values = df.tail(100)

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