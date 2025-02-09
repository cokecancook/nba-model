import pandas as pd
import numpy as np

import streamlit as st
import streamlit.components.v1 as components

# ========== Page Config ========== #
st.set_page_config(page_title="NBA Points Predictor", layout="wide")

# ========== CSS ========== #
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# ========== JS ========== #
def local_js(file_name):
    with open(file_name) as f:
        st.markdown(f'<script>{f.read()}</script>', unsafe_allow_html=True)

local_js("script.js")

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
    with open("introduction.html", "r", encoding="utf-8") as file:
        html_content = file.read()
        
    components.html(html_content)
    
def eda():
    with open("eda.html", "r", encoding="utf-8") as file:
        html_content = file.read()
        
    components.html(html_content)    
    
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
    with open("prediction.html", "r", encoding="utf-8") as file:
        html_content = file.read()
        
    components.html(html_content)    
    
    

# ========== Main App ========== #
def main():
    page = st.sidebar.radio("", ["🏀 Introduction", "🔍 EDA", "🔮 Prediction"])

    if page == "🏀 Introduction":
        introduction()
    elif page == "🔍 EDA":
        eda()
    elif page == "🔮 Prediction":
        prediction()
        
    with open("main.html", "r", encoding="utf-8") as file:
        html_content = file.read()
        
    components.html(html_content)    

if __name__ == "__main__":
    main()