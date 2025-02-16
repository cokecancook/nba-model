import time
from datetime import datetime

import pandas as pd
from pandas import DataFrame
import re

from nba_api.stats.endpoints import playergamelog, teamgamelog, playercareerstats, commonplayerinfo, commonallplayers, commonteamroster, leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams

from teams import TEAMS_DF

def get_seasons_played(player_id):

    career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
    career_data = career_stats.get_data_frames()[0]
    player_stats = career_data[['SEASON_ID']]

    year_start = player_stats['SEASON_ID'].iloc[0][:4]
    year_end = player_stats['SEASON_ID'].iloc[len(player_stats) - 1][:4]
    year_start, year_end

    return year_start, year_end


def get_player_stats_per_game(player_id, year_start, year_end):
    
    player_game_stats = pd.DataFrame()
    print('Downloading... ', end='')
    
    for year in range(year_start, year_end + 1):
        game_log = playergamelog.PlayerGameLog(player_id=player_id, season=year)
        game_data = game_log.get_data_frames()[0]
        player_game_stats = pd.concat([player_game_stats, game_data], ignore_index=True)
        print(f'{year}', end=' ')
        time.sleep(0.5)

    return player_game_stats


def date_to_year_day(date_str):
    
    date_object = datetime.strptime(date_str, "%b %d, %Y")
    day_of_year = date_object.timetuple().tm_yday
    
    return day_of_year


def date_to_weekday(date_str):

    date_object = datetime.strptime(date_str, "%b %d, %Y")
    return date_object.isoweekday()


def calculate_rest_days(df, date_column='GAME_DATE'):
    
    data = df.copy()
    data[date_column] = pd.to_datetime(data[date_column], format="%b %d, %Y")
    data = data.sort_values(by=date_column).reset_index(drop=True)

    data['REST_DAYS'] = (data[date_column] - data[date_column].shift(1)).dt.total_seconds() / (24 * 60 * 60)

    data['REST_DAYS'] = data['REST_DAYS'].fillna(0).astype(int)
    
    return data

def get_home_game(df):
    
    df['HOME'] = df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
    
    return df

def get_team_id(df):
    
    df['TEAM_ID'] = df['MATCHUP'].apply(lambda x: x[:3])

    team_id_lookup = TEAMS_DF.set_index('abbreviation')['id'].to_dict()
    team_id_lookup = {key: str(value)[-2:] for key, value in team_id_lookup.items()}

    team_id_lookup = {key: value for key, value in team_id_lookup.items()}

    df['TEAM_ID'] = df['TEAM_ID'].map(team_id_lookup)

    return df


def get_opponent_id(df):

    df['OPPONENT_ID'] = df['MATCHUP'].apply(lambda x: x[-3:])

    team_id_lookup = TEAMS_DF.set_index('abbreviation')['id'].to_dict()
    team_id_lookup = {key: str(value)[-2:] for key, value in team_id_lookup.items()}

    team_id_lookup = {key: value for key, value in team_id_lookup.items()}

    df['OPPONENT_ID'] = df['OPPONENT_ID'].map(team_id_lookup)

    return df


def format_season_id(df):
    df['SEASON_ID'] = df['SEASON_ID'].astype(str).str[1:]
    return df


def encode_win(df):
    df['W'] = df['WL'].map({'W': 1, 'L': 0}).astype('int')
    return df


def calculate_win_percentages(df):
    df['W%_TOTAL'] = 0.0
    df['W%_OPPONENT'] = 0.0

    total_games = 0
    total_wins = 0
    opponent_stats = {}

    for index, row in df.iterrows():
        opponent_id = row['OPPONENT_ID']
        result = row['W'] 
        
        total_games += 1
        if result == 1:
            total_wins += 1
        
        if opponent_id not in opponent_stats:
            opponent_stats[opponent_id] = {'games': 0, 'wins': 0}
        opponent_stats[opponent_id]['games'] += 1
        if result == 1:
            opponent_stats[opponent_id]['wins'] += 1
        
        df.at[index, 'W%_TOTAL'] = round(total_wins / total_games, 2)
        df.at[index, 'W%_OPPONENT'] = round((
            opponent_stats[opponent_id]['wins'] / opponent_stats[opponent_id]['games']
        ), 2)

    return df


def calculate_recent_win_percentage(df, n=5):

    df['W'] = df['W'].astype(int)
    df['W%_RECENT'] = 0.0

    recent_results = []

    for index, row in df.iterrows():
        recent_results.append(row['W'])

        if len(recent_results) > n:
            recent_results.pop(0)

        recent_wins = sum(recent_results)
        recent_games = len(recent_results)

        df.at[index, 'W%_RECENT'] = round(recent_wins / recent_games, 2)

    return df


def calculate_ppg_vs_opponent(df):
    df['PPG_VS_OPPONENT'] = 0.0
    
    opponent_stats = {}

    for index, row in df.iterrows():
        opponent_team_id = row['OPPONENT_ID']
        points_scored = row['PTS']
        
        if opponent_team_id not in opponent_stats:
            opponent_stats[opponent_team_id] = {'games': 0, 'points': 0}

        opponent_stats[opponent_team_id]['games'] += 1
        opponent_stats[opponent_team_id]['points'] += points_scored
        
        avg_ppg_vs_opponent = opponent_stats[opponent_team_id]['points'] / opponent_stats[opponent_team_id]['games']
        
        df.at[index, 'PPG_VS_OPPONENT'] = round(avg_ppg_vs_opponent, 2)

    return df


def create_rolling_features(df, columns, window_sizes=[5, 10]):
    """
    Crea columnas de medias móviles para las columnas especificadas,
    eliminando las originales.

    Args:
        df (pd.DataFrame): DataFrame con los datos históricos
        columns (list): Lista de columnas a transformar
        window_sizes (list): Tamaños de ventana para las medias móviles

    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas
    """
    df = df.copy()

    for col in columns:
        for window in window_sizes:
            # Media móvil con shift para evitar data leakage
            new_col = f"{col}_LAST{window}"
            df[new_col] = df[col].rolling(window=window, min_periods=1).mean().shift(1)

        df.drop(columns=[col], inplace=True)

    return df


def get_player_name(id):
    
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=id)

    # Extract the player's name from the response
    player_name = player_info.get_normalized_dict()['CommonPlayerInfo'][0]['DISPLAY_FIRST_LAST']

    #Replace special characters with their ANSI equivalent
    player_name = re.sub(r'[čć]', 'c', player_name)
    player_name = re.sub(r'[đ]', 'd', player_name)
    player_name = re.sub(r'[š]', 's', player_name)
    player_name = re.sub(r'[ž]', 'z', player_name)

    #player_name = player_name.replace(" ", "-")
    player_name = player_name.replace(" ", "-").lower()
    
    return player_name


def export_csv(df, id):
    player_name = get_player_name(id)
    # date_formatted = datetime.now().strftime("%d%m%Y")
    
    #df.to_csv(f'{player_name}-{date_formatted}.csv', index=False)
    df.to_csv(f'data/{player_name}.csv', index=False)