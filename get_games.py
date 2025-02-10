import functions as f

class getGames:
    def __init__(self, player_id, recent_seasons=1):
        self.player_id = player_id
        year_start, year_end = f.get_seasons_played(self.player_id)
        year_start = int(year_end) - recent_seasons
        self.player_all_games = f.get_player_stats_per_game(player_id, int(year_start), int(year_end))
        
    def enrich_stats(self):
        self.player_all_games['YEAR_DAY'] = self.player_all_games['GAME_DATE'].apply(f.date_to_year_day)
        self.player_all_games['WEEK_DAY'] = self.player_all_games['GAME_DATE'].apply(f.date_to_weekday)
        self.player_all_games = f.calculate_rest_days(self.player_all_games)

        f.get_team_id(self.player_all_games)
        f.get_opponent_id(self.player_all_games)
        f.format_season_id(self.player_all_games)
        f.encode_win(self.player_all_games)
        f.calculate_win_percentages(self.player_all_games)
        f.calculate_recent_win_percentage(self.player_all_games, 10)
        f.calculate_ppg_vs_opponent(self.player_all_games)
        self.player_all_games = self.player_all_games.drop(columns=["Player_ID", "WL", "FGA", "FG3A", "FTA", "REB", "Game_ID", "GAME_DATE", "MATCHUP", "VIDEO_AVAILABLE"])
        
        # Calcular la media de las columnas de los últimos 5 partidos anteriores para cada fila
        columns_to_average = ["MIN", "FGM", "FG_PCT", "FG3M", "FG3_PCT", "FTM", "FT_PCT", 
                            "W%_TOTAL", "W%_OPPONENT", "W%_RECENT", "PPG_VS_OPPONENT"]

        self.player_all_games = f.create_rolling_features(self.player_all_games, columns_to_average)
        
        f.export_csv(self.player_all_games, self.player_id)
        
        return self.player_all_games

        
if __name__ == "__main__":
    player_id = 2544
    player_all_games = getGames(player_id).enrich_stats()
    print(player_all_games)