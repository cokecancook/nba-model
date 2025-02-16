import pandas as pd

TEAMS_DF = pd.DataFrame({
    'id': [
        # Current Teams
        1610612737, 1610612738, 1610612739, 1610612740, 1610612741, 
        1610612742, 1610612743, 1610612744, 1610612745, 1610612746, 
        1610612747, 1610612748, 1610612749, 1610612750, 1610612751, 
        1610612752, 1610612753, 1610612754, 1610612755, 1610612756, 
        1610612757, 1610612758, 1610612759, 1610612760, 1610612761, 
        1610612762, 1610612763, 1610612764, 1610612765, 1610612766,
        # Historical Teams and Previous Iterations
        1610612767, 1610612768, 1610612769, 1610612770, 1610612771,
        1610612772, 1610612773, 1610612774, 1610612775, 1610612776,
        1610612777, 1610612778, 1610612779, 1610612780, 1610612781,
        1610612782, 1610612783, 1610612784, 1610612785, 1610612786,
        1610612787, 1610612788, 1610612789, 1610612790, 1610612791
    ],
    'abbreviation': [
        # Current Teams
        'ATL', 'BOS', 'CLE', 'NOP', 'CHI', 'DAL', 'DEN', 'GSW', 'HOU', 'LAC',
        'LAL', 'MIA', 'MIL', 'MIN', 'BKN', 'NYK', 'ORL', 'IND', 'PHI', 'PHX',
        'POR', 'SAC', 'SAS', 'OKC', 'TOR', 'UTA', 'MEM', 'WAS', 'DET', 'CHA',
        # Historical Teams and Previous Iterations
        'NJN', 'NOH', 'SEA', 'NOK', 'VAN', 'SDC', 'BUF', 'SDR', 'KC', 'CIN',
        'BAL', 'STL', 'SYR', 'WSB', 'AND', 'IND', 'DNN', 'PIT', 'ROC', 'CHP',
        'TRI', 'SHE', 'WAT', 'INO', 'PRO'
    ],
    'team_names': [
        # Current Teams
        'Atlanta Hawks', 'Boston Celtics', 'Cleveland Cavaliers', 'New Orleans Pelicans', 'Chicago Bulls',
        'Dallas Mavericks', 'Denver Nuggets', 'Golden State Warriors', 'Houston Rockets', 'Los Angeles Clippers',
        'Los Angeles Lakers', 'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves', 'Brooklyn Nets',
        'New York Knicks', 'Orlando Magic', 'Indiana Pacers', 'Philadelphia 76ers', 'Phoenix Suns',
        'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Oklahoma City Thunder', 'Toronto Raptors',
        'Utah Jazz', 'Memphis Grizzlies', 'Washington Wizards', 'Detroit Pistons', 'Charlotte Hornets',
        
        # Historical Teams and Previous Iterations
        'New Jersey Nets', 'New Orleans Hornets', 'Seattle SuperSonics', 'New Orleans/Oklahoma City Hornets', 'Vancouver Grizzlies',
        'San Diego Clippers', 'San Diego Kings', 'San Diego Mavericks', 'Kansas City Kings', 'Cincinnati Royals',
        'Baltimore Bullets', 'St. Louis Hawks', 'St. Louis Bulls', 'Washington Bullets', 'Washington Capitals',
        'Indiana Pacers', 'Denver Nuggets', 'Pittsburgh Penguins', 'Rochester Royals', 'Philadelphia Flyers',
        'Tri-Cities Blackhawks', 'Sheffield Steelers', 'Winnipeg Jets', 'Inonite Hawks', 'Providence Bruins'
    ]       
})

if __name__ == "__main__":
    print(TEAMS_DF)

