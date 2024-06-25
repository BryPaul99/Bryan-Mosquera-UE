import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import main
import visuals
import seaborn as sns
import requests
import utils
import os
if __name__ == "__main__":
    driver = webdriver.Chrome()
    
# whoscored match centre url of the required match (Example)
url = "https://www.whoscored.com/Matches/1789428/Live/Europe-Champions-League-2023-2024-FC-Porto-Arsenal"
match_data = main.getMatchData(driver, url, close_window=True)

# Match dataframe containing info about the match
matches_df = main.createMatchesDF(match_data)

# Events dataframe      
events_df = main.createEventsDF(match_data)

# match Id
matchId = match_data['matchId']

# Information about respective teams as dictionary
home_data = matches_df['home'][matchId]
away_data = matches_df['away'][matchId]
events_df = main.addEpvToDataFrame(events_df)
events_df
events_df.to_csv('ARvsCh.csv')
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException

league_urls = main.getLeagueUrls()
league_urls

match_urls = main.getMatchUrls(comp_urls=league_urls, competition='Premier League', season='2023/2024')
match_urls[:38]
match_urls = main.getMatchUrls(comp_urls=league_urls, competition='Premier League', season='2023/2024')
match_urls[:38]
team_urls = main.getTeamUrls(team='Arsenal', match_urls=match_urls)
team_urls[:38]
team_urls
# eg. first 5 matches of Arsenal
matches_data = main.getMatchesData(match_urls=team_urls[:38])
matches_data
events_ls = [main.createEventsDF(match) for match in matches_data]
# Add EPV column
events_list = [main.addEpvToDataFrame(match) for match in events_ls]
events_dfs = pd.concat(events_list)
events_dfs.head()
events_dfs.to_csv('Premier2024.csv')
# Team data (from match data of single match)
team = 'Porto'
teamId = 297
opponent = 'Arsenal' 
venue = 'home'

# Get Player Names for home team
team_players_dict = {}
for player in matches_df[venue][match_data['matchId']].iloc[0][0]['players']:
    team_players_dict[player['playerId']] = player['name'] 
    
# Total Passes
passes_df = events_df.loc[events_df['type']=='Pass'].reset_index(drop=True)
passes_df = passes_df.loc[passes_df['outcomeType']=='Successful'].reset_index(drop=True)
passes_df = passes_df.loc[passes_df['teamId'] == teamId].reset_index(drop=True)
passes_df.head()
# Cut in 2
first_half_passes = passes_df.loc[passes_df['period']=='FirstHalf']
second_half_passes = passes_df.loc[passes_df['period']=='SecondHalf'].reset_index(drop=True)

# Cut in 4 (quarter = 25 mins)
first_quarter_passes = first_half_passes.loc[first_half_passes['minute'] <=40]
second_quarter_passes = first_half_passes.loc[first_half_passes['minute'] > 40].reset_index(drop=True)
third_quarter_passes = second_half_passes.loc[second_half_passes['minute'] <= 80]
fourth_quarter_passes = second_half_passes.loc[second_half_passes['minute'] > 80].reset_index(drop=True)

first_quarter_passes.head()
# Team data
team = 'Porto'
teamId = 297
opponent = 'Arsenal'
venue = 'home'

visuals.getTeamTotalPasses(events_df, teamId, team, opponent, pitch_color='#22312b')

# Team data
team = 'Porto'
teamId = 13
opponent = 'Arsenal'
venue = 'home'

# Create Pass Network     
# you can change marker_label to 'name' as well
fig,ax = plt.subplots(figsize=(16,11))
fig.set_facecolor("#000000")
visuals.createPassNetworks(match_data, events_df, matchId=match_data['matchId'], team='Arsenal', max_line_width=6, 
                           marker_size=1500, edgewidth=3, dh_arrow_width=25, marker_color="#000000",
                           marker_edge_color='w', shrink=25, ax=ax, kit_no_size=25)
# Team data
team = 'Porto'
teamId = 13
opponent = 'Arsenal'
venue = 'home'

# Create Progressive Pass Network
# you can change marker_label to 'name' as well
fig,ax = plt.subplots(figsize=(16,11))
fig.set_facecolor("#000000")
visuals.createAttPassNetworks(match_data, events_df, matchId=match_data['matchId'], team='Arsenal', max_line_width=6, 
                              marker_size=1300, edgewidth=3, dh_arrow_width=25, marker_color="#000000", 
                              marker_edge_color='w', shrink=24, ax=ax, kit_no_size=25)
# Team data
team = 'Arsenal'
teamId = 13
opponent = 'Liverpool'
venue = 'home'

fig,ax = plt.subplots(figsize=(16,11))
visuals.createShotmap(match_data, events_df, team='Arsenal', pitchcolor='black', shotcolor='white', 
                      goalcolor='red', titlecolor='white', legendcolor='white', marker_size=500, fig=fig, ax=ax)
# Team data
team = 'Arsenal'
teamId = 13
opponent = 'Liverpool'
venue = 'home'

# Choose your color palette from here: https://seaborn.pydata.org/tutorial/color_palettes.html
fig,ax = plt.subplots(figsize=(16,11))
fig.set_facecolor('#171717')
visuals.createPVFormationMap(match_data, events_df, team='Liverpool', color_palette=sns.color_palette("flare", as_cmap=True),
                             markerstyle='o', markersize=2000, markeredgewidth=2, labelsize=8, labelcolor='w', ax=ax)