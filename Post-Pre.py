import json
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import lxml
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from pandas import json_normalize
import matplotlib
import matplotlib.pyplot as plt
import mplsoccer
from mplsoccer import Pitch, VerticalPitch, FontManager
import seaborn as sns
import numpy as np
chrome_driver_path = r"C:\Users/bryanmosquera/Downloads/chromedriver-mac-arm64/chromedriver.exe"
h_team = 'FC Porto'
a_team = 'Estrela da Amadora'
h_id = 297
a_id = 28635
competition_date = 'Liga Portugal - Feb 17, 2024'
#scraping code inspired by Varun Vasudevan (@TheDevilsDNA)
driver = webdriver.Chrome()
df1 = pd.DataFrame()
links = ['https://es.whoscored.com/Matches/1748620/MatchReport/Portugal-Liga-Portugal-2023-2024-FC-Porto-Estrela-da-Amadora']

wait = WebDriverWait(driver, 20)  # Increase time if needed

for i in links:    
    driver.get(i)
    
    # Close ad or accept cookies (replace with the actual element)
    try:
        close_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="some-button-id"]')))
        close_button.click()
    except:
        pass
    
    # Wait for the overlay to disappear (if it does so automatically)
    wait.until(EC.invisibility_of_element((By.ID, 'qc-cmp2-ui')))
    
    match_centre = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="sub-navigation"]/ul/li[4]/a')))
    match_centre.click()
    
    element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="layout-wrapper"]/script[1]')))
    match_centre = driver.find_element("xpath",'//*[@id="sub-navigation"]/ul/li[4]/a')
    match_centre.click()
    element = driver.find_element("xpath",'//*[@id="layout-wrapper"]/script[1]')
    script_content = element.get_attribute('innerHTML')
    script_ls = script_content.split(sep="  ")
    script_ls = list(filter(None, script_ls))
    script_ls = [name for name in script_ls if name.strip()]
    dictstring = script_ls[2][17:-2]
    content = json.loads(dictstring)
    match = json_normalize(content['events'],sep='_')
    hometeam = content['home']['name']
    awayteam = content['away']['name']
    homeid = content['home']['teamId']
    awayid = content['away']['teamId']
    players = pd.DataFrame()
    homepl = json_normalize(content['home']['players'],sep='_')[['name', 'position', 'shirtNo', 'playerId']]
    awaypl = json_normalize(content['away']['players'],sep='_')[['name', 'position', 'shirtNo', 'playerId']]
    players = pd.concat([homepl,awaypl])
    match = match.merge(players, how='left')
    df1 = pd.concat([df1, match])
    #match_id += 1
    driver.close()
    def extract_json_from_html(html_path, save_output=False):
    html_file = open(html_path, 'r',encoding="utf8")
    html = html_file.read()
    html_file.close()
    regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
    data_txt = re.findall(regex_pattern, html)[0]

    # add quotations for json parser
    data_txt = data_txt.replace('matchId', '"matchId"')
    data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
    data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
    data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
    data_txt = data_txt.replace('};', '}')

    if save_output:
        # save json data to txt
        output_file = open(f"{html_path}.txt", "wt")
        n = output_file.write(data_txt)
        output_file.close()

    return data_txt
def extract_data_from_dict(data):
    # load data from json
    event_types_json = data["matchCentreEventTypeJson"]
    formation_mappings = data["formationIdNameMappings"]
    events_dict = data["matchCentreData"]["events"]
    teams_dict = {data["matchCentreData"]['home']['teamId']: data["matchCentreData"]['home']['name'],
                  data["matchCentreData"]['away']['teamId']: data["matchCentreData"]['away']['name']}
    players_dict = data["matchCentreData"]["playerIdNameDictionary"]
    # create players dataframe
    players_home_df = pd.DataFrame(data["matchCentreData"]['home']['players'])
    players_home_df["teamId"] = data["matchCentreData"]['home']['teamId']
    players_away_df = pd.DataFrame(data["matchCentreData"]['away']['players'])
    players_away_df["teamId"] = data["matchCentreData"]['away']['teamId']
    players_df = pd.concat([players_home_df, players_away_df])
    players_ids = data["matchCentreData"]["playerIdNameDictionary"]
    return events_dict, players_df, teams_dict, players_ids
match_html_path = "/Users/bryanmosquera/Desktop/Porto Pre Partido vs Estelar /FC Porto 2-0 Estrela da Amadora - Liga Portugal 2023:2024 Live.html"
json_data_txt = extract_json_from_html(match_html_path)
data = json.loads(json_data_txt)
events_dict, players_df, teams_dict, players_ids = extract_data_from_dict(data)
#passes
df_passes = df1.loc[df1['type_displayName']=='Pass'].copy()
h_passes = df_passes[df_passes['teamId']==h_id]
a_passes = df_passes[df_passes['teamId']==a_id]
#passes IN opponent's half
h_opphalf_passes = h_passes[h_passes['x']>=50]
a_opphalf_passes = a_passes[a_passes['x']>=50]
#completed passes in opponent's half 
h_comp_opphalf_passes = h_opphalf_passes[h_opphalf_passes['outcomeType_displayName']=='Successful']
a_comp_opphalf_passes = a_opphalf_passes[a_opphalf_passes['outcomeType_displayName']=='Successful']
#passes in their own half
h_ownhalf_passes = h_passes[h_passes['x']<50]
a_ownhalf_passes = a_passes[a_passes['x']<50]
#completed passes in own half 
h_comp_ownhalf_passes = h_ownhalf_passes[h_ownhalf_passes['outcomeType_displayName']=='Successful']
a_comp_ownhalf_passes = a_ownhalf_passes[a_ownhalf_passes['outcomeType_displayName']=='Successful']
#passes INTO the final third
h_final_3rd_passes = h_passes[h_passes['endX']>=66]
h_final_3rd_passes = h_final_3rd_passes[h_final_3rd_passes['x']<=66]
a_final_3rd_passes = a_passes[a_passes['endX']>=66]
a_final_3rd_passes = a_final_3rd_passes[a_final_3rd_passes['x']<=66]
#completed passes into the final third
h_comp_final3rd_passes = h_final_3rd_passes[h_final_3rd_passes['outcomeType_displayName']=='Successful']
a_comp_final3rd_passes = a_final_3rd_passes[a_final_3rd_passes['outcomeType_displayName']=='Successful']
#passes into the opponent's box
h_passes_within_box = (
    (h_passes['endX'] >= 85) & 
    (h_passes['endY'] > 20) & 
    (h_passes['endY'] < 80)
)

h_passes_start_inside_box = (
    (h_passes['x'] > 85) & 
    (h_passes['y'] > 20) & 
    (h_passes['y'] < 80)
)

h_passes_intobox = h_passes[h_passes_within_box & ~h_passes_start_inside_box]

a_passes_within_box = (
    (a_passes['endX'] >= 85) & 
    (a_passes['endY'] > 20) & 
    (a_passes['endY'] < 80)
)

a_passes_start_inside_box = (
    (a_passes['x'] > 85) & 
    (a_passes['y'] > 20) & 
    (a_passes['y'] < 80)
)

a_passes_intobox = a_passes[a_passes_within_box & ~a_passes_start_inside_box]

#completed/successful passes into the opponent's box
h_comp_passes_intobox = h_passes_intobox[h_passes_intobox['outcomeType_displayName']=='Successful']
a_comp_passes_intobox = a_passes_intobox[a_passes_intobox['outcomeType_displayName']=='Successful']

#take-ons/dribbles
df_dribbles = df1.loc[df1['type_displayName']=='TakeOn'].copy()
h_dribbles = df_dribbles[df_dribbles['teamId']==h_id]
a_dribbles = df_dribbles[df_dribbles['teamId']==a_id]
#touches
df_touches = df1.loc[df1['isTouch']==True].copy()
h_touches = df_touches[df_touches['teamId']==h_id]
a_touches = df_touches[df_touches['teamId']==a_id]
df_touches2 = df1.loc[df1['type_displayName']=='BallTouch'].copy()
h_touches2 = df_touches2[df_touches2['teamId']==h_id]
a_touches2 = df_touches2[df_touches2['teamId']==a_id]
#dispossessions
df_dispossessions = df1.loc[df1['type_displayName']=='Dispossessed'].copy()
h_dispossessions = df_dispossessions[df_dispossessions['teamId']==h_id]
a_dispossessions = df_dispossessions[df_dispossessions['teamId']==a_id]
#shots
df_shots = df1.loc[df1['isShot']==True].copy()
h_shots = df_shots[df_shots['teamId']==h_id]
a_shots = df_shots[df_shots['teamId']==a_id]
#goals
df_goals = df1.loc[df1['type_displayName']=='Goal'].copy()
h_goals = df_goals[df_goals['teamId']==h_id]
a_goals = df_goals[df_goals['teamId']==a_id]
#tackles
df_tackles = df1.loc[df1.type_displayName=='Tackle'].copy()
h_tackles = df_tackles[df_tackles['teamId']==h_id]
a_tackles = df_tackles[df_tackles['teamId']==a_id]
#challenges
df_challenges = df1.loc[df1.type_displayName=='Challenge'].copy()
h_challenges = df_challenges[df_challenges['teamId']==h_id]
a_challenges = df_challenges[df_challenges['teamId']==a_id]
#clearances
df_clearances = df1.loc[df1.type_displayName=='Clearance'].copy()
h_clearances = df_clearances[df_clearances['teamId']==h_id]
a_clearances = df_clearances[df_clearances['teamId']==a_id]
#recoveries
df_recoveries = df1.loc[df1.type_displayName=='BallRecovery'].copy()
h_recoveries = df_recoveries[df_recoveries['teamId']==h_id]
a_recoveries = df_recoveries[df_recoveries['teamId']==a_id]
#blocked passes
df_blocked_passes = df1.loc[df1.type_displayName=='BlockedPass'].copy()
h_blocked_passes = df_blocked_passes[df_blocked_passes['teamId']==h_id]
a_blocked_passes = df_blocked_passes[df_blocked_passes['teamId']==a_id]
#blocked shots
df_blocked_shots = df_shots.loc[df_shots['blockedX'].notna()].copy()
h_blocked_shots = df_blocked_shots[df_blocked_shots['teamId']==a_id]
a_blocked_shots = df_blocked_shots[df_blocked_shots['teamId']==h_id]
#interceptions
df_interceptions = df1.loc[df1.type_displayName=='Interception'].copy()
h_interceptions = df_interceptions[df_interceptions['teamId']==h_id]
a_interceptions = df_interceptions[df_interceptions['teamId']==a_id]
#aerial duels
df_aerials = df1.loc[df1['type_displayName']=='Aerial'].copy()
h_aerials = df_aerials[df_aerials['teamId']==h_id]
a_aerials = df_aerials[df_aerials['teamId']==a_id]
#all defensive actions: 
df_defensive_actions = pd.concat([df1.loc[(df1.type_displayName=='Challenge') |  
        (df1.type_displayName=='BlockedPass')|
        (df1.type_displayName=='Tackle')|
        (df1.type_displayName=='Interception')|
        (df1.type_displayName=='Aerial')|
        (df1.type_displayName=='Foul')].copy(),df_blocked_shots])
h_defensive_actions = df_defensive_actions[df_defensive_actions['teamId']==h_id]
a_defensive_actions = df_defensive_actions[df_defensive_actions['teamId']==a_id]



#gk saves
df_saves = df1.loc[df1['type_displayName']=='SavedShot'].copy()
#gk pickups
df_pickups = df1.loc[df1['type_displayName']=='KeeperPickup'].copy()
#gk sweeps
df_sweeps = df1.loc[df1['type_displayName']=='KeeperSweeper']
#gk claims
df_claims = df1.loc[df1['type_displayName']=='Claim'].copy()
#gk punches
df_punches = df1.loc[df1['type_displayName']=='Punch'].copy()
#all gk actions
gk_actions = pd.concat([df_saves, df_pickups, df_sweeps, df_claims, df_punches])
df_passes['beginning'] = np.sqrt(np.square(100 - df_passes['x']) + np.square(50 - df_passes['y']))
df_passes['end'] = np.sqrt(np.square(100 - df_passes['endX']) + np.square(50 - df_passes['endY']))
df_passes.loc[:, 'progressive'] = df_passes['end'] / df_passes['beginning'] < 0.75
df_progressive_passes = df_passes[df_passes['progressive']==True]
a_comp_passes_intobox.columns
a_comp_passes_intobox
def get_events_df(events_dict):
    df = pd.DataFrame(events_dict)
    df['eventType'] = df.apply(lambda row: row['type']['displayName'], axis=1)
    df['outcomeType'] = df.apply(lambda row: row['outcomeType']['displayName'], axis=1)

    # create receiver column based on the next event
    # this will be correct only for successfull passes
    df["receiver"] = df["playerId"].shift(-1)

    return df
df = get_events_df(events_dict)
#For now I can only create carries and progressive carries dataframmes for one player based on his WhoScored playerId
#Put the player's id here, you can find the list of ids in the players_df below
player_id = 347204
players_df
carries = []

for index, event in df.iterrows():
    if event['eventType'] == 'Pass' and event['receiver'] == player_id:
        # The player received the ball
        start_event = event
        carry_ongoing = True
        for subsequent_index in range(index + 1, len(df)):
            subsequent_event = df.iloc[subsequent_index]
            if subsequent_event['playerId'] != player_id:
                continue  # Skip events not involving this player

            # Check if the event ends the carry
            if subsequent_event['playerId'] == player_id:
                # Consider shots and touches as potential carry-ending events
                if subsequent_event['isShot'] or (subsequent_event['eventType'] in ['Pass', 'Goal'] or subsequent_event['Dispossessed']):
                    end_event = subsequent_event
                    carry_ongoing = False
                    break

        if not carry_ongoing:
            # Extract carry data
            carry_data = {
                'start_x': start_event['endX'],  # End position of the pass
                'start_y': start_event['endY'],
                'end_x': end_event['x'],        # Position of the ending event
                'end_y': end_event['y'],
                # Add other details as needed
            }
            carries.append(carry_data)

# Create the carries DataFrame
player_carries = pd.DataFrame(carries)

# Assuming the opponent's goal is at x = 100
goal_x = 100

for index, row in player_carries.iterrows():
    start_distance = np.sqrt(np.square(goal_x - row['start_x']) + np.square(50 - row['start_y']))
    end_distance = np.sqrt(np.square(goal_x - row['end_x']) + np.square(50 - row['end_y']))

    # Check if the reduction in distance is at least 25%
    player_carries.at[index, 'progressive'] = (end_distance / start_distance) < 0.75
    #passes dataframe
def get_passes_df(events_dict):
    df = pd.DataFrame(events_dict)
    df['eventType'] = df.apply(lambda row: row['type']['displayName'], axis=1)
    df['outcomeType'] = df.apply(lambda row: row['outcomeType']['displayName'], axis=1)

    # create receiver column based on the next event
    # this will be correct only for successfull passes
    df["receiver"] = df["playerId"].shift(-1)

    # filter only passes
    passes_ids = df.index[df['eventType'] == 'Pass']
    df_Passes = df.loc[
        passes_ids, ["id", "x", "y", "endX", "endY", "teamId", "playerId", "receiver", "eventType", "outcomeType"]]

    return df_Passes
passes_df = get_passes_df(events_dict)
#this is for the passing network -> average pass locations and counts of passes between players, etc.
def get_passes_between_df(team_id, passes_df, players_df):
    # filter for only team
    passes_df = passes_df[passes_df["teamId"] == team_id]

    # add column with first eleven players only
    passes_df = passes_df.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')
    # filter on first eleven column
    passes_df = passes_df[passes_df['isFirstEleven'] == True]

    # calculate mean positions for players
    average_locs_and_count_df = (passes_df.groupby('playerId')
                                 .agg({'x': ['mean'], 'y': ['mean', 'count']}))
    average_locs_and_count_df.columns = ['x', 'y', 'count']
    average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position']],
                                                                on='playerId', how='left')
    average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')

    # calculate the number of passes between each position (using min/ max so we get passes both ways)
    passes_player_ids_df = passes_df.loc[:, ['id', 'playerId', 'receiver', 'teamId']]
    passes_player_ids_df['pos_max'] = (passes_player_ids_df[['playerId', 'receiver']].max(axis='columns'))
    passes_player_ids_df['pos_min'] = (passes_player_ids_df[['playerId', 'receiver']].min(axis='columns'))

    # get passes between each player
    passes_between_df = passes_player_ids_df.groupby(['pos_min', 'pos_max']).id.count().reset_index()
    passes_between_df.rename({'id': 'pass_count'}, axis='columns', inplace=True)

    # add on the location of each player so we have the start and end positions of the lines
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_min', right_index=True)
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_max', right_index=True,
                                                suffixes=['', '_end'])
    return passes_between_df, average_locs_and_count_df
home_team_id = list(teams_dict.keys())[0]  # selected home team
home_passes_between_df, home_average_locs_and_count_df = get_passes_between_df(home_team_id, passes_df, players_df)
away_team_id = list(teams_dict.keys())[1]  # selected home team
away_passes_between_df, away_average_locs_and_count_df = get_passes_between_df(away_team_id, passes_df, players_df)
def pass_network_visualization(ax, passes_between_df, average_locs_and_count_df, flipped=False):
    MAX_LINE_WIDTH = 20
    MAX_MARKER_SIZE = 1000
    passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max() *
                                  MAX_LINE_WIDTH)
    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']
                                                / average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)

    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('lime'))
    color = np.tile(color, (len(passes_between_df), 1))
    c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency

    pitch = Pitch(pitch_type='opta', pitch_color='#260000')
    pitch.draw(ax=ax)

    if flipped:
        passes_between_df['x'] = pitch.dim.right - passes_between_df['x']
        passes_between_df['y'] = pitch.dim.right - passes_between_df['y']
        passes_between_df['x_end'] = pitch.dim.right - passes_between_df['x_end']
        passes_between_df['y_end'] = pitch.dim.right - passes_between_df['y_end']
        average_locs_and_count_df['x'] = pitch.dim.right - average_locs_and_count_df['x']
        average_locs_and_count_df['y'] = pitch.dim.right - average_locs_and_count_df['y']

    pass_lines = pitch.lines(passes_between_df.x, passes_between_df.y,
                             passes_between_df.x_end, passes_between_df.y_end, lw=passes_between_df.width,
                             color=color, zorder=1, ax=ax)
    pass_nodes = pitch.scatter(average_locs_and_count_df.x, average_locs_and_count_df.y,
                               s=average_locs_and_count_df.marker_size,
                               color='black', edgecolors='lavender', linewidth=1, alpha=1, ax=ax)
    for index, row in average_locs_and_count_df.iterrows():
        player_name = row["name"].split()
        player_initials = "".join(word[0] for word in player_name).upper()
        pitch.annotate(player_initials, xy=(row.x, row.y), c='lavender', va='center',
                       ha='center', size=10, ax=ax)

    return pitch
#Events Map
#Pass into the box 
pitch = VerticalPitch(positional=True, pitch_type='opta', half=True, 
                      pitch_color='#000000', positional_linestyle='--',pad_top=10)
fig, ax = pitch.draw(figsize=(16, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
# Plot the completed passes
pitch.scatter(h_comp_passes_intobox.endX, h_comp_passes_intobox.endY,color='blue',zorder=3,marker='D',ax=ax)
pitch.lines(h_comp_passes_intobox.x, h_comp_passes_intobox.y,
             h_comp_passes_intobox.endX, h_comp_passes_intobox.endY, 
            comet=True, transparent=True, color='blue', label='completed passes',zorder=3, ax=ax)
pitch.scatter(h_passes_intobox.endX, h_passes_intobox.endY,color='white',marker='D',ax=ax)
pitch.lines(h_passes_intobox.x, h_passes_intobox.y,
             h_passes_intobox.endX, h_passes_intobox.endY, 
            comet=True, transparent=True, color='white', label='other passes',ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=10, loc='upper left', handlelength=4)
ax_title = ax.set_title(f'{h_team} passes into the box vs {a_team}', fontsize=22,color='white')

fig.text(0.72,0.83,f"{competition_date}",ha='right',color='aquamarine',size=12)
fig.text(0.64,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.3,0.1,'Data: Opta',color='white',size=11)
pitch = VerticalPitch(positional=True, pitch_type='opta', half=True, 
                      pitch_color='#000000', positional_linestyle='--',pad_top=10)
fig, ax = pitch.draw(figsize=(16, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
# Plot the completed passes
pitch.scatter(a_comp_passes_intobox.endX, a_comp_passes_intobox.endY,color='red',zorder=3,marker='D',ax=ax)
pitch.lines(a_comp_passes_intobox.x, a_comp_passes_intobox.y,
             a_comp_passes_intobox.endX, a_comp_passes_intobox.endY, 
            comet=True, transparent=True, color='red', label='completed passes',zorder=3, ax=ax)
pitch.scatter(a_passes_intobox.endX, a_passes_intobox.endY,color='tan',marker='D',ax=ax)
pitch.lines(a_passes_intobox.x, a_passes_intobox.y,
             a_passes_intobox.endX, a_passes_intobox.endY, 
            comet=True, transparent=True, color='tan', label='other passes',ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=10, loc='upper left', handlelength=4)
ax_title = ax.set_title(f'{a_team} Passes into the Box vs {h_team}', fontsize=22,color='white')

fig.text(0.72,0.82,f"{competition_date}",ha='right',color='aquamarine',size=12)
fig.text(0.64,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.3,0.1,'Data: Opta',color='white',size=11)
#In own half
pitch = Pitch(positional=True, pitch_type='opta', 
                      pitch_color='#000000', positional_linestyle='--',pad_top=10)
fig, ax = pitch.draw(figsize=(16, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
# Plot the completed passes
pitch.scatter(h_ownhalf_passes.endX, h_ownhalf_passes.endY,color='tan',marker='D',ax=ax)
pitch.lines(h_ownhalf_passes.x, h_ownhalf_passes.y,
             h_ownhalf_passes.endX, h_ownhalf_passes.endY, 
            comet=True, transparent=True, color='tan', label='failed passes',ax=ax)
pitch.scatter(h_comp_ownhalf_passes.endX, h_comp_ownhalf_passes.endY,color='darkorange',marker='D',ax=ax)
pitch.lines(h_comp_ownhalf_passes.x, h_comp_ownhalf_passes.y,
             h_comp_ownhalf_passes.endX, h_comp_ownhalf_passes.endY, 
            comet=True, transparent=True, color='darkorange', label='completed passes', ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=9, loc='upper left', handlelength=4)
ax_title = ax.set_title(f'{h_team} Passes in Own Half vs {a_team}', fontsize=25,color='white')

fig.text(0.77,0.83,f"{competition_date}",ha='right',color='aquamarine',size=11)
fig.text(0.69,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.25,0.1,'Data: Opta',color='white',size=11)
pitch = Pitch(positional=True, pitch_type='opta', 
                      pitch_color='#000000', positional_linestyle='--',pad_top=10)
fig, ax = pitch.draw(figsize=(16, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
# Plot the completed passes
pitch.scatter(100-a_ownhalf_passes.endX, 100-a_ownhalf_passes.endY,color='tan',marker='D',ax=ax)
pitch.lines(100-a_ownhalf_passes.x, 100-a_ownhalf_passes.y,
             100-a_ownhalf_passes.endX, 100-a_ownhalf_passes.endY, 
            comet=True, transparent=True, color='tan', label='failed passes',ax=ax)
pitch.scatter(100-a_comp_ownhalf_passes.endX, 100-a_comp_ownhalf_passes.endY,color='darkorange',marker='D',ax=ax)
pitch.lines(100-a_comp_ownhalf_passes.x, 100-a_comp_ownhalf_passes.y,
             100-a_comp_ownhalf_passes.endX, 100-a_comp_ownhalf_passes.endY, 
            comet=True, transparent=True, color='darkorange', label='completed passes', ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=10, loc='upper left', handlelength=4)
ax_title = ax.set_title(f'{a_team} Passes in Own Half vs {h_team}', fontsize=22,color='white')

fig.text(0.76,0.83,f"{competition_date}",ha='right',color='aquamarine',size=11)
fig.text(0.70,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.25,0.1,'Data: Opta',color='white',size=11)
#In Opponents's half
pitch = Pitch(positional=True, pitch_type='opta', 
                      pitch_color='#000000', positional_linestyle='--',pad_top=10)
fig, ax = pitch.draw(figsize=(16, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
# Plot the completed passes
pitch.scatter(h_opphalf_passes.endX, h_opphalf_passes.endY,color='tan',marker='D',ax=ax)
pitch.lines(h_opphalf_passes.x, h_opphalf_passes.y,
             h_opphalf_passes.endX, h_opphalf_passes.endY, 
            comet=True, transparent=True, color='tan', label='failed passes',ax=ax)
pitch.scatter(h_comp_opphalf_passes.endX, h_comp_opphalf_passes.endY,color='darkorange',marker='D',ax=ax)
pitch.lines(h_comp_opphalf_passes.x, h_comp_opphalf_passes.y,
             h_comp_opphalf_passes.endX, h_comp_opphalf_passes.endY, 
            comet=True, transparent=True, color='darkorange', label='completed passes', ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=9, loc='upper left', handlelength=4)
ax_title = ax.set_title(f"{h_team} Passes in {a_team} Half", fontsize=23,color='white')

fig.text(0.77,0.83,f"{competition_date}",ha='right',color='aquamarine',size=11)
fig.text(0.69,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.25,0.1,'Data: Opta',color='white',size=11)
pitch = Pitch(positional=True, pitch_type='opta', 
                      pitch_color='#000000', positional_linestyle='--',pad_top=10)
fig, ax = pitch.draw(figsize=(16, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
# Plot the completed passes
pitch.scatter(100-a_opphalf_passes.endX, 100-a_opphalf_passes.endY,color='tan',marker='D',ax=ax)
pitch.lines(100-a_opphalf_passes.x, 100-a_opphalf_passes.y,
             100-a_opphalf_passes.endX, 100-a_opphalf_passes.endY, 
            comet=True, transparent=True, color='tan', label='failed passes',ax=ax)
pitch.scatter(100-a_comp_opphalf_passes.endX, 100-a_comp_opphalf_passes.endY,color='darkorange',marker='D',ax=ax)
pitch.lines(100-a_comp_opphalf_passes.x, 100-a_comp_opphalf_passes.y,
             100-a_comp_opphalf_passes.endX, 100-a_comp_opphalf_passes.endY, 
            comet=True, transparent=True, color='darkorange', label='completed passes', ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=9, loc='upper left', handlelength=4)
ax_title = ax.set_title(f"{a_team} Passes in {h_team} Half", fontsize=22,color='white')

fig.text(0.77,0.83,f"{competition_date}",ha='right',color='aquamarine',size=11)
fig.text(0.70,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.25,0.1,'Data: Opta',color='white',size=11)
##Passes into the Final Third
pitch = Pitch(positional=True, pitch_type='opta', 
                      pitch_color='#000000', positional_linestyle='--',pad_top=10)
fig, ax = pitch.draw(figsize=(16, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
# Plot the completed passes
pitch.scatter(h_comp_final3rd_passes.endX, h_comp_final3rd_passes.endY,color='darkorange',marker='D',ax=ax)
pitch.lines(h_comp_final3rd_passes.x, h_comp_final3rd_passes.y,
             h_comp_final3rd_passes.endX, h_comp_final3rd_passes.endY, 
            comet=True, transparent=True, color='darkorange', label='completed passes',zorder=3, ax=ax)
pitch.scatter(h_final_3rd_passes.endX, h_final_3rd_passes.endY,color='tan',marker='D',ax=ax)
pitch.lines(h_final_3rd_passes.x, h_final_3rd_passes.y,
             h_final_3rd_passes.endX, h_final_3rd_passes.endY, 
            comet=True, transparent=True, color='tan', label='other passes',ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=9, loc='upper left', handlelength=4)
ax_title = ax.set_title(f'{h_team} Passes into the Final Third vs {a_team}', fontsize=22,color='white')

fig.text(0.77,0.83,f"{competition_date}",ha='right',color='aquamarine',size=11)
fig.text(0.70,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.25,0.1,'Data: Opta',color='white',size=11)
pitch = Pitch(positional=True, pitch_type='opta', 
                      pitch_color='#000000', positional_linestyle='--',pad_top=10)
fig, ax = pitch.draw(figsize=(16, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
# Plot the completed passes
pitch.scatter(100-a_comp_final3rd_passes.endX, 100-a_comp_final3rd_passes.endY,color='darkorange',marker='D',ax=ax)
pitch.lines(100-a_comp_final3rd_passes.x, 100-a_comp_final3rd_passes.y,
             100-a_comp_final3rd_passes.endX, 100-a_comp_final3rd_passes.endY, 
            comet=True, transparent=True, color='darkorange', label='completed passes',zorder=3, ax=ax)
pitch.scatter(100-a_final_3rd_passes.endX, 100-a_final_3rd_passes.endY,color='tan',marker='D',ax=ax)
pitch.lines(100-a_final_3rd_passes.x, 100-a_final_3rd_passes.y,
             100-a_final_3rd_passes.endX, 100-a_final_3rd_passes.endY, 
            comet=True, transparent=True, color='tan', label='other passes',ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=9, loc='upper left', handlelength=4)
ax_title = ax.set_title(f'{a_team} Passes into the Final Third vs {h_team}', fontsize=22,color='white')

fig.text(0.77,0.83,f"{competition_date}",ha='right',color='aquamarine',size=11)
fig.text(0.70,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.25,0.1,'Data: Opta',color='white',size=11)
##Shotmap
pitch = VerticalPitch(pitch_type='opta', pitch_color='#ead8cd', half=True)
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)

df_missed_shots = df1.loc[df1['type_displayName']=='MissedShots']
h_missed = df_missed_shots.loc[df_missed_shots['teamId']==h_id]
# Update 'endX' column for unblocked shots
h_shots.loc[h_shots['blockedX'].isna(), 'endX'] = 100
h_goals.loc[h_goals['blockedX'].isna(), 'endX'] = 100
h_missed.loc[h_missed['blockedX'].isna(), 'endX'] = 100

# Differentiate between blocked shots, goals, on-target shots, and missed shots
h_shots['on_target'] = np.where((h_shots['endX'] >= 100) & (h_shots['goalMouthY'] >= 45) & (h_shots['goalMouthY'] <= 55), True, False)
h_shots['missed'] = ~h_shots['on_target']

# Plot goals as red arrows
pitch.scatter(h_goals['x'], h_goals['y'],color='red',ax=ax)
pitch.arrows(h_goals['x'], h_goals['y'], h_goals['endX'], h_goals['goalMouthY'],
             width=2, headwidth=10, headlength=10, color='red', label='Goals',ax=ax)

# Plot missed shots (off target) as blue arrows
#missed_shots = h_shots[(h_shots['missed']) & (h_shots['isGoal'] != 1)]
pitch.scatter(h_missed.x,h_missed.y,color='blue',ax=ax)
#pitch.arrows(missed_shots['x'], missed_shots['y'], missed_shots['endX'], missed_shots['goalMouthY'],
#             width=2, headwidth=10, headlength=10, color='blue', label='Missed Shots', ax=ax)
pitch.arrows(h_missed.x,h_missed.y,h_missed.endX,h_missed.goalMouthY,
             width=2, headwidth=10, headlength=10, color='blue', label='Missed Shots', ax=ax)

# Plot on-target shots as green arrows (excluding goals)
on_target_shots = h_shots[(h_shots['on_target']) & (h_shots['isGoal'] != 1)]
#on_target_shots = h_shots(h_shots['on_target']==True).copy()
pitch.scatter(on_target_shots['x'], on_target_shots['y'],color='green',ax=ax)
pitch.arrows(on_target_shots['x'], on_target_shots['y'], on_target_shots['endX'], on_target_shots['goalMouthY'],
             width=2, headwidth=10, headlength=10, color='green', label='Shots On Target', ax=ax)

# Plot blocked shots as black arrows
blocked_shots = h_shots[h_shots['blockedX'].notna()]
pitch.scatter(blocked_shots['x'], blocked_shots['y'],color='black',ax=ax)
pitch.arrows(blocked_shots['x'], blocked_shots['y'], blocked_shots['blockedX'], blocked_shots['blockedY'],
             width=2, headwidth=10, headlength=10, color='black', label='Blocked Shots',ax=ax)
#title
fig.set_facecolor('#ead8cd')
ax_title = ax.set_title(f'{h_team} Shots vs {a_team}', fontsize=20,color='black')

# Add a legend to the plot
ax.legend(facecolor='#ead8cd', edgecolor='black',labelcolor='black',fontsize=15,loc='upper left', handlelength=5)
fig.text(0.82,0.85,f"{competition_date}",ha='right',color='black',size=11)
fig.text(0.8,0.1,'Bryan Mosquera',color='black',size=12)
fig.text(0.18,0.1,'Data: Opta',color='black',size=12)
pitch = VerticalPitch(pitch_type='opta', pitch_color='#ead8cd', half=True)
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)

df_missed_shots = df1.loc[df1['type_displayName']=='MissedShots']
a_missed = df_missed_shots.loc[df_missed_shots['teamId']==a_id]
# Update 'endX' column for unblocked shots
a_shots.loc[a_shots['blockedX'].isna(), 'endX'] = 100
a_goals.loc[a_goals['blockedX'].isna(), 'endX'] = 100
a_missed.loc[a_missed['blockedX'].isna(), 'endX'] = 100

# Differentiate between blocked shots, goals, on-target shots, and missed shots
a_shots['on_target'] = np.where((a_shots['endX'] >= 100) & (a_shots['goalMouthY'] >= 45) & (a_shots['goalMouthY'] <= 55), True, False)
a_shots['missed'] = ~a_shots['on_target']

# Plot goals as red arrows
pitch.scatter(a_goals['x'], a_goals['y'],color='red',ax=ax)
pitch.arrows(a_goals['x'], a_goals['y'], a_goals['endX'], a_goals['goalMouthY'],
             width=2, headwidth=10, headlength=10, color='red', label='Goals',ax=ax)

# Plot missed shots (off target) as blue arrows
#missed_shots = h_shots[(h_shots['missed']) & (h_shots['isGoal'] != 1)]
pitch.scatter(a_missed.x,a_missed.y,color='blue',ax=ax)
#pitch.arrows(missed_shots['x'], missed_shots['y'], missed_shots['endX'], missed_shots['goalMouthY'],
#             width=2, headwidth=10, headlength=10, color='blue', label='Missed Shots', ax=ax)
pitch.arrows(a_missed.x,a_missed.y,a_missed.endX,a_missed.goalMouthY,
             width=2, headwidth=10, headlength=10, color='blue', label='Missed Shots', ax=ax)

# Plot on-target shots as green arrows (excluding goals)
on_target_shots = a_shots[(a_shots['on_target']) & (a_shots['isGoal'] != 1)]
pitch.scatter(on_target_shots['x'], on_target_shots['y'],color='green',ax=ax)
pitch.arrows(on_target_shots['x'], on_target_shots['y'], on_target_shots['endX'], on_target_shots['goalMouthY'],
             width=2, headwidth=10, headlength=10, color='green', label='Shots On Target', ax=ax)

# Plot blocked shots as black arrows
blocked_shots = a_shots[a_shots['blockedX'].notna()]
pitch.scatter(blocked_shots['x'], blocked_shots['y'],color='black',ax=ax)
pitch.arrows(blocked_shots['x'], blocked_shots['y'], blocked_shots['blockedX'], blocked_shots['blockedY'],
             width=2, headwidth=10, headlength=10, color='black', label='Blocked Shots',ax=ax)
#title
fig.set_facecolor('#ead8cd')
ax_title = ax.set_title(f'{a_team} Shots vs {h_team}', fontsize=20,color='black')

# Add a legend to the plot
ax.legend(facecolor='#ead8cd', edgecolor='black',labelcolor='black',fontsize=15,loc='upper left', handlelength=5)
fig.text(0.82,0.85,f"{competition_date}",ha='right',color='black',size=11)
fig.text(0.8,0.1,'Bryan Mosquera',color='black',size=11)
fig.text(0.18,0.1,'Data: Opta',color='black',size=11)
pitch = VerticalPitch(pitch_type='opta', pitch_color='#000000', half=True,
                      pad_left=-10,pad_right=-10,pad_top=10,pad_bottom=-25,
                      spot_scale=0.005,linewidth=3,line_alpha=0.7, goal_alpha=0.9,
                      goal_type='box',linestyle='--',line_color='aquamarine')
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)

df_missed_shots = df1.loc[df1['type_displayName']=='MissedShots']
h_missed = df_missed_shots.loc[df_missed_shots['teamId']==h_id]
# Update 'endX' column for unblocked shots
h_shots.loc[h_shots['blockedX'].isna(), 'endX'] = 100
h_goals.loc[h_goals['blockedX'].isna(), 'endX'] = 100
h_missed.loc[h_missed['blockedX'].isna(), 'endX'] = 100

# Differentiate between blocked shots, goals, on-target shots, and missed shots
h_shots['on_target'] = np.where((h_shots['endX'] >= 100) & (h_shots['goalMouthY'] >= 45) & (h_shots['goalMouthY'] <= 55), True, False)
h_shots['missed'] = ~h_shots['on_target']

# Plot goals as red arrows
pitch.scatter(h_goals['endX'], h_goals['goalMouthY'],marker='D',color='red',ax=ax)
pitch.lines(h_goals['x'], h_goals['y'], h_goals['endX'], h_goals['goalMouthY'], comet=True,
            transparent=True, color='red', label='Goals',ax=ax)

# Plot missed shots (off target) as blue arrows
#missed_shots = h_shots[(h_shots['missed']) & (h_shots['isGoal'] != 1)]
pitch.scatter(h_missed.endX,h_missed.goalMouthY,color='blue',marker='D',ax=ax)
#pitch.arrows(missed_shots['x'], missed_shots['y'], missed_shots['endX'], missed_shots['goalMouthY'],
#             width=2, headwidth=10, headlength=10, color='blue', label='Missed Shots', ax=ax)
pitch.lines(h_missed.x,h_missed.y,h_missed.endX,h_missed.goalMouthY,
             comet=True, transparent=True, color='blue', label='Missed Shots', ax=ax)

# Plot on-target shots as green arrows (excluding goals)
on_target_shots = h_shots[(h_shots['on_target']) & (h_shots['isGoal'] != 1)]
pitch.scatter(on_target_shots['endX'], on_target_shots['goalMouthY'],color='lime',marker='D',ax=ax)
pitch.lines(on_target_shots['x'], on_target_shots['y'], on_target_shots['endX'], on_target_shots['goalMouthY'],
             comet=True, transparent=True,  color='lime', label='Shots On Target', ax=ax)

# Plot blocked shots as black arrows
blocked_shots = h_shots[h_shots['blockedX'].notna()]
#pitch.scatter(blocked_shots['x'], blocked_shots['y'],color='black',ax=ax)
pitch.scatter(blocked_shots['blockedX'], blocked_shots['blockedY'],marker='D',color='#ead8cd',ax=ax)
pitch.lines(blocked_shots['x'], blocked_shots['y'], blocked_shots['blockedX'], blocked_shots['blockedY'],
             comet=True, transparent=True, color='#ead8cd', label='Blocked Shots',ax=ax)
#title
fig.set_facecolor('#000000')
ax_title = ax.set_title(f'{h_team} Shots vs {a_team}', fontsize=30,color='white')

# Add a legend to the plot
ax.legend(facecolor='#999999', edgecolor='black',labelcolor='black',fontsize=12,loc='upper left', handlelength=5)
fig.text(0.83,0.81,f"{competition_date}",ha='right',color='darkorange',size=13)
fig.text(0.8,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.18,0.1,'Data: Opta',color='white',size=11)
pitch = VerticalPitch(pitch_type='opta', pitch_color='#000000', half=True,
                      pad_left=-10,pad_right=-10,pad_top=10,pad_bottom=-25,
                      spot_scale=0.005,linewidth=3,line_alpha=0.7, goal_alpha=0.9,
                      goal_type='box',linestyle='--',line_color='aquamarine')
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)

df_missed_shots = df1.loc[df1['type_displayName']=='MissedShots']
a_missed = df_missed_shots.loc[df_missed_shots['teamId']==a_id]
# Update 'endX' column for unblocked shots
a_shots.loc[a_shots['blockedX'].isna(), 'endX'] = 100
a_goals.loc[a_goals['blockedX'].isna(), 'endX'] = 100
a_missed.loc[a_missed['blockedX'].isna(), 'endX'] = 100

# Differentiate between blocked shots, goals, on-target shots, and missed shots
a_shots['on_target'] = np.where((a_shots['endX'] >= 100) & (a_shots['goalMouthY'] >= 45) & (a_shots['goalMouthY'] <= 55), True, False)
a_shots['missed'] = ~a_shots['on_target']

# Plot goals as red arrows
pitch.scatter(a_goals['endX'], a_goals['goalMouthY'],marker='D',color='red',ax=ax)
pitch.lines(a_goals['x'], a_goals['y'], a_goals['endX'], a_goals['goalMouthY'], comet=True,
            transparent=True, color='red', label='Goals',ax=ax)

# Plot missed shots (off target) as blue arrows
#missed_shots = h_shots[(h_shots['missed']) & (h_shots['isGoal'] != 1)]
pitch.scatter(a_missed.endX,a_missed.goalMouthY,color='blue',marker='D',ax=ax)
#pitch.arrows(missed_shots['x'], missed_shots['y'], missed_shots['endX'], missed_shots['goalMouthY'],
#             width=2, headwidth=10, headlength=10, color='blue', label='Missed Shots', ax=ax)
pitch.lines(a_missed.x,a_missed.y,a_missed.endX,a_missed.goalMouthY,
             comet=True, transparent=True, color='blue', label='Missed Shots', ax=ax)

# Plot on-target shots as green arrows (excluding goals)
on_target_shots = a_shots[(a_shots['on_target']) & (a_shots['isGoal'] != 1)]
pitch.scatter(on_target_shots['endX'], on_target_shots['goalMouthY'],color='lime',marker='D',ax=ax)
pitch.lines(on_target_shots['x'], on_target_shots['y'], on_target_shots['endX'], on_target_shots['goalMouthY'],
             comet=True, transparent=True,  color='lime', label='Shots On Target', ax=ax)

# Plot blocked shots as black arrows
blocked_shots = a_shots[a_shots['blockedX'].notna()]
#pitch.scatter(blocked_shots['x'], blocked_shots['y'],color='black',ax=ax)
pitch.scatter(blocked_shots['blockedX'], blocked_shots['blockedY'],marker='D',color='#ead8cd',ax=ax)
pitch.lines(blocked_shots['x'], blocked_shots['y'], blocked_shots['blockedX'], blocked_shots['blockedY'],
             comet=True, transparent=True, color='#ead8cd', label='Blocked Shots',ax=ax)
#title
fig.set_facecolor('#000000')
ax_title = ax.set_title(f'{a_team} Shots vs {h_team}', fontsize=30,color='white')

# Add a legend to the plot
ax.legend(facecolor='#999999', edgecolor='black',labelcolor='black',fontsize=12,loc='upper left', handlelength=5)
fig.text(0.82,0.82,f"{competition_date}",ha='right',color='darkorange',size=13)
fig.text(0.8,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.18,0.1,'Data: Opta',color='white',size=11)
##Goalkeeper Action Map
h_gk = gk_actions[gk_actions.teamId==h_id]
h_pickups = h_gk.loc[gk_actions.type_displayName=='KeeperPickup']
h_sweeps = h_gk.loc[gk_actions.type_displayName=='KeeperSweeper']
h_claims = h_gk.loc[gk_actions.type_displayName=='Claim']
h_punches = h_gk.loc[gk_actions.type_displayName=='Punch']
a_gk = gk_actions[gk_actions.teamId==a_id]
a_pickups = a_gk.loc[gk_actions.type_displayName=='KeeperPickup']
a_sweeps = a_gk.loc[gk_actions.type_displayName=='KeeperSweeper']
a_claims = a_gk.loc[gk_actions.type_displayName=='Claim']
a_punches = a_gk.loc[gk_actions.type_displayName=='Punch']
pitch = Pitch(pitch_type='opta', pitch_color='#ead8cd',
             spot_scale=0.01,linewidth=3, line_color='white',
              stripe_color='#c38f6f', stripe=True)
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
pitch.scatter(h_pickups.x,h_pickups.y,color='black',label='Pickups',ax=ax)
pitch.scatter(h_sweeps.x,h_sweeps.y,color='red',label='Sweeps',ax=ax)
pitch.scatter(h_claims.x,h_claims.y,color='green',label='Claims',ax=ax)
pitch.scatter(h_punches.x,h_punches.y,color='blue',label='Punches',ax=ax)
pitch.scatter(100-a_pickups.x,100-a_pickups.y,color='black',ax=ax)
pitch.scatter(100-a_sweeps.x,100-a_sweeps.y,color='red',ax=ax)
pitch.scatter(100-a_claims.x,100-a_claims.y,color='green',ax=ax)
pitch.scatter(100-a_punches.x,100-a_punches.y,color='blue',ax=ax)
ax.legend(facecolor='#ead8cd', edgecolor='black',labelcolor='black',fontsize=15,loc='center', handlelength=5)
fig.set_facecolor('#ead8cd')
ax_title = ax.set_title(f'Goalkeeper Actions Map', fontsize=22,color='black',fontweight='bold')
fig.text(0.88,0.85,f"{competition_date}",ha='right',color='black',size=12)
fig.text(0.79,0.1,'Bryan Mosquera',color='black',size=12)
fig.text(0.2,0.1,'Data: Opta',color='black',size=12)
# Initialize the pitch
pitch = Pitch(pitch_type='opta', pitch_color='green',spot_scale=0.01,linewidth=3, line_color='lavender')
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)

# Set with all action types
action_types = {'Pickups', 'Sweeps', 'Claims', 'Punches'}

# Iterate over each row in the DataFrame
for index, row in gk_actions.iterrows():
    # Skip rows that do not match the four action types
    if row['type_displayName'] not in ['KeeperPickup', 'KeeperSweeper', 'Claim', 'Punch']:
        continue

    # Initialize variables
    action_label = None

    # Determine the action type and corresponding color
    if row['type_displayName'] == 'KeeperPickup':
        color = 'black'
        action_type = 'Pickups'
    elif row['type_displayName'] == 'KeeperSweeper':
        color = 'red'
        action_type = 'Sweeps'
    elif row['type_displayName'] == 'Claim':
        color = 'blue'
        action_type = 'Claims'
    elif row['type_displayName'] == 'Punch':
        color = 'blue'
        action_type = 'Punches'

    # Check if this action type is to be plotted for the first time
    if action_type in action_types:
        action_label = action_type
        action_types.remove(action_type)

    # Adjust coordinates for the away team
    x, y = (row['x'], row['y']) if row['teamId'] == h_id else (100 - row['x'], 100 - row['y'])

    # Plot the action
    pitch.scatter(x, y, color=color, label=action_label, ax=ax)

# Set legend and other plot properties
ax.legend(facecolor='#ead8cd', edgecolor='black', labelcolor='black', fontsize=15, loc='center', handlelength=5)
fig.set_facecolor('green')
ax_title = ax.set_title('Goalkeeper Defensive Actions Map', fontsize=20, color='black',fontweight='bold')
fig.text(0.88,0.85,f"{competition_date}",ha='right',color='white',size=11)
fig.text(0.75,0.1,'Bryan Mosquera',color='white',size=13)
fig.text(0.3,0.1,'Data: Opta',color='white',size=12)
#Heatmap 
# Initialize the pitch
pitch = Pitch(pitch_type='opta', pitch_color='green',spot_scale=0.01,linewidth=3, line_color='lavender')
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)

# Set with all action types
action_types = {'Pickups', 'Sweeps', 'Claims', 'Punches'}

# Iterate over each row in the DataFrame
for index, row in gk_actions.iterrows():
    # Skip rows that do not match the four action types
    if row['type_displayName'] not in ['KeeperPickup', 'KeeperSweeper', 'Claim', 'Punch']:
        continue

    # Initialize variables
    action_label = None

    # Determine the action type and corresponding color
    if row['type_displayName'] == 'KeeperPickup':
        color = 'black'
        action_type = 'Pickups'
    elif row['type_displayName'] == 'KeeperSweeper':
        color = 'red'
        action_type = 'Sweeps'
    elif row['type_displayName'] == 'Claim':
        color = 'blue'
        action_type = 'Claims'
    elif row['type_displayName'] == 'Punch':
        color = 'blue'
        action_type = 'Punches'

    # Check if this action type is to be plotted for the first time
    if action_type in action_types:
        action_label = action_type
        action_types.remove(action_type)

    # Adjust coordinates for the away team
    x, y = (row['x'], row['y']) if row['teamId'] == h_id else (100 - row['x'], 100 - row['y'])

    # Plot the action
    pitch.scatter(x, y, color=color, label=action_label, ax=ax)

# Set legend and other plot properties
ax.legend(facecolor='#ead8cd', edgecolor='black', labelcolor='black', fontsize=15, loc='center', handlelength=5)
fig.set_facecolor('green')
ax_title = ax.set_title('Goalkeeper Defensive Actions Map', fontsize=20, color='black',fontweight='bold')
fig.text(0.88,0.85,f"{competition_date}",ha='right',color='white',size=11)
fig.text(0.75,0.1,'Bryan Mosquera',color='white',size=13)
fig.text(0.3,0.1,'Data: Opta',color='white',size=12)
fig ,ax = plt.subplots(figsize=(13.5,8))
fig.set_facecolor('#ffff99')


#this is how we create the pitch
pitch = Pitch(pitch_type='opta',pitch_color='#ffff99',line_zorder=2)

#Draw the pitch on the ax figure as well as invert the axis for this specific pitch
pitch.draw(ax=ax)

#Create the heatmap
kde = sns.kdeplot(data=a_touches,
        x='x',
        y='y',
        fill = True,
        #shade_lowest=False,
        
        alpha=.7,
        n_levels=10,
        cmap='YlOrBr'
)

plt.title(f"{a_team} Touch Heatmap vs {h_team}",size=20,fontweight='bold')
fig.text(0.84,0.86,f"{competition_date}",ha='right',color='black',size=10)
fig.text(0.70,0.1,'Bryan Mosquera',color='black',size=12)
fig.text(0.2,0.1,'Data: Opta',color='black',size=12)
fig ,ax = plt.subplots(figsize=(13.5,8))
fig.set_facecolor('#000000')

#this is how we create the pitch
pitch = Pitch(pitch_type='opta',pitch_color='#000000',line_zorder=2)

#Draw the pitch on the ax figure as well as invert the axis for this specific pitch
pitch.draw(ax=ax)

#Create the heatmap
kde = sns.kdeplot(data=h_passes,
        x='x',
        y='y',
        fill = True,
        #shade_lowest=False,
        
        alpha=.7,
        n_levels=30,
        cmap='magma'
)

plt.title(f"{h_team} Touch Heatmap vs {a_team}",size=20,color='white',fontweight='bold')
fig.text(0.84,0.86,f"{competition_date}",ha='right',color='aquamarine',size=10)
fig.text(0.70,0.1,'Bryan Mosquera',color='lavender',size=12)
fig.text(0.3,0.1,'Data: Opta',color='lavender',size=12)
fig ,ax = plt.subplots(figsize=(13.5,8))
fig.set_facecolor('#000000')

#this is how we create the pitch
pitch = Pitch(pitch_type='opta',pitch_color='#000000',line_zorder=2)

#Draw the pitch on the ax figure as well as invert the axis for this specific pitch
pitch.draw(ax=ax)

#Create the heatmap
kde = sns.kdeplot(data=a_passes,
        x='x',
        y='y',
        fill = True,
        #shade_lowest=False,
        
        alpha=.7,
        n_levels=30,
        cmap='magma'
)

plt.title(f"{a_team} Touch Heatmap vs {h_team}",size=20,color='white',fontweight='bold')
fig.text(0.84,0.86,f"{competition_date}",ha='right',color='aquamarine',size=10)
fig.text(0.67,0.1,'Bryan Mosquera',color='lavender',size=12)
fig.text(0.3,0.1,'Data: Opta',color='lavender',size=12)
fig ,ax = plt.subplots(figsize=(13.5,8))
fig.set_facecolor('#000000')

#this is how we create the pitch
pitch = Pitch(pitch_type='opta',pitch_color='#000000',half=True,line_zorder=2)

#Draw the pitch on the ax figure as well as invert the axis for this specific pitch
pitch.draw(ax=ax)

#Create the heatmap
kde = sns.kdeplot(data=h_shots,
        x='x',
        y='y',
        fill = True,
        #shade_lowest=False,
        alpha=.7,
        n_levels=200,
        cmap='viridis'
)

plt.title(f"{h_team} Shot Heatmap vs {a_team}",size=20,fontweight='bold',color='white')
fig.text(0.6,0.86,f"{competition_date}",ha='right',color='aquamarine',size=10)
fig.text(0.60,0.1,'Bryan Mosquera',color='lavender',size=10)
fig.text(0.3,0.1,'Data: Opta',color='lavender',size=10)
fig ,ax = plt.subplots(figsize=(13.5,8))
fig.set_facecolor('#000000')

#this is how we create the pitch
pitch = Pitch(pitch_type='opta',pitch_color='#000000',half=True,line_zorder=2)

#Draw the pitch on the ax figure as well as invert the axis for this specific pitch
pitch.draw(ax=ax)

#Create the heatmap
kde = sns.kdeplot(data=a_shots,
        x='x',
        y='y',
        fill = True,
        #shade_lowest=False,
        alpha=.7,
        n_levels=200,
        cmap='viridis'
)

plt.title(f"{a_team} Shot Heatmap vs {h_team}",size=20,fontweight='bold',color='white')
fig.text(0.6,0.86,f"{competition_date}",ha='right',color='aquamarine',size=10)
fig.text(0.60,0.1,'Bryan Mosquera',color='lavender',size=10)
fig.text(0.3,0.1,'Data: Opta',color='lavender',size=10)
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


# setup pitch
pitch = Pitch(pitch_type='opta', line_zorder=2,
              pitch_color='#000000', line_color='#efefef',pad_top=10)
# draw
fig, ax = pitch.draw(figsize=(7.5, 4.125))
fig.set_facecolor('#000000')
bin_statistic = pitch.bin_statistic(h_passes.x, h_passes.y, statistic='count', bins=(25, 25))
bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
# Add the colorbar and format off-white
#cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
#cbar.outline.set_edgecolor('#efefef')
#cbar.ax.yaxis.set_tick_params(color='#efefef')
#ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

ax_title = ax.set_title(f"{h_team} Pass Heatmap vs {a_team}", fontsize=18,color='white',fontweight='bold')
x, y, text = 37, 107, "Starting locations"
ax.text(x, y, text,color='white',style='italic')
fig.text(0.35, 0.84, f"{competition_date}",color='white')
#fig.text(0.6,0.03,'Bryan Mosquera',color='white')
#fig.text(0.2,0.03,'Data: Opta',color='white')
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


# setup pitch
pitch = Pitch(pitch_type='opta', line_zorder=2,
              pitch_color='#000000', line_color='#efefef',pad_top=10)
# draw
fig, ax = pitch.draw(figsize=(7.5, 4.125))
fig.set_facecolor('#000000')
bin_statistic = pitch.bin_statistic(a_passes.x, a_passes.y, statistic='count', bins=(25, 25))
bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
# Add the colorbar and format off-white
#cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
#cbar.outline.set_edgecolor('#efefef')
#cbar.ax.yaxis.set_tick_params(color='#efefef')
#ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

ax_title = ax.set_title(f"{a_team} Pass Heatmap vs {h_team}", fontsize=18,color='white',fontweight='bold')
x, y, text = 37, 107, "Starting locations"
ax.text(x, y, text,color='white',style='italic')
fig.text(0.35, 0.84, f"{competition_date}",color='white')
#fig.text(0.7,0.03,'Bryan Mosquera',color='white')
#fig.text(0.2,0.03,'Data: Opta',color='white')
# path effects
import matplotlib.patheffects as path_effects
path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'),
            path_effects.Normal()]

# see the custom colormaps example for more ideas on setting colormaps
pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                       ['#15242e', '#4393c4'], N=10)
pitch = VerticalPitch(pitch_type='opta', line_zorder=2, pitch_color='#000000',pad_top=15)
fig, ax = pitch.draw(figsize=(6, 6))
fig.set_facecolor('#000000')
bin_statistic = pitch.bin_statistic(h_touches.x, h_touches.y, statistic='count', bins=(6, 5), normalize=True)
pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9')
labels = pitch.label_heatmap(bin_statistic, color='#ead8cd', fontsize=18,
                             ax=ax, ha='center', va='center',
                             str_format='{:.0%}', path_effects=path_eff)

ax.set_title(f"{h_team} Touches Heatmap vs {a_team}",color='white',fontweight='bold',fontsize=14)
fig.text(0.3, 0.90, f"{competition_date}",color='white',fontsize=10)
fig.text(0.60,0.01,'Bryan Mosquera',color='white')
fig.text(0.23,0.01,'Data: Opta',color='white')
# path effects
import matplotlib.patheffects as path_effects
path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'),
            path_effects.Normal()]

# see the custom colormaps example for more ideas on setting colormaps
pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                       ['#15242e', '#4393c4'], N=10)
pitch = VerticalPitch(pitch_type='opta', line_zorder=2, pitch_color='#000000',pad_top=15)
fig, ax = pitch.draw(figsize=(6, 6))
fig.set_facecolor('#000000')
bin_statistic = pitch.bin_statistic(a_touches.x, a_touches.y, statistic='count', bins=(6, 5), normalize=True)
pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9')
labels = pitch.label_heatmap(bin_statistic, color='#ead8cd', fontsize=18,
                             ax=ax, ha='center', va='center',
                             str_format='{:.0%}', path_effects=path_eff)

ax.set_title(f"{a_team} Touches Heatmap vs {h_team}",color='white',fontweight='bold',fontsize=16)
fig.text(0.3, 0.91, f"{competition_date}",color='white',fontsize=10)
fig.text(0.65,0.01,'Bryan Mosquera',color='white')
fig.text(0.23,0.01,'Data: Opta',color='white')
pitch = VerticalPitch(pitch_type='opta', line_zorder=2, pitch_color='#f2edf1',pad_top=15)
fig, ax = pitch.draw(figsize=(6, 6))
fig.set_facecolor('#f2edf1')
bin_x = np.linspace(pitch.dim.left, pitch.dim.right, num=7)
bin_y = np.sort(np.array([pitch.dim.bottom, pitch.dim.six_yard_bottom,
                          pitch.dim.six_yard_top, pitch.dim.top]))
bin_statistic = pitch.bin_statistic(h_passes.x, h_passes.y, statistic='count',
                                    bins=(bin_x, bin_y), normalize=True)
pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9')
labels2 = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                              ax=ax, ha='center', va='center',
                              str_format='{:.0%}', path_effects=path_eff)

ax.set_title(f"{h_team} Passes Heatmap vs {a_team}",color='black',fontweight='bold',fontsize=14)
fig.text(0.4, 0.91, "Starting locations",color='red',style='italic')
fig.text(0.3, 0.88, f"{competition_date}",color='black')
#fig.text(0.63,0.01,'Bryan Mosquera',color='black')
#fig.text(0.23,0.01,'Data: Opta',color='black')
pitch = VerticalPitch(pitch_type='opta', line_zorder=2, pitch_color='#f2edf1',pad_top=15)
fig, ax = pitch.draw(figsize=(6, 6))
fig.set_facecolor('#f2edf1')
bin_x = np.linspace(pitch.dim.left, pitch.dim.right, num=7)
bin_y = np.sort(np.array([pitch.dim.bottom, pitch.dim.six_yard_bottom,
                          pitch.dim.six_yard_top, pitch.dim.top]))
bin_statistic = pitch.bin_statistic(a_passes.x, a_passes.y, statistic='count',
                                    bins=(bin_x, bin_y), normalize=True)
pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9')
labels2 = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                              ax=ax, ha='center', va='center',
                              str_format='{:.0%}', path_effects=path_eff)

ax.set_title(f"{a_team} Passes Heatmap vs {h_team}",color='black',fontweight='bold',fontsize=14)
fig.text(0.4, 0.91, "Starting locations",color='Red',style='italic')
fig.text(0.3, 0.88, f"{competition_date}",color='black')
#fig.text(0.63,0.01,'Bryan Mosquera',color='black')
#fig.text(0.23,0.01,'Data: Opta',color='black')
# see the custom colormaps example for more ideas on setting colormaps
pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                       ['#15242e', '#4393c4'], N=10)

path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]

# setup pitch
pitch = VerticalPitch(pitch_type='opta', line_zorder=2,
                      pitch_color='#000000', line_color='white',pad_top=15)
fig, ax = pitch.draw(figsize=(6, 6))
fig.set_facecolor('#000000')

# draw

bin_statistic = pitch.bin_statistic_positional(h_touches.x, h_touches.y, statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
pitch.scatter(a_touches.x, a_touches.y, c='white', s=2, ax=ax)
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                             ax=ax, ha='center', va='center',
                             str_format='{:.0%}', path_effects=path_eff)
ax.set_title(f"{h_team} Touches Heatmap vs {a_team}",color='lavender',fontweight='bold',fontsize=15)
fig.text(0.3, 0.90, f"{competition_date}",color='aquamarine')
#fig.text(0.60,0.01,'Bryan Mosquera',color='lavender')
#fig.text(0.23,0.01,'Data: Opta',color='lavender')
# see the custom colormaps example for more ideas on setting colormaps
pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                       ['#15242e', '#4393c4'], N=10)

path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
            path_effects.Normal()]

# setup pitch
pitch = VerticalPitch(pitch_type='opta', line_zorder=2,
                      pitch_color='#000000', line_color='white',pad_top=15)
fig, ax = pitch.draw(figsize=(6, 6))
fig.set_facecolor('#000000')

# draw

bin_statistic = pitch.bin_statistic_positional(a_touches.x, a_touches.y, statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')
pitch.scatter(a_touches.x, a_touches.y, c='white', s=2, ax=ax)
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                             ax=ax, ha='center', va='center',
                             str_format='{:.0%}', path_effects=path_eff)
ax.set_title(f"{a_team} Touches Heatmap vs {h_team}",color='lavender',fontweight='bold',fontsize=15)
fig.text(0.3, 0.89, f"{competition_date}",color='aquamarine')
#fig.text(0.60,0.01,'Bryan Mosquera',color='lavender')
#fig.text(0.23,0.01,'Data: Opta',color='lavender')
from matplotlib.colors import to_rgba
# create plot
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
axes = axes.flat
plt.tight_layout()
fig.set_facecolor("#280000")

# plot variables
main_color = 'white'

# home team viz
pass_network_visualization(axes[0], home_passes_between_df, home_average_locs_and_count_df)
axes[0].set_title(teams_dict[home_team_id], color=main_color, fontsize=19)

# away team viz
pass_network_visualization(axes[1], away_passes_between_df, away_average_locs_and_count_df, flipped=True)
axes[1].set_title(teams_dict[away_team_id], color=main_color, fontsize=19)

plt.suptitle(f"{teams_dict[home_team_id]} - {teams_dict[away_team_id]}", color=main_color, fontsize=42)
subtitle = "Passing networks and top combinations by volume of passes"
plt.text(-10, 120, subtitle, horizontalalignment='center', verticalalignment='center', color=main_color, fontsize=17)
#plt.savefig('output.png', bbox_inches='tight')
plt.show()
#mask_homePasses = (df.type_name == 'Pass') & (df.team_name == "England Women's") & (df.sub_type_name != "Throw-in")
#home_passes = df.loc[mask_homePasses, ['x', 'y', 'endX', 'endY', 'name']]
#get the list of all players who made a pass
names = h_passes['name'].unique()

#draw 4x4 pitches
pitch = Pitch(pitch_type='opta', line_color='white', pad_top=25, pitch_color='#000000')
fig, axs = pitch.grid(ncols = 4, nrows = 4, grid_height=0.85, title_height=0.03, axis=False,
                     endnote_height=0.01, title_space=0.02, endnote_space=0.01)
fig.set_facecolor("#000000")

#for each player
for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
    #put player name over the plot
    ax.text(50, 110, name,
            ha='center', va='center', fontsize=10,color='white')
    #take only passes by this player
    player_df = h_passes.loc[h_passes["name"] == name]
    #scatter
    #pitch.scatter(player_df.x, player_df.y, alpha = 0.2, s = 50, color = "blue", ax=ax)
    #plot arrow
    pitch.arrows(player_df.x, player_df.y,
            player_df.endX, player_df.endY, color = "springgreen", ax=ax, width=1)

#We have more than enough pitches - remove them
for ax in axs['pitch'][-1, 16 - len(names):]:
    ax.remove()

#Another way to set title using mplsoccer
axs['title'].text(0.5, 0.5, f"{h_team} Passing against {a_team}", ha='center', va='center', fontsize=20,color='white')
fig.text(0.4,0.91,f"{competition_date}",color='aquamarine')
#fig.text(0.03,0.03,"Data: Opta",color='white')
#fig.text(0.9,0.03,"Bryan Mosquera",color='white')
#mask_homePasses = (df.type_name == 'Pass') & (df.team_name == "England Women's") & (df.sub_type_name != "Throw-in")
#home_passes = df.loc[mask_homePasses, ['x', 'y', 'endX', 'endY', 'name']]
#get the list of all players who made a pass
names = a_passes['name'].unique()

#draw 4x4 pitches
pitch = Pitch(pitch_type='opta', line_color='white', pad_top=25, pitch_color='#000000')
fig, axs = pitch.grid(ncols = 4, nrows = 4, grid_height=0.85, title_height=0.03, axis=False,
                     endnote_height=0.01, title_space=0.02, endnote_space=0.01)
fig.set_facecolor("#000000")

#for each player
for name, ax in zip(names, axs['pitch'].flat[:len(names)]):
    #put player name over the plot
    ax.text(50, 110, name,
            ha='center', va='center', fontsize=10, color='white')
    #take only passes by this player
    player_df = a_passes.loc[a_passes["name"] == name]
    #scatter
    #pitch.scatter(player_df.x, player_df.y, alpha = 0.2, s = 50, color = "blue", ax=ax)
    #plot arrow
    pitch.arrows(player_df.x, player_df.y,
            player_df.endX, player_df.endY, color = "springgreen", ax=ax, width=1)

#We have more than enough pitches - remove them
for ax in axs['pitch'][-1, 16 - len(names):]:
    ax.remove()

#Another way to set title using mplsoccer
axs['title'].text(0.5, 0.5, f"{a_team} Passing against {h_team}", ha='center', va='center', fontsize=20,color='white')
fig.text(0.4,0.91,f"{competition_date}",color='aquamarine')
fig.text(0.03,0.03,"Data: Opta",color='white')
fig.text(0.9,0.03,"Bryan Mosquera",color='white')
##Events Map Player
player_name='Galeno'
player_team = 'Porto'
player_df = df1.loc[df1['name']==player_name].copy()
#player_passes = player_df.loc[player_df['type_displayName']=='Pass'].copy()
player_passes = df_passes[df_passes['name']==player_name]
#passes IN opponent's half
player_opphalf_passes = player_passes[player_passes['x']>=50]
#passes in their own half
player_ownhalf_passes = player_passes[player_passes['x']<50]
#passes INTO the final third
player_final_3rd_passes = player_passes[player_passes['endX']>=66]
player_final_3rd_passes = player_final_3rd_passes[player_final_3rd_passes['x']<=66]
#completed passes into the final third
player_comp_final3rd_passes = player_final_3rd_passes[player_final_3rd_passes['outcomeType_displayName']=='Successful']
#passes into the opponent's box
passes_within_box = (
    (player_passes['endX'] >= 85) & 
    (player_passes['endY'] > 20) & 
    (player_passes['endY'] < 80)
)

passes_start_inside_box = (
    (player_passes['x'] > 85) & 
    (player_passes['y'] > 20) & 
    (player_passes['y'] < 80)
)

player_passes_intobox = player_passes[passes_within_box & ~passes_start_inside_box]
#player_passes_intobox = player_passes_intobox[player_passes_intobox['x']<82.9]
#completed/successful passes into the opponent's box
player_comp_passes_intobox = player_passes_intobox[player_passes_intobox['outcomeType_displayName']=='Successful']

#take-ons/dribbles
player_dribbles = player_df.loc[player_df['type_displayName']=='TakeOn'].copy()
#touches
player_touches = player_df.loc[player_df['isTouch']==True].copy()
player_touches2 = player_df.loc[player_df['type_displayName']=='BallTouch'].copy()
#dispossessions
player_dispossessions = player_df.loc[player_df['type_displayName']=='Dispossessed'].copy()
#shots
player_shots = player_df.loc[player_df['isShot']==True].copy()
#goals
player_goals = player_df.loc[player_df['isGoal']==True].copy()
#tackles
player_tackles = player_df.loc[player_df.type_displayName=='Tackle'].copy()
#challenges
player_challenges = player_df.loc[player_df.type_displayName=='Challenge'].copy()
#clearances
player_clearances = player_df.loc[player_df.type_displayName=='Clearance'].copy()
#recoveries
player_recoveries = player_df.loc[player_df.type_displayName=='BallRecovery'].copy()
#blocked passes
player_blocked_passes = player_df.loc[player_df.type_displayName=='BlockedPass'].copy()
#interceptions
player_interceptions = player_df.loc[player_df.type_displayName=='Interception'].copy()
#aerial duels
player_aerials = player_df.loc[player_df['type_displayName']=='Aerial'].copy()
#all defensive actions: 
player_defensive_actions = player_df.loc[(player_df.type_displayName=='Challenge') |  
        (player_df.type_displayName=='BlockedPass')|
        (player_df.type_displayName=='Tackle')|
        (player_df.type_displayName=='Interception')|
        (player_df.type_displayName=='Aerial')|
        (player_df.type_displayName=='Foul')].copy()
#player_passes['beginning'] = np.sqrt(np.square(100 - player_passes['x']) + np.square(50 - player_passes['y']))
#player_passes['end'] = np.sqrt(np.square(100 - player_passes['endX']) + np.square(50 - player_passes['endY']))
#player_passes.loc[:, 'progressive'] = (player_passes['end'] / player_passes['beginning'] < 0.75) & (player_passes['x']<player_passes['endX'])
#player_progressive_passes = player_passes[player_passes['progressive']==True]

#player_passes = df_passes[df_passes['name']==player_name]
player_prog_passes = player_passes[player_passes['progressive']==True]
pitch = VerticalPitch(positional=True, pitch_type='opta', half=True, 
                      pitch_color='#000000', positional_linestyle='--',pad_top=10)
fig, ax = pitch.draw(figsize=(16, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
pitch.arrows(player_comp_passes_intobox.x, player_comp_passes_intobox.y,
             player_comp_passes_intobox.endX, player_comp_passes_intobox.endY, width=2,
             headwidth=5, headlength=5, color='blue', label='completed passes',zorder=3, ax=ax)
pitch.arrows(player_passes_intobox.x, player_passes_intobox.y,
             player_passes_intobox.endX, player_passes_intobox.endY, width=2,
             headwidth=5, headlength=5, color='white',alpha=.7, label='missed passes',ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=10, loc='upper left', handlelength=4)
ax_title = ax.set_title(f"{player_name} passes into the box", fontsize=23,color='white')
fig.text(0.72,0.85,f"{player_team}",ha='right',color='white',size=17)
fig.text(0.72,0.81,f"{competition_date}",ha='right',color='aquamarine',size=11)
fig.text(0.64,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.3,0.1,'Data: Opta',color='white',size=11)
pitch = Pitch(positional=True, pitch_type='opta', 
                      pitch_color='#000000', positional_linestyle='--',pad_top=12)
fig, ax = pitch.draw(figsize=(19, 9), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
# Plot the completed passes
pitch.arrows(player_final_3rd_passes.x, player_final_3rd_passes.y,
             player_final_3rd_passes.endX, player_final_3rd_passes.endY, width=2,
             headwidth=5, headlength=5, color='white', label='missed passes',ax=ax)
pitch.arrows(player_comp_final3rd_passes.x, player_comp_final3rd_passes.y,
             player_comp_final3rd_passes.endX, player_comp_final3rd_passes.endY, width=2,
             headwidth=5, headlength=5, color='springgreen', label='completed passes',zorder=3, ax=ax)
ax.legend(facecolor='#999999', edgecolor='None', fontsize=10, loc='upper left', handlelength=4)
ax_title = ax.set_title(f"{player_name}'s passes into final third", fontsize=20,color='white')
fig.text(0.75,0.85,f"{player_team}",ha='right',color='white',size=11)
fig.text(0.75,0.81,f"{competition_date}",ha='right',color='white',size=11)
fig.text(0.7,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.28,0.1,'Data: Opta',color='white',size=11)
pitch = VerticalPitch(pitch_type='opta', pitch_color='#000000', half=True,
                      pad_left=-10,pad_right=-10,pad_top=10,pad_bottom=-25,
                      spot_scale=0.005,linewidth=3,line_alpha=0.7, goal_alpha=0.9,
                      goal_type='box',linestyle='--',line_color='aquamarine')
fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)

df_missed_shots = df1.loc[df1['type_displayName']=='MissedShots']
player_missed = df_missed_shots.loc[df_missed_shots['name']==player_name]
# Update 'endX' column for unblocked shots
player_shots.loc[player_shots['blockedX'].isna(), 'endX'] = 100
player_goals.loc[player_goals['blockedX'].isna(), 'endX'] = 100
player_missed.loc[player_missed['blockedX'].isna(), 'endX'] = 100

# Differentiate between blocked shots, goals, on-target shots, and missed shots
player_shots['on_target'] = np.where((player_shots['endX'] >= 100) & (player_shots['goalMouthY'] >= 45) & (player_shots['goalMouthY'] <= 55), True, False)
player_shots['missed'] = ~player_shots['on_target']

# Plot goals as red arrows
pitch.scatter(player_goals['endX'], player_goals['goalMouthY'],marker='D',color='red',ax=ax)
pitch.lines(player_goals['x'], player_goals['y'], player_goals['endX'], player_goals['goalMouthY'], comet=True,
            transparent=True, color='red', label='Goals',ax=ax)

# Plot missed shots (off target) as blue arrows
#missed_shots = h_shots[(h_shots['missed']) & (h_shots['isGoal'] != 1)]
pitch.scatter(player_missed.endX,player_missed.goalMouthY,color='blue',marker='D',ax=ax)
#pitch.arrows(missed_shots['x'], missed_shots['y'], missed_shots['endX'], missed_shots['goalMouthY'],
#             width=2, headwidth=10, headlength=10, color='blue', label='Missed Shots', ax=ax)
pitch.lines(player_missed.x,player_missed.y,player_missed.endX,player_missed.goalMouthY,
             comet=True, transparent=True, color='blue', label='Missed Shots', ax=ax)

# Plot on-target shots as green arrows (excluding goals)
on_target_shots = player_shots[(player_shots['on_target']) & (player_shots['isGoal'] != 1)]
pitch.scatter(on_target_shots['endX'], on_target_shots['goalMouthY'],color='lime',marker='D',ax=ax)
pitch.lines(on_target_shots['x'], on_target_shots['y'], on_target_shots['endX'], on_target_shots['goalMouthY'],
             comet=True, transparent=True,  color='lime', label='Shots On Target', ax=ax)

# Plot blocked shots as black arrows
blocked_shots = player_shots[player_shots['blockedX'].notna()]
#pitch.scatter(blocked_shots['x'], blocked_shots['y'],color='black',ax=ax)
pitch.scatter(blocked_shots['blockedX'], blocked_shots['blockedY'],marker='D',color='#ead8cd',ax=ax)
pitch.lines(blocked_shots['x'], blocked_shots['y'], blocked_shots['blockedX'], blocked_shots['blockedY'],
             comet=True, transparent=True, color='#ead8cd', label='Blocked Shots',ax=ax)
#title
fig.set_facecolor('#000000')
ax_title = ax.set_title(f"{player_name} Shotmap", fontsize=26,color='white')

# Add a legend to the plot
ax.legend(facecolor='#999999', edgecolor='black',labelcolor='black',fontsize=12,loc='upper left', handlelength=5)
fig.text(0.8,0.83,f"{player_team}",ha='right',color='white',size=15)
fig.text(0.8,0.78,f"{competition_date}",ha='right',color='White',size=13)
fig.text(0.8,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.18,0.1,'Data: Opta',color='white',size=11)
player_successfulPasses = player_passes[player_passes['outcomeType_displayName']=='Successful']
player_otherPasses = player_passes[player_passes['outcomeType_displayName']=='Unsuccessful']
pitch = Pitch(positional=True,pitch_type='opta', positional_color='grey', positional_linestyle='--', 
              shade_color='#4d4d4d', pitch_color='#000000')
fig, ax = pitch.draw(figsize=(13.5, 8), constrained_layout=True, tight_layout=False)
fig.set_facecolor('#000000')
pitch.scatter(player_successfulPasses.endX, player_successfulPasses.endY,color='aquamarine',marker='D',zorder=3,ax=ax)
pitch.lines(player_successfulPasses.x, player_successfulPasses.y,
             player_successfulPasses.endX, player_successfulPasses.endY, 
            comet=True, transparent=True, color='aquamarine', label='completed passes',zorder=3,ax=ax)
pitch.scatter(player_otherPasses.endX, player_otherPasses.endY,color='tan',marker='D',ax=ax)
pitch.lines(player_otherPasses.x, player_otherPasses.y,
             player_otherPasses.endX, player_otherPasses.endY, 
            comet=True, transparent=True, color='tan', label='missed passes', ax=ax)
ax.legend(facecolor='#4d4d4d', edgecolor='white',fontsize=8, labelcolor='white',loc='upper left', handlelength=4)
plt.suptitle(f"{player_name} Passing",size=25,color='lavender',fontweight='bold')
fig.text(0.84,0.91,f"{player_team}",ha='right',color='white',size=15)
fig.text(0.84,0.87,f"{competition_date}",ha='right',color='white',size=11)
fig.text(0.75,0.1,'Bryan Mosquera',color='white',size=11)
fig.text(0.19,0.1,'Data: Opta',color='white',size=11)
pitch = VerticalPitch(positional=True,pitch_type='opta', positional_color='grey', 
                      positional_linestyle='--', line_zorder=2,positional_zorder=1.5,
                      pitch_color='#000000',pad_top=23)
fig, ax = pitch.draw(figsize=(5, 6))
fig.set_facecolor('#000000')

hexmap = pitch.hexbin(player_passes.x, player_passes.y, ax=ax, edgecolors='#f4f4f4',
                      gridsize=(8, 8), cmap='YlOrRd')
ax.set_title(f"{player_name} Passes Heatmap",color='white',fontweight='bold',size=17)
fig.text(0.38, 0.91, "Starting locations",color='white',style='italic')
fig.text(0.25, 0.84, f"{competition_date}",color='aquamarine')
fig.text(0.44, 0.87, f"{player_team}",color='white',size=15)
#fig.text(0.60,0.02,'Bryan Mosquera',color='white')
#fig.text(0.2,0.02,'Data: Opta',color='white')
pitch = Pitch(positional=True,pitch_type='opta', positional_color='grey', 
              positional_linestyle='--', pitch_color='#90EE90',pad_top=10)
fig, ax = pitch.draw(figsize=(8, 6))
fig.set_facecolor('#90EE90')
hull = pitch.convexhull(player_passes.x, player_passes.y)
poly = pitch.polygon(hull, ax=ax, edgecolor='cornflowerblue', facecolor='cornflowerblue', alpha=0.3)
scatter = pitch.scatter(player_passes.x, player_passes.y, ax=ax, edgecolor='black', facecolor='cornflowerblue')
ax_title = ax.set_title(f"{player_name} Passing - Convex Hull", fontsize=23, color='black',fontweight='bold')
fig.text(0.25, 0.88, f"{player_team} - {competition_date}",color='black',size=12)
#fig.text(0.85,0.03,'Bryan Mosquera',color='black',size=12)
#fig.text(0.04,0.03,'Data: Opta',color='black',size=12)
# Define a dictionary for action types and colors
action_colors = {
    'BlockedPass': 'yellow',
    'Interception': 'lime',
    'Foul': 'pink',
    'Aerial': 'aquamarine',
    'Tackle': 'darkorange',
    'Challenge': 'green', 
}

# Initialize the pitch
pitch = Pitch(pitch_type='opta', pitch_color='#000000')
fig, ax = pitch.draw(figsize=(14, 9), constrained_layout=True, tight_layout=False)

# Iterate over action types and plot
for action_type, color in action_colors.items():
    action_data = player_defensive_actions[player_defensive_actions['type_displayName'] == action_type]
    if not action_data.empty:
        pitch.scatter(action_data.x, action_data.y, color=color, label=action_type,marker='D',ax=ax)

# Legend and title
#ax.legend(facecolor='#ead8cd', edgecolor='black', labelcolor='black', fontsize=12, loc='upper right', handlelength=5)
#fig.legend(facecolor='#ead8cd', edgecolor='black', labelcolor='black', fontsize=12, loc='outside upper right', handlelength=5)
plt.legend(bbox_to_anchor=(1, .7), loc='center left', borderaxespad=0)
fig.set_facecolor('#000000')
ax_title = ax.set_title(f"{player_name} Defensive Actions", fontsize=20, color='white',fontweight='bold')
#fig.text(0.88, 0.8, f"{player_team}",color='lavender',size=20)
fig.text(0.88, 0.7, f"{competition_date}",color='lavender',size=12)
#fig.text(0.88,0.2,'Bryan Mosquera',color='lavender',size=12)
#fig.text(0.88,0.3,'Data: Opta',color='lavender',size=12)
# Initialize the pitch
pitch = Pitch(pitch_type='opta', pitch_color='#000000',pad_top=11)
fig, ax = pitch.draw(figsize=(13, 8), constrained_layout=True, tight_layout=False)

player_unsuccessful_dribbles = player_dribbles.loc[player_dribbles['outcomeType_displayName']=='Unsuccessful'].copy()
player_successful_dribbles = player_dribbles.loc[player_dribbles['outcomeType_displayName']=='Successful'].copy()
pitch.scatter(player_unsuccessful_dribbles.x, player_unsuccessful_dribbles.y, 
              marker='D',color='#ff00ff',alpha=.3,label='Failed Dribble Attempts',ax=ax)
pitch.scatter(player_successful_dribbles.x, player_successful_dribbles.y, 
              marker='D',color='#ff00ff',label='Successful Dribble Attempts',ax=ax)
# Legend and title
ax.legend(facecolor='#000000', edgecolor='lavender', labelcolor='lavender', fontsize=10, loc='upper right', handlelength=6)
fig.set_facecolor('#000000')
ax_title = ax.set_title(f"{player_name} Dribbling", fontsize=22, color='white',fontweight='bold')
fig.text(0.2, 0.85, f"{player_team}",color='white',size=12)
fig.text(0.2, 0.82, f"{competition_date}",color='white',size=12)
#fig.text(0.78,0.1,'Bryan Mosquera',color='lavender',size=12)
#fig.text(0.18,0.1,'Data: Opta',color='lavender',size=12)
fig ,ax = plt.subplots(figsize=(13.5,8))

#this is how we create the pitch
pitch = Pitch(pitch_type='opta',pitch_color='#000000')

fig.set_facecolor('#000000')

#Draw the pitch on the ax figure as well as invert the axis for this specific pitch
pitch.draw(ax=ax)

#Create the heatmap
kde = sns.kdeplot(data=player_touches,
        x='x',
        y='y',
        fill = True,
        #shade_lowest=False,
        
        alpha=.7,
        n_levels=30,
        cmap='magma'
)

plt.suptitle(f"{player_name} Touches Heatmap",size=24,color='white')
plt.title(f"{player_team} - {competition_date}",size=13,color='white')
#fig.text(0.78,0.1,'Bryan Mosquera',color='white')
#fig.text(0.19,0.1,'Data: Opta',color='white')
pitch = VerticalPitch(pitch_type='opta', line_zorder=2, pitch_color='#260000',pad_top=10)
fig, axs = pitch.grid(endnote_height=0.03, endnote_space=0,
                      title_height=0.08, title_space=0,
                      # Turn off the endnote/title axis. I usually do this after
                      # I am happy with the chart layout and text placement
                      axis=False,
                      grid_height=0.84)
fig.set_facecolor('#260000')

# heatmap and labels
bin_statistic = pitch.bin_statistic_positional(player_passes.x, player_passes.y, statistic='count',
                                               positional='full', normalize=True)
pitch.heatmap_positional(bin_statistic, ax=axs['pitch'],
                         cmap='coolwarm', edgecolors='#22312b')
labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
                             ax=axs['pitch'], ha='center', va='center',
                             str_format='{:.0%}', path_effects=path_eff)

# endnote and title
axs['endnote'].text(1, 0.5, 'Bryan Mosquera', va='center', ha='right', fontsize=10,color='#dee6ea')
axs['endnote'].text(0.15, 0.5, 'Data: Opta', va='center', ha='right', fontsize=10,color='#dee6ea')
axs['title'].text(0.5, 0.5, f"{player_name} Passes Heatmap", color='#dee6ea',
                  va='center', ha='center', path_effects=path_eff,fontsize=20)
fig.text(0.35, 0.9, "Starting locations",color='darkorange',style='italic')
fig.text(0.15, 0.87, f"{player_team} {competition_date}",color='aquamarine')
