import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib import rcParams
from matplotlib.patches import Arc
import numpy as np
from highlight_text import fig_text
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import json
# Entering the league's  link
link = "https://understat.com/league/EPL"
res = requests.get(link)
soup = BeautifulSoup(res.content,'lxml')
scripts = soup.find_all('script')
# Get the table 
strings = scripts[2].string 
# Getting rid of unnecessary characters from json data
ind_start = strings.index("('")+2 
ind_end = strings.index("')") 
json_data = strings[ind_start:ind_end] 
json_data = json_data.encode('utf8').decode('unicode_escape')
data = json.loads(json_data)
data
# Creating the dataframe
df =  pd.DataFrame(data['71']['history'])

# Selecting the useful columns 
df = df[['xG','scored','xGA','missed','npxG','npxGA','xpts','npxGD']]

# Creating one new column
df['Match'] = np.arange(1,39)
# Creating the rolling/moving average columns
df['xgSMA'] = df['xG'].rolling(window=3).mean()
df['xgaSMA'] = df['xGA'].rolling(window=3).mean()
df['GSMA'] = df['scored'].rolling(window=3).mean()
df['GASMA'] = df['missed'].rolling(window=3).mean()
# Entering the league's  link
link = "https://understat.com/team/Arsenal/2023"
res = requests.get(link)
soup = BeautifulSoup(res.content,'lxml')
scripts = soup.find_all('script')
# Get the table 
strings = scripts[1].string 
# Getting rid of unnecessary characters from json data
ind_start = strings.index("('")+2 
ind_end = strings.index("')") 
json_data = strings[ind_start:ind_end] 
json_data = json_data.encode('utf8').decode('unicode_escape')
data = json.loads(json_data)
df1 = pd.DataFrame(data)
# Home and Away fixtures
df_a = df1['a'].apply(pd.Series)
df_h = df1['h'].apply(pd.Series)
# Append both together
df_lei = pd.DataFrame(df_h)
display(df_h,df_a)
df1['short_title_h'] = df_h['short_title']
df1['short_title_a'] = df_a['short_title']
df1 = df1[['short_title_h','short_title_a']]
df1['final'] = df1['short_title_h']+df1['short_title_a']
df1['final'] = df1['final'].str.replace("ARS","") # Eliminate the LEI name from the column
teams_played = df1['final'].tolist() # And now create a list containing the teams
display(teams_played)
# plot style 
plt.style.use('fivethirtyeight')
fig,ax = plt.subplots(figsize = (14,8))
# plotting xG and xGA
ax.plot(df.Match,df.xgSMA,label='xG',color='cyan')
ax.plot(df.Match,df.xgaSMA,color='red',label='xGA')
#plotting G and GA
ax.plot(df.Match,df.GSMA,label='G',color='cyan',linestyle='dashed',linewidth=1,alpha=0.8)
ax.plot(df.Match,df.GASMA,color='red',label='GA',linestyle='dashed',linewidth=1,alpha=0.4)
# style 
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
plt.legend()

# title
fig_text(0.08,1.03, s="Arsenal 2023 season\n", fontsize = 25, fontweight = "bold")
fig_text(0.08,0.97, s=" <Expected Goals Scored (xG)> vs <Expected Goals conceded (xG)>",highlight_textprops=[{"color":'cyan'}, {'color':"red"}], fontsize = 20, fontweight="light")

# text
fig_text(0.5,0.01, s="Matches\n", fontsize = 20, fontweight = "bold", color = "black")
fig_text(0.01,0.6, s="Rolling averages\n", fontsize = 20, fontweight = "bold", color = "black",rotation=90)
