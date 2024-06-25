import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import image
df = pd.read_html('https://fbref.com/en/comps/9/Premier-League-Stats')[0]
df.head()
df.rename(columns={'Rk': 'Pos'}, inplace=True)
df['badge'] = df['Squad'].apply(lambda x: f"C:/Users/bryanmosquera/Documents/Football Analytics/Team Logos/{x.lower()}_logo.png")
df['badge'][0]
df.columns
df = df[[
    'Pos', 'badge','Squad', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Pts/MP',
    'xG', 'xGA', 'xGD'
]]
# Background colour
bg_colour = "#FFFFFF"
text_colour = "#000000"
row_colors = {
    "top2": "#E1FABC",
    "top6": "#FFFC97",
    "relegation": "#E79A9A",
    "even": "#E2E2E1",
    "odd": "#B3B0B0",
}
plt.rcParams["text.color"] = text_colour
plt.rcParams["font.family"] = "monospace"
df.columns
col_defs = [
    ColumnDefinition(
        name="Pos",
        textprops={"ha" : "center"},
        width=0.2,
    ),
    ColumnDefinition(
        name="badge",
        textprops={"ha" : "center", "va" : "center", "color" : bg_colour},
        width=0.5,
        plot_fn=image,
    ),
    ColumnDefinition(
        name="Squad",
        textprops={"ha" : "left", "weight" : "bold"},
        width=1.75,
    ),
    ColumnDefinition(
        name="MP",
        group="Matches Played",
        textprops={"ha" : "center"},
        width=0.5,
    ),
    ColumnDefinition(
        name="W",
        group="Matches Played",
        textprops={"ha" : "center"},
        width=0.5,
    ),
    ColumnDefinition(
        name="D",
        group="Matches Played",
        textprops={"ha" : "center"},
        width=0.5,
    ),
    ColumnDefinition(
        name="L",
        group="Matches Played",
        textprops={"ha" : "center"},
        width=0.5,
    ),
    ColumnDefinition(
        name="GF",
        group="Goals",
        textprops={"ha" : "center"},
        width=0.5,
    ),
    ColumnDefinition(
        name="GA",
        group="Goals",
        textprops={"ha" : "center"},
        width=0.5,
    ),
    ColumnDefinition(
        name="GD",
        group="Goals",
        textprops={"ha" : "center"},
        width=0.5,
    ),
    ColumnDefinition(
        name="Pts",
        group="Points",
        textprops={"ha" : "center"},
        width=0.5,
    ),
    ColumnDefinition(
        name="Pts/MP",
        group="Points",
        textprops={"ha" : "center"},
        width=0.5,
    ),
    ColumnDefinition(
        name="xG",
        group="Expected Goals",
        textprops={"ha" : "center", "color" : text_colour, "weight" : "bold", "bbox" : {"boxstyle" : "circle", "pad" : 0.35}},
        cmap=normed_cmap(df["xG"], cmap=matplotlib.cm.PiYG, num_stds=2)
    ),
    ColumnDefinition(
        name="xGA",
        group="Expected Goals",
        textprops={"ha" : "center", "color" : text_colour, "weight" : "bold", "bbox" : {"boxstyle" : "circle", "pad" : 0.35}},
        cmap=normed_cmap(df["xGA"], cmap=matplotlib.cm.PiYG_r, num_stds=2)
    ),
    ColumnDefinition(
        name="xGD",
        group="Expected Goals",
        textprops={"ha" : "center", "color" : text_colour, "weight" : "bold", "bbox" : {"boxstyle" : "circle", "pad" : 0.35}},
        cmap=normed_cmap(df["xGD"], cmap=matplotlib.cm.PiYG, num_stds=2)
    )
]
fig, ax = plt.subplots(figsize=(20, 22))
fig.set_facecolor(bg_colour)
ax.set_facecolor(bg_colour)

table = Table(
    df,
    column_definitions=col_defs,
    index_col="Pos",
    row_dividers=True,
    row_divider_kw={"linewidth" : 1, "linestyle" : (0, (1,5))},
    footer_divider=True,
    textprops={"fontsize" : 14},
    ax=ax,
).autoset_fontcolors(colnames=["xG", "xGA", "xGD"])

for idx in [0,1]:
    table.rows[idx].set_facecolor(row_colors["top2"])

for idx in range(2,6):
    table.rows[idx].set_facecolor(row_colors["top6"])

for idx in range(21,24):
    table.rows[idx].set_facecolor(row_colors["relegation"])