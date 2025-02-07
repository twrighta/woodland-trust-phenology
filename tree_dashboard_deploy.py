# Imports
from datetime import date
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import warnings
import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from flask_caching import Cache
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly_express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import hdbscan
import gunicorn

# Remove warning messages
warnings.simplefilter("ignore")

# Dataset path
# DF_PATH = "C:/Users/tomwr/Datascience/Datasets/Tabular/woodland_trust_phenology/tree_phenology.csv"
DF_PATH = 'https://raw.githubusercontent.com/twrighta/woodland-trust-phenology/main/tree_phenology.csv'
# Instantiate Dashapp
dash._dash_renderer._set_react_version('18.2.0')

# Instantiate Dash app and server
app = Dash(__name__,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.MINTY,
                                 dmc.styles.DATES])
server = app.server

# Create a flask Cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple'})  # simple in-memory cache


# Load dataset and convert observation date function in cache
@cache.memoize(timeout=120)
def load_df(path):
    """
    :param path: Path string of dataframe to load
    :return: Pandas Dataframe with an ObservationDate column with type datetime64[ns]
    """
    df = pd.read_csv(path,
                     parse_dates=["ObservationDate"])
    df[["Latitude", "Longitude"]] = df[["Latitude", "Longitude"]].astype("float32")
    return df


# Load df
df = load_df(DF_PATH)

# Create Lists used in filtering
SPECIES = sorted(list(df["Species"].unique()))
EVENTS = sorted(list(df["Event"].unique()))

# Dictionary of colours:
COLOURS = {"primary_green": "#78C2AD",
           "primary_purple": "#AD78C2",
           "primary_olive": "#C2AD78",
           "lighter_olive": "#d6ccb3"}

# Create a top section containing just the page heading
page_heading_section = html.Div(children=[
    dbc.Row([
        dbc.Col([
            html.H1(children=[
                "Woodland Trust Tree Phenology 2015-2024"
            ],
                style={"font-weight": "bold",
                       "padding": "5px",
                       "alignText": "center"})
        ],
            width=12)

    ],
        style={"height": "5vh",
               "backgroundColor": COLOURS["primary_green"]})
])

# Create a second top section, containing the 2 dropdown menus and a slider
page_filters_section = html.Div(children=[
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(options=SPECIES,
                         value=SPECIES[0],
                         id="species-dropdown",
                         placeholder="Select a species",
                         style={"border-radius": "10px",
                                "padding": "5px"})
        ],
            width=4),
        dbc.Col([
            dcc.Dropdown(options=EVENTS,
                         value=pd.Series(EVENTS).dropna().tolist()[0],
                         id="event-options",
                         placeholder="Select an event",
                         style={"border-radius": "10px",
                                "padding": "5px"})
        ],
            width=4),
        dbc.Col([
            html.Div([

                dcc.DatePickerRange(
                    id="date-picker-input",
                    start_date=date(2015, 1, 1),
                    end_date=date(2024, 12, 31),
                    minimum_nights=1,
                    clearable=True,
                    min_date_allowed=date(2015, 1, 2),
                    style={"width": "400",
                           "border-radius": "10px",
                           "padding": "5px"}
                ),
                dbc.Button("Scale observations by year",
                           id="scale-btn",
                           color="info"),

                dbc.Button("Aggregate by Week/Month",
                           id="agg-btn",
                           color="info")

            ],
                style={"display": "flex",
                       "padding": "5px"})
        ],
            width=4)

    ],
        style={"height": "5vh",
               "backgroundColor": COLOURS["primary_olive"]})
])

# Content section - 2 equally sized charts, then rightmost 2 stacked charts.
content_section = html.Div([
    dbc.Row([
        html.Div(id="start-date", style={"display": "none"}),  # Hidden div for start date
        html.Div(id="end-date", style={"display": "none"}),  # Hidden div for end date
        dbc.Col([
            html.Div([
                dcc.Loading(dcc.Graph(id="map-fig",
                                      style={"height": "70vh"}),
                            type="cube")
            ])

        ],
            width=4),
        dbc.Col([
            html.Div([
                html.Div([
                    dbc.Col([
                        dcc.RangeSlider(id="year-slider",
                                        min=2015,
                                        max=2024,
                                        step=1,
                                        value=[2015, 2024],
                                        marks={i: str(i) for i in range(2015, 2024, 1)})  #,
                    ]),
                    dbc.Col([
                        html.Div([
                            dbc.Button("Mann-Whitney Difference Check",
                                       id="mann-whitney-btn",
                                       color="info",
                                       style={"padding": "5px"}),
                            # This is to be revealed when the user clicks the mann-whitney-btn
                            dbc.Toast(id="mann-whitney-toast",
                                      children=[html.P(id="mann-whitney-text")],
                                      header="Mann Whitney Statistical Difference Test",
                                      icon="primary",
                                      duration=5000,
                                      is_open=False)
                        ])
                    ])
                ],
                    style={"display": "flex"}),
                dcc.Loading(dcc.Graph(id="month-line-fig",
                                      style={"height": "70vh"}),
                            type="cube")

            ])
        ],
            width=4),
        # 3rd column: HDBSCAN plot
        dbc.Col([
            dbc.Row([
                dcc.Loading(dcc.Graph(id="hdbscan-fig"),
                            type="cube")
            ]),
            dbc.Row([
                dcc.Loading(dcc.Graph(id="line-fig"),
                            type="cube")
            ])
        ],
            width=4)

    ])
],
    # Whole Div style
    style={"backgroundColor": COLOURS["lighter_olive"]})

# Combine all Sections together into a layout
app.layout = dmc.MantineProvider(
    dbc.Container([
        dbc.Row([
            dbc.Col(page_heading_section, width=12)
        ],
            style={"height": "5vh"}),
        dbc.Row([
            dbc.Col(page_filters_section, width=12)
        ],
            style={"height": "5vh"}),
        dbc.Row([
            dbc.Col(content_section, width=12)
        ],
            style={"height": "80vh"}),  # orig 90
        # Bottom Rows of just colours matching top headers and page filters
        dbc.Row([
            dbc.Col(children=None, width=12)
        ],
            style={"height": "5vh",
                   "backgroundColor": COLOURS["primary_olive"]}),
        dbc.Row([
            dbc.Col(children=None, width=12)
        ],
            style={"height": "5vh",
                   "backgroundColor": COLOURS["primary_green"]}
        )

    ],
        style={"height": "100vh",
               "backgroundColor": COLOURS["lighter_olive"]},
        fluid=True),
    id="mantine-provider"
)


# Helper functions
# Use for calculating period in month
def month_period(day):
    if day <= 10:
        return 1
    elif 11 >= day <= 20:
        return 2
    else:
        return 3


# Function to return a list of species which have a given selected event from the full df
def return_species_for_event(event):
    species_list = list(df[df["Event"] == event]["Species"].unique())
    return species_list


# Function to calculate scaled observation count for each record in dataframe, based on the year
def calc_scaled_obs_count(df):
    scaling_dict = {2015: 1,
                    2016: 30782 / 22715,
                    2017: 30782 / 18370,
                    2018: 30782 / 13832,
                    2019: 30782 / 13776,
                    2020: 30782 / 7545,
                    2021: 30782 / 2890,
                    2022: 30782 / 2880,
                    2023: 30782 / 2596,
                    2024: 30782 / 2226}
    df["ScaleFactor"] = df["Year"].map(scaling_dict)
    df["NormalizedObs"] = df["ObservationCount"] * df["ScaleFactor"]
    df.drop(columns=["ScaleFactor"],
            inplace=True)
    return df


# Callback Functions #
# Dynamically change options in Events dropdown based on Species
@cache.memoize(timeout=30)
@app.callback(
    Output("event-options", "options"),
    Input("species-dropdown", "value"))
def dynamic_dropdown(species_dropdown):
    """
    :param species_dropdown: Value selected by user in the species dropdown
    :return: options: Alphabetically sorted list of Events that occurred in a given species = Id for event dropdown
    """
    options = sorted(list(df[df["Species"] == species_dropdown]["Event"].unique()))
    return options


# Function to use when an empty plot is generated
@cache.memoize(timeout=300)
def return_empty_plot():
    plot = px.scatter(title="<b>No data available for the selected filters</b>",
                      template="seaborn")
    plot.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                       paper_bgcolor='rgba(0, 0, 0, 0)',
                       showlegend=False)
    plot.update_yaxes(griddash="dot",
                      gridcolor="#D3D3D3")
    plot.update_xaxes(griddash="dot",
                      gridcolor="#D3D3D3")
    return plot


# Filter dataframe to the dropdown options selected by user. Used at start of user functions.
@cache.memoize(timeout=128)
def filter_df(orig_df, species_dropdown, event_options, start_date, end_date):
    """
    :param orig_df: Original dataframe to filter down
    :param species_dropdown: Species selected in the species dropdown menu
    :param event_options: Event selected in the event dropdown menu, based on Species chosen.
    :param start_date: Datepicker start date selected
    :param end_date: Datepicker end date
    :return: A pandas dataframe filtered to the user-specified filters.
    """

    filtered_df = orig_df[(orig_df["Species"] == species_dropdown) &
                          (orig_df["Event"] == event_options) &
                          (orig_df["ObservationDate"].between(pd.to_datetime(start_date), pd.to_datetime(end_date)))]

    return filtered_df


# Filter dataframe to the dropdown options selected by user, EXCLUDING TIME
@cache.memoize(timeout=128)
def filter_df_EXCL_TIME(orig_df, species_dropdown, event_options):
    """
    :param orig_df: Original dataframe to filter down
    :param species_dropdown: Species selected in the species dropdown menu
    :param event_options: Event selected in the event dropdown menu, based on Species chosen.
    :return: A pandas dataframe filtered to the user-specified filters.
    """

    filtered_df = orig_df[(orig_df["Species"] == species_dropdown) &
                          (orig_df["Event"] == event_options)]
    return filtered_df


# Plot 1: Plot a map with a slider option to move through time
@cache.memoize(timeout=120)
@app.callback(
    Output("map-fig", "figure"),
    [Input("species-dropdown", "value"),
     Input("event-options", "value"),
     Input("date-picker-input", "start_date"),
     Input("date-picker-input", "end_date")]
)
def plot_time_map(species_dropdown, event_options, start_date, end_date):
    # Try immediately break out if nothing valid passed
    if event_options is None or event_options == '':
        return_empty_plot()

    filtered_df = filter_df(df, species_dropdown, event_options, start_date, end_date)[["ObservationDate",
                                                                                        "Latitude",
                                                                                        "Longitude"]]

    filtered_df["Month"] = filtered_df["ObservationDate"].dt.month.astype("str")  # Used for point colouring.

    # Create a boolean mask for months not in ["10", "11", "12"]
    non_special_months = ~filtered_df["Month"].isin(["10", "11", "12"])

    # Year month creation by conditional assignment
    filtered_df["Year_Month"] = np.where(
        non_special_months,
        (filtered_df["ObservationDate"].dt.year.astype(str) + "0" + filtered_df["Month"]).astype(int),
        (filtered_df["ObservationDate"].dt.year.astype(str) + filtered_df["Month"]).astype(int)
    )

    len_filtered_df = len(filtered_df)

    # Return a blank map if no data for these filters
    if len_filtered_df == 0:
        return return_empty_plot()

    # Only attempt to plot if there is data in this selection
    elif len_filtered_df > 0:

        all_year_months = [201501, 201502, 201503, 201504, 201505, 201506, 201507, 201508, 201509,
                           201510, 201511, 201512,
                           201601, 201602, 201603, 201604, 201605, 201606, 201607, 201608, 201609,
                           201610, 201611, 201612,
                           201701, 201720, 201703, 201704, 201705, 201706, 201707, 201708, 201709,
                           201710, 201711, 201712,
                           201801, 201802, 201803, 201804, 201805, 201806, 201807, 201808, 201809,
                           201810, 201811, 201812,
                           201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909,
                           201910, 201911, 201912,
                           202001, 202002, 202003, 202004, 202005, 202006, 202007, 202008, 202009,
                           202010, 202011, 202012,
                           202101, 202102, 202103, 202104, 202105, 202106, 202107, 202108, 202109,
                           202110, 202111, 202112,
                           202201, 202202, 202203, 202204, 202205, 202206, 202207, 202208, 202209,
                           202210, 202211, 202212,
                           202301, 202302, 202303, 202304, 202305, 202306, 202307, 202308, 202309,
                           202310, 202311, 202312,
                           202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409,
                           202410, 202411, 202412]

        all_months = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

        num_add_rows = len(all_year_months)

        add_rows = pd.DataFrame({"ObservationDate": [np.nan] * num_add_rows,
                                 "Latitude": [np.nan] * num_add_rows,
                                 "Longitude": [np.nan] * num_add_rows,
                                 "Year_Month": all_year_months,
                                 "Month": all_months})

        full_filtered_df = pd.concat([filtered_df, add_rows], ignore_index=True)

        # Fill missing coordinates and observation date with dummy data just so it thinks there is data there.
        full_filtered_df[["Latitude", "Longitude"]] = full_filtered_df[["Latitude", "Longitude"]].fillna(50)

        # Reorder so its in correct order for timeframe in map animation
        full_filtered_df.sort_values(by="Year_Month", ascending=True, inplace=True)

        # Plot the map
        map_plot = px.scatter_mapbox(full_filtered_df,
                                     lat="Latitude",
                                     lon="Longitude",
                                     mapbox_style="carto-positron",
                                     center={"lat": 53.3,
                                             "lon": -1.4},
                                     zoom=4,
                                     title=f"<b>{species_dropdown} - {event_options}: Monthly Observations {str(start_date)} and {str(end_date)}",
                                     animation_frame="Year_Month",
                                     animation_group="Year_Month",
                                     color="Month",
                                     color_discrete_map={
                                         "January": "#001f4d",
                                         "February": "#003c99",
                                         "March": "#007fff",
                                         "April": "#0055ff",
                                         "May": "#ff7f7f",
                                         "June": "#ff0000",
                                         "July": "#ff4000",
                                         "August": "#ff8000",
                                         "September": "#ffbf40",
                                         "October": "#ff00ff",
                                         "November": "#800080",
                                         "December": "#4b0082"
                                     },
                                     hover_name="Year_Month",
                                     hover_data={"Year_Month": False,
                                                 "Month": True,
                                                 "Latitude": False,
                                                 "Longitude": False})
        # Don't show legend
        map_plot.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                               paper_bgcolor='rgba(0, 0, 0, 0)',
                               showlegend=False)

        return map_plot


# New version of month line plot
@cache.memoize(timeout=128)
@app.callback(
    [Output("month-line-fig", "figure"),
     Output("mann-whitney-toast", "is_open"),
     Output("mann-whitney-text", "children")],
    [Input("species-dropdown", "value"),
     Input("event-options", "value"),
     Input("year-slider", "value"),
     Input("mann-whitney-btn", "n_clicks"),
     Input("scale-btn", "n_clicks"),
     Input("agg-btn", "n_clicks")],
    prevent_initial_call=True
)
def plot_month_line(species_dropdown, event_options, year_slider, mann_whitney, scale_btn, agg_btn):
    """
    :param species_dropdown: species selected
    :param event_options: event selected
    :param year_slider: year values selected
    :param mann_whitney: mann-whitney U button
    :param scale_btn: scaling button clicks
    :return:
    """
    # Cancel plot if no details provided
    if event_options is None or species_dropdown is None or year_slider is None:
        return return_empty_plot(), False, ""

    # For display and filtering
    MONTHS_NOS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Use this to order the df later
    WEEK_NOS = [num for num in range(1, 53, 1)]
    start_year = year_slider[0]
    end_year = year_slider[1]

    # Create Year, Week, and Month column to filter data
    df["Year"] = df["ObservationDate"].dt.year
    df["Week"] = df["ObservationDate"].dt.isocalendar().week
    df["Month"] = df["ObservationDate"].dt.month

    # Set observation count to 1 for all existing records for group summing
    df["ObservationCount"] = 1

    # Filter dataframe to the years selected
    filtered_df = filter_df_EXCL_TIME(df[(df["Year"] >= int(start_year)) &
                                         (df["Year"] <= int(end_year))], species_dropdown, event_options)

    # Display empty ploty if no data in that interval
    if filtered_df.empty:
        return return_empty_plot(), False, ""

    # Create different missing append_dfs depending on aggregation type selected.
    if agg_btn is None:  # Force it to be a number so doesn't break next comparison
        agg_btn = 0

    if int(agg_btn) % 2 == 0:
        agg_type = "Week"
        append_df = pd.DataFrame({"Year": [year for year in range(2015, 2025) for _ in range(52)],
                                  agg_type: WEEK_NOS * 10,
                                  "ObservationCount": [0] * 520})
        filtered_df.drop(columns=["Latitude", "Longitude", "Species", "Event", "ObservationDate", "Month"],
                         inplace=True)

    else:
        agg_type = "Month"
        append_df = pd.DataFrame({"Year": [year for year in range(2015, 2025) for _ in range(12)],
                                  "Month": MONTHS_NOS * 10,
                                  "ObservationCount": [0] * 120})
        filtered_df.drop(columns=["Latitude", "Longitude", "Species", "Event", "ObservationDate", "Week"],
                         inplace=True)

    # Concatenate append_df and filtered_df
    grouped_filtered_df = pd.concat([filtered_df, append_df],
                                    ignore_index=True).groupby(["Year", agg_type],
                                                               as_index=False, observed=True).sum().sort_values(
        by=["Year", agg_type])

    # Map Month names to numbers
    if agg_type == "Month":
        month_map = {1: "January",
                     2: "February",
                     3: "March",
                     4: "April",
                     5: "May",
                     6: "June",
                     7: "July",
                     8: "August",
                     9: "September",
                     10: "October",
                     11: "November",
                     12: "December"}
        grouped_filtered_df[agg_type] = grouped_filtered_df[agg_type].map(month_map)

    # reprocess df, adding a normalized observation count column
    grouped_filtered_df = calc_scaled_obs_count(grouped_filtered_df)

    # Now bool var for whether user has clicked scaling button. Use for sections below
    if scale_btn is None:  # Force it to be a number so doesn't break next comparison
        scale_btn = 0

    if int(scale_btn) % 2 == 0:
        obs_type_col = "ObservationCount"
        norm = ""  # Used for in plot title.
    else:
        obs_type_col = "NormalizedObs"
        norm = "Normalized"  # Used in plot title

    # If the user clicks the mann whitney button, requesting a significant difference check:
    if mann_whitney is not None and mann_whitney != 0:
        start_year_count = np.array(list(grouped_filtered_df[grouped_filtered_df["Year"] == start_year].groupby(
            agg_type, as_index=False, observed=True).sum()[obs_type_col]))

        start_year_count[start_year_count == np.nan] = 0  # Replace np.nans with 0

        end_year_count = np.array(list(grouped_filtered_df[grouped_filtered_df["Year"] == end_year].groupby(
            agg_type, as_index=False, observed=True).sum()[obs_type_col]))
        end_year_count[end_year_count == np.nan] = 0  # Replace np.nans with 0

        # Two-sided mann whitney U test.
        ustat, pvalue = mannwhitneyu(start_year_count, end_year_count, alternative="two-sided")

        if pvalue < 0.05:
            mann_whitney_text = (
                f"There is a statistically significant difference in {species_dropdown} {event_options} "
                f"observations between {str(start_year)} and {str(end_year)}. "
                f"We can"
                f" be {abs(round(1 - (pvalue * 100), 2))}% confident there is a difference not due to chance")

        else:
            mann_whitney_text = (
                f"There is no statistically significant difference in {species_dropdown} {event_options} observations "
                f"between {str(start_year)} and {str(end_year)}. We are only {abs(round(1 - (pvalue * 100), 2))}% confident their "
                f" differences"
                f" are not due to random variability")

        line_plot = px.line(data_frame=grouped_filtered_df,
                            x=agg_type,
                            y=obs_type_col,
                            color="Year",
                            title=f"<b>{species_dropdown} - {event_options}: Monthly {norm} Observations {start_year} - {end_year}</b>",
                            template="seaborn",
                            hover_name=agg_type,
                            hover_data={agg_type: False,
                                        obs_type_col: True,
                                        "Year": True})
        # Organize months if month chosen
        if agg_type == "Month":
            line_plot.update_xaxes(categoryorder="array",
                                   categoryarray=['January', 'February', 'March', 'April', 'May', 'June', 'July',
                                                  'August',
                                                  'September', 'October', 'November', 'December'],
                                   griddash="dot",
                                   gridcolor="#D3D3D3")
        else:
            line_plot.update_xaxes(griddash="dot",
                                   gridcolor="#D3D3D3")
        # Update rest of plot
        line_plot.update_yaxes(griddash="dot",
                               gridcolor="#D3D3D3")

        line_plot.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                                paper_bgcolor='rgba(0, 0, 0, 0)')

        for line_trace in line_plot.data:
            if line_trace.name in [str(start_year), str(end_year)]:
                line_trace.opacity = 1
                line_trace.line.dash = "solid"
            else:
                line_trace["opacity"] = 0.425
                line_trace.line.dash = "dashdot"

        return line_plot, True, mann_whitney_text

    # Users who DID NOT click mann whitney button
    else:
        line_plot = px.line(data_frame=grouped_filtered_df,
                            x=agg_type,
                            y=obs_type_col,
                            color="Year",
                            title=f"<b>{species_dropdown} - {event_options}: Monthly {norm} Observations {start_year} - {end_year}</b>",
                            template="seaborn",
                            hover_name=agg_type,
                            hover_data={agg_type: False,
                                        obs_type_col: True,
                                        "Year": True})
        if agg_type == "Month":
            line_plot.update_xaxes(categoryorder="array",
                                   categoryarray=['January', 'February', 'March', 'April', 'May', 'June', 'July',
                                                  'August',
                                                  'September', 'October', 'November', 'December'],
                                   griddash="dot",
                                   gridcolor="#D3D3D3")
        else:
            line_plot.update_xaxes(griddash="dot",
                                   gridcolor="#D3D3D3")
        # Update rest of plot normally.
        line_plot.update_yaxes(griddash="dot",
                               gridcolor="#D3D3D3")

        line_plot.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                                paper_bgcolor='rgba(0, 0, 0, 0)')

        for line_trace in line_plot.data:
            if line_trace.name in [str(start_year), str(end_year)]:
                line_trace.opacity = 1
                line_trace.line.dash = "solid"
            else:
                line_trace["opacity"] = 0.425
                line_trace.line.dash = "dashdot"

        return line_plot, False, None


# HDBSCAN plot
@cache.memoize(timeout=128)
@app.callback(
    Output("hdbscan-fig", "figure"),
    [Input("species-dropdown", "value"),
     Input("event-options", "value"),
     Input("date-picker-input", "start_date"),
     Input("date-picker-input", "end_date")],
    prevent_initial_call=True)
def plot_hdbscan(species_dropdown, event_options, start_date, end_date):
    if event_options is None or event_options == '':
        return return_empty_plot()

    filtered_df = filter_df(df[["Species", "Event", "ObservationDate", "Longitude", "Latitude"]], species_dropdown,
                            event_options, start_date,
                            end_date)

    if filtered_df.empty:
        return return_empty_plot()

    # Replace ObservationDate with 3 columns for day, month, year
    filtered_df["Year"] = filtered_df["ObservationDate"].dt.year
    filtered_df["month"] = filtered_df["ObservationDate"].dt.month
    filtered_df["month_period"] = filtered_df["month"].apply(month_period)

    min_year, max_year = filtered_df["Year"].min(), filtered_df["Year"].max()

    # Remove unneeded columns now:
    filtered_df.drop(columns=["ObservationDate", "Species", "Event"],
                     inplace=True)

    filtered_df_cols = list(filtered_df.columns)  # For use later once been converted to numpy array.

    # Ensure there is data filled.
    if not filtered_df.empty:
        # Instantiate HDBSCAN
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=20,
                                            cluster_selection_method="eom",  # 'Excess of mass'
                                            allow_single_cluster=False,
                                            algorithm="best",
                                            leaf_size=10,
                                            approx_min_span_tree=True,
                                            core_dist_n_jobs=-1)
        hdbscan_clusterer.fit(filtered_df)
        labels = hdbscan_clusterer.labels_  # These will just be numbers, one for each cluster created.

        # Generate X and Y components using TSNE for plotting - requires scaling
        ss = StandardScaler()
        filtered_df = ss.fit_transform(filtered_df)

        # Convert dataset into n_components
        tsne_components = TSNE(n_components=2,
                               perplexity=min(30, len(filtered_df) - 1),
                               max_iter=500,
                               learning_rate="auto",
                               n_jobs=-1).fit_transform(filtered_df)

        # Convert filtered_df back into df from being a numpy array (from scaling)
        filtered_df = pd.DataFrame(filtered_df, columns=filtered_df_cols)

        # Add the new calculated columns
        filtered_df["x_component"] = tsne_components[:, 0]
        filtered_df["y_component"] = tsne_components[:, 1]
        filtered_df["labels"] = labels

        # Create a map of labels to replace. .map() is quicker than .replace()
        cluster_map = {-1: "Noise",
                       0: "Cluster 1",
                       1: "Cluster 2",
                       2: "Cluster 3",
                       3: "Cluster 4",
                       4: "Cluster 5",
                       5: "Cluster 6",
                       6: "Cluster 7",
                       8: "Cluster 9"}
        filtered_df["labels"] = filtered_df["labels"].map(cluster_map)

        # Create a scatter plot of clustering
        scatter = px.scatter(filtered_df,
                             x="x_component",
                             y="y_component",
                             color="labels",
                             title=f"<b>{species_dropdown} - {event_options}: TSNE & HDBSCAN Clustering {str(min_year)} - {str(max_year)}</b>",
                             template="seaborn",
                             hover_name="labels",
                             hover_data={"x_component": False,
                                         "y_component": False,
                                         "labels": False})

        scatter.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                              paper_bgcolor='rgba(0, 0, 0, 0)',
                              xaxis_title="Component 1",
                              yaxis_title="Component 2",
                              legend_title_text="Label")
        scatter.update_xaxes(griddash="dot",
                             gridcolor="#D3D3D3")
        scatter.update_yaxes(griddash="dot",
                             gridcolor="#D3D3D3")

        return scatter
    else:
        return return_empty_plot()


# Plot 4: Line plot to go underneath clustering - observation count by month for whole filtered period.
# scale it by the total number of observations in the year (try to counter for fewer users using it..)
# Include other species with the same event in a lighter opacity (and colours) on the chart.
@cache.memoize(timeout=128)
@app.callback(
    Output("line-fig", "figure"),
    [Input("species-dropdown", "value"),
     Input("event-options", "value"),
     Input("date-picker-input", "start_date"),
     Input("date-picker-input", "end_date"),
     Input("scale-btn", "n_clicks"),
     Input("agg-btn", "n_clicks")],
    prevent_initial_call=True
)
def plot_species_line(species_dropdown, event_options, start_date, end_date, scale_btn, agg_btn):
    # Initial conditional otherwise plot no data
    if not (species_dropdown and event_options and start_date and end_date):
        return return_empty_plot()

    if agg_btn is None:
        agg_btn = 0  # So this doesnt break

    # List of other species which also have data on the selected event with that event selected (in full df)
    species_list = return_species_for_event(event_options)

    species_df = df.loc[
        df["Species"].isin(species_list) &
        (df["Event"] == event_options) &
        df["ObservationDate"].between(pd.to_datetime(start_date), pd.to_datetime(end_date)),
        ["Species", "ObservationDate"]
    ].copy()

    species_df["Month"] = species_df["ObservationDate"].dt.month.astype(int)
    species_df["Year"] = species_df["ObservationDate"].dt.year
    species_df["Week"] = species_df["ObservationDate"].dt.isocalendar().week
    # Set observation count equal to 1 for now.
    species_df["ObservationCount"] = 1

    # Apply scaling function
    species_df = calc_scaled_obs_count(species_df)

    # Only try to plot if there is valid data.
    if not species_df.empty:

        # Decide on type of aggregation based on agg_btn
        if agg_btn % 2 == 0:
            agg_col_type = "Week"
            agg_col_type_title = "Weekly"
        else:
            agg_col_type = "Month"
            agg_col_type_title = "Monthly"

            # Add a column for selection - if it's the selected species give a different value to others (for colouring)
        species_df["user_selection"] = np.where(species_df["Species"] == species_dropdown, "Selected",
                                                "Other Species")

        # Make all grouping columns categorical for faster grouping
        species_df[agg_col_type] = species_df[agg_col_type].astype("category")
        species_df["user_selection"] = species_df["user_selection"].astype("category")
        species_df["Species"] = species_df["Species"].astype("category")

        # Aggregate data by week (Agg_col_type
        grouped = species_df[
            ["Species", agg_col_type, "user_selection", "ObservationCount", "NormalizedObs"]].groupby(
            ["Species", agg_col_type, "user_selection"], as_index=False, observed=True).sum()

        # Replace Month values with full month names if month is chosen
        if agg_col_type == "Month":
            month_map = {1: "January",
                         2: "February",
                         3: "March",
                         4: "April",
                         5: "May",
                         6: "June",
                         7: "July",
                         8: "August",
                         9: "September",
                         10: "October",
                         11: "November",
                         12: "December"}
            grouped[agg_col_type] = grouped[agg_col_type].map(month_map)

        # Now bool var for whether user has clicked scaling button. Use for sections below
        if scale_btn is None:  # Force it to be a number so doesn't break next comparison
            scale_btn = 0

        if int(scale_btn) % 2 == 0:
            obs_type_col = "ObservationCount"
            norm = ""
        else:
            obs_type_col = "NormalizedObs"
            norm = "Normalized"

        # Create line plot. Using hover_data to control what is (True) and is not (False) shown when hovering
        line_plot = px.line(data_frame=grouped,
                            x=agg_col_type,
                            y=obs_type_col,
                            color="Species",
                            title=f"<b>{species_dropdown} - {event_options}: Total {norm} {agg_col_type_title} Occurrence by Species</b>",
                            template="seaborn",
                            line_dash="user_selection",
                            hover_name="Species",
                            hover_data={"Species": False,
                                        "user_selection": False,
                                        agg_col_type: True,
                                        obs_type_col: True})

        # Set the opacity of the lines depending on if theyre the selected species or not.
        for trace in line_plot.data:

            # Set a colour for the selected species consistently:
            if "Selected" in str(trace):
                trace.line.color = 'rgb(120,194,173)'
                trace.opacity = 1
                trace.line.dash = "solid"
            else:
                trace["opacity"] = 0.425
                trace.line.dash = "dashdot"

        # Hide legend - have to hover to see the different species. Reduce clutter, just show there is difference
        line_plot.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                                paper_bgcolor='rgba(0, 0, 0, 0)',
                                showlegend=False)
        line_plot.update_xaxes(griddash="dot",
                               gridcolor="#D3D3D3")
        line_plot.update_yaxes(griddash="dot",
                               gridcolor="#D3D3D3")

        return line_plot

    # No data in the option combination selected by user.
    else:
        return return_empty_plot()

# Run app
if __name__ == "__main__":
    app.run(debug=True)
