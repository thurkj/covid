#!/usr/bin/env python
# coding: utf-8

# In[26]:


# Load necessary packages
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import plotly.express as px        # high-level plotly module
import plotly.graph_objects as go  # lower-level plotly module with more functions
import datetime as dt              # for time and date
import json                        # For loading county FIPS data

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Modeled after 
# https://covid19-bayesian.fz-juelich.de/
# https://github.com/FZJ-JSC/jupyter-jsc-dashboards/blob/master/covid19/covid19dynstat-dash.ipynb


# # 1. Read Data

# In[27]:


minnesota_data = pd.read_csv('s3://mncovid19data/minnesota_data.csv',index_col=False)
minnesota_data_today = pd.read_csv('s3://mncovid19data/minnesota_data_today.csv',index_col=False)

# Load json file
with open('./Data/geojson-counties-fips.json') as response:  # Loads local file
    counties = json.load(response)    


# In[28]:


today = dt.datetime.now().strftime('%B %d, %Y')  # today's date. this will be useful when sourcing results 

# Set dates to datetime
minnesota_data['Date'] = pd.to_datetime(minnesota_data['Date'], format='%Y-%m-%d')


# # 2. Build Web Application
# 

# ## Define Application Structure
# 
# Set-up main html and call-back structure for the application.

# In[29]:


# Initialize Dash
#app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app.title = 'Covid-19 County Dashboard'
server = app.server  # Name Heroku will look for


# ## (Row 1, Col 1) Minnesota Maps (Snapshots):

# ### Map of Positivity Rates

# In[30]:


#===========================================
# County 14-day Case Rate Alongside MN Dept
# of Health School Recommendations
# (Choropleth Map of MN Counties)
#===========================================

# Load geojson county location information, organized by FIPS code
#with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
#    counties = json.load(response)
    
df = minnesota_data_today.dropna(subset=['infect'])

fig_infect_map = px.choropleth(df, geojson=counties, locations='FIPS', color='infect',
                           color_discrete_map={
                            'Less than 5%':'lightyellow',
                            'Between 5% and 15%':'yellow',
                            'Between 15% and 30%':'orange',
                            'Between 30% and 50%':'red',
                            'Between 50% and 70%':'darkred',
                            'Greater than 70%':'black'},
                           category_orders = {
                            'infect':['Less than 5%',
                            'Between 5% and 15%',
                            'Between 15% and 30%',
                            'Between 30% and 50%',
                            'Between 50% and 70%',
                            'Greater than 70%'
                            ]},
                           projection = "mercator",
                           labels={'infect':'Percent Infected:'},
                           hover_name = df['text'],
                           hover_data={'FIPS':False,'infect':False},
                          )

fig_infect_map.update_geos(fitbounds="locations", visible=False)
fig_infect_map.update_layout(legend=dict(
                        yanchor="top",
                        y=0.5,
                        xanchor="left",
                        x=0.6,
                        font_size=10
                      ),
                      height = 800,
                      margin={"r":0,"t":0,"l":0,"b":0},
                      dragmode=False
                      )


# ### Map of 14-day Case Rates

# In[31]:


#===========================================
# County 14-day Case Rate Alongside MN Dept
# of Health School Recommendations
# (Choropleth Map of MN Counties)
#===========================================
    
df = minnesota_data_today.dropna(subset=['schooling'])

fig_school_map = px.choropleth(df, geojson=counties, locations='FIPS', color='schooling',
                           color_discrete_map={
                            'Elem. & MS/HS in-person':'green',
                            'Elem. in-person, MS/HS hybrid':'tan',
                            'Elem. & MS/HS hybrid':'yellow',
                            'Elem. hybrid, MS/HS distance':'orange',
                            'Elem. & MS/HS distance':'red',
                            'Armageddon?':'black'},
                           category_orders = {
                            'schooling':['Elem. & MS/HS in-person',
                            'Elem. in-person, MS/HS hybrid',
                            'Elem. & MS/HS hybrid',
                            'Elem. hybrid, MS/HS distance',
                            'Elem. & MS/HS distance',
                            'Armageddon?'
                            ]},
                           projection = "mercator",
                           labels={'schooling':'Recommended Format:'},
                           hover_name = df['text'],
                           hover_data={'FIPS':False,'schooling':False},
                          )

fig_school_map.update_geos(fitbounds="locations", visible=False)
fig_school_map.update_layout(legend=dict(
                        yanchor="top",
                        y=0.5,
                        xanchor="left",
                        x=0.6,
                        font_size=10
                      ),
                      height = 800,
                      margin={"r":0,"t":0,"l":0,"b":0},
                      dragmode=False
                      )


# ## (Row 1, Col 2)  County Trends

# In[32]:


#===========================================
# County Infections (Line Graphs by County)
#===========================================

@app.callback(
    Output('county_infect_trend', 'figure'),
    [Input('county-dropdown2', 'value')])
    
# Update Figure
def update_county_figure(county_values):
                
    if county_values is None:
        dff = minnesota_data.pivot(index='Date',columns='Admin2',values='perc_infected')
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics

    else:
        if not isinstance(county_values, list): county_values = [county_values]
        temp = minnesota_data.loc[minnesota_data['Admin2'].isin(county_values)]
            
        dff = temp.pivot(index='Date',columns='Admin2',values='perc_infected')              
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics
        
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>County: ' + column + '<br>Date: ' + pd.to_datetime(dff.index).strftime('%Y-%m-%d') +'<br>Value: %{y:.1f}'
            )
        )

    # Update remaining layout properties
    fig.update_layout(
        height=800,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=12),
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            title="Percent of Residents Which Have Tested Positive",
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ="Source: Minnesota Department of Health. Author's calculations.")
                    ]
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig


#===========================================
# County 14-day Case Rate Trend (Line Graphs by County)
#===========================================

@app.callback(
    Output('county_trend', 'figure'),
    [Input('county-dropdown1', 'value')])
    
# Update Figure
def update_county_figure(county_values):
                
    if county_values is None:
        dff = minnesota_data.pivot(index='Date',columns='Admin2',values='ratio')
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics

    else:
        if not isinstance(county_values, list): county_values = [county_values]
        temp = minnesota_data.loc[minnesota_data['Admin2'].isin(county_values)]
            
        dff = temp.pivot(index='Date',columns='Admin2',values='ratio')              
        dff = dff[(dff != 0).all(1)]   # Remove early values not included in the statistics
        
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>County: ' + column + '<br>Date: ' + pd.to_datetime(dff.index).strftime('%Y-%m-%d') +'<br>Value: %{y:.1f}'
            )
        )

    # Update remaining layout properties
    fig.update_layout(
        height=800,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=12),
        xaxis=dict(
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            fixedrange = True
            ),
        yaxis=dict(
            title="14-day Case Count per 10k Residents",
            zeroline=True, 
            showgrid=False,  # Removes Y-axis grid lines
            fixedrange = True
            ),
        annotations=[  # Source annotation
                        dict(xref='paper',
                            yref='paper',
                            x=0.5, y=1.0,
                            showarrow=False,
                            text ="Source: Minnesota Department of Health. Author's calculations.")
                    ]
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    return fig


# ## Call-backs and Control Utilities

# In[33]:


# County Dropdown
county_dropdown1 =  html.P([
            dcc.Dropdown(
            id='county-dropdown1',
            options=[{'label': i, 'value': i} for i in minnesota_data['Admin2'].dropna().unique().tolist()],
            multi=True,
            searchable= True,
            value=['Hennepin','Carver'])
            ], style = {'width' : '90%',
                        'fontSize' : '20px',
                        'padding-right' : '0px'})

county_dropdown2 =  html.P([
            dcc.Dropdown(
            id='county-dropdown2',
            options=[{'label': i, 'value': i} for i in minnesota_data['Admin2'].dropna().unique().tolist()],
            multi=True,
            searchable= True,
            value=['Hennepin','Carver'])
            ], style = {'width' : '90%',
                        'fontSize' : '20px',
                        'padding-right' : '0px'})


# ## Define HTML

# In[34]:


#---------------------------------------------------------------------------
# DASH App formating
#---------------------------------------------------------------------------
header = html.H1(children="COVID-19 TRENDS (as of " + today + ")")

desc = dcc.Markdown(
f"""
#### The following graphs depict Covid-19 trends. The graphs are interactive; e.g., hover your cursor over a data-series to observe specific values.

-----
"""    
)

mn_head = html.H1(children="1. Covid-19 Prevalance Across Minnesota Counties")
mn_desc = dcc.Markdown(
            f"""
The following figures present county-level COVID-19 data. Source: Minnesota Dept of Health. 


The left panel presents current variation in COVID-19 across Minnesota counties.
Hover your cursor over a county to observe relevant characteristics. The 14-day case rate is defined as total new 
positive cases in the last 14 days per 10,000 residents. Data are organized according to school guidelines presented
by the state of Minnesota in July 2020.


The right panel presents the evolution of COVID-19 by county.
Select which counties to analyze using the pull-down menu or by entering in the county name.

In the second tab, I present estimates of the percent of each county's residents who have been infected.
This calculation begins with cumulative positive cases by location and time. I then adjust for un-reported
cases assuming that for every 1 positive case, there exist 7.1 unreported cases (i.e., I multiply case counts by 8.1).
Finally, I divide by total county population using 2019 estimates. Results are sensitive to the above report:unreport statistic. The 1:7.1 statistic I use 
is on the low-end of estimates and is sourced from a recent publication by CDC researchers using
data from February through September 2020 [article](https://academic.oup.com/cid/advance-article/doi/10.1093/cid/ciaa1780/6000389).
                #
            """   
)

state_head = html.H1(children="2. State COVID-19 Trends")
state_desc = dcc.Markdown(
            f"""
The following graphs compare COVID-19 statistics. 
Select midwestern states are included by default but you can modify the analysis by choosing a different subset of
states, periods, and/or standardize the statistics by population.
            """   
)
        
tab1_content = dbc.Card(
    dbc.CardBody(
        [
        dbc.Row([    
        dbc.Col([dcc.Graph(id="school_map", figure = fig_school_map)],width=12), 
        dbc.Col([county_dropdown1,dcc.Graph(id="county_trend")],width=12)
            ])
        ],
    ),
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
        dbc.Row([
        dbc.Col([dcc.Graph(id="infected_map", figure = fig_infect_map)],width=12), 
        dbc.Col([county_dropdown2,dcc.Graph(id="county_infect_trend")],width=12)
            ])
        ],
    ),
)
    
# App Layout
app.layout = dbc.Container(fluid=True, children=[    
    ## 
    dbc.Row(dbc.Col(width=12, children = [
        dbc.Tabs(
            [
                dbc.Tab(tab1_content, label="14-Day Case Rates"),
                dbc.Tab(tab2_content, label="Level of Infection"),
            ]
        )           
        ])
    ),        
])


# # 3. Run Application

# In[35]:


if __name__ == '__main__':
    #app.run_server(debug=True, use_reloader=False)  # Jupyter
    app.run_server(debug=False,host='0.0.0.0')    # Use this line prior to heroku deployment
    #application.run(debug=False, port=8080) # Use this line for AWS

