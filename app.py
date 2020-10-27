#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load necessary packages
import pandas as pd
import plotly.express as px        # high-level plotly module
import plotly.graph_objects as go  # lower-level plotly module with more functions
import pandas_datareader as pdr    # we are grabbing the data and wb functions from the package
import datetime as dt              # for time and date
import requests                    # api module
#from ipywidgets import widgets     # interactive graphs

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


# In[2]:


today = dt.datetime.now().strftime('%B %d, %Y')  # today's date. this will be useful when sourcing results 


# In[3]:


url = 'https://api.covidtracking.com/v1/states/daily.json'
    
response = requests.get(url)

if response.status_code == 200: 
    print('Download successful')
    covid = pd.read_json(response.text)  # convert to dataframe
elif response.status_code == 301: 
    print('The server redirected to a different endpoint.')
elif response.status_code == 401: 
    print('Bad request.')
elif response.status_code == 401: 
    print('Authentication required.')
elif response.status_code == 403: 
    print('Access denied.')
elif response.status_code == 404: 
    print('Resource not found')
else: 
    print('Server busy')


# In[4]:


covid.tail()


# In[5]:


covid.dtypes


# In[6]:


covid['pos_rate'] = 100*(covid['positiveTestsViral']+covid['positiveTestsAntigen'])/(covid['totalTestsPeopleViral']+ covid['totalTestsAntigen'])
covid['date'] = pd.to_datetime(covid['date'], format='%Y%m%d')
covid['month'] = covid['date'].dt.strftime('%B %Y') 
covid.head()


# In[7]:


covid.drop(covid[covid['state'] == "AS"].index, inplace=True)         # Drop American Samoa
covid.drop(covid[covid['date'] < '2020-03-01'].index, inplace=True)   # Drop January and February when there are few cases

months = covid['month'].unique().tolist()
months.reverse()

# Look only at Texas, Florida, Georgia, and Minnesota
#covid.drop(covid[(covid['state'] != "TX") & (covid['state'] != "FL") & (covid['state'] != "MN") & (covid['state'] != "GA")].index, inplace=True)

df_positive = covid.pivot(index='date',columns='state',values='positive')

df_curhospital = covid.pivot(index='date',columns='state',values='hospitalizedCurrently')

df_newdeaths = covid.pivot(index='date',columns='state',values='deathIncrease')

df_totdeaths = covid.pivot(index='date',columns='state',values='deathConfirmed')


# In[8]:


#fig_positive = multi_plot(df_positive, title="COVID-19 Confirmed Cases by State<br> (as of " + today + ")")  
#fig_curhospital = multi_plot(df_curhospital, title="COVID-19 Hospitalizations by State<br> (as of " + today + ")")  
#fig_newdeaths = multi_plot(df_newdeaths, title="COVID-19 Deaths by State<br> (as of " + today + ")")    
#fig_totdeaths = multi_plot(df_totdeaths, title="COVID-19 Deaths by State<br> (as of " + today + ")")  

#fig_positive0 = multi_plot0(df_positive)  


# In[9]:


import plotly.io as pio
#pio.write_html(fig_positive, file='covid_positive.html', auto_open=True, config={"displayModeBar": False, "showTips": False, "responsive": True})
#pio.write_html(fig_curhospital, file='covid_hospital.html', auto_open=True, config={"displayModeBar": False, "showTips": False, "responsive": True})
#pio.write_html(fig_newdeaths, file='covid_deaths.html', auto_open=True, config={"displayModeBar": False, "showTips": False, "responsive": True})
#pio.write_html(fig_totdeaths, file='covid_totdeaths.html', auto_open=True, config={"displayModeBar": False, "showTips": False, "responsive": True})


# In[10]:


# Prepare for Dash
df = covid[['date','state','positive','hospitalizedCurrently','deathIncrease','death']]
df2 = df.copy()
df2.state = 'All States'
df = df.append(df2, ignore_index=True)


# In[11]:


app = dash.Dash()

header = html.H1(children="United States Covid-19 Trends (as of " + today + ")")

markdown_text = '''
### Dash and Markdown
The following graphs depict select Covid-19 trends by state. 
Use the filters to choose a subset of states and/or dates to sharpen the analysis. 
Source: The Atlantic [Covid-19 Tracking Project](https://covidtracking.com)
'''
markdown = dcc.Markdown(children=markdown_text)

# Dropdown
dropdown =  html.P([
            html.Label("Select States"),
            dcc.Dropdown(
            id='state-dropdown',
            options=[{'label': i, 'value': i} for i in df['state'].unique().tolist()],
            multi=True,
            searchable= True)
            ], style = {'width' : '80%',
                        'fontSize' : '20px',
                        'padding-left' : '100px',
                        'display': 'inline-block'})
    
# range slider
slider =    html.P([
            html.Label("Select Months"),
            dcc.RangeSlider(id = 'slider',
                        marks = {i : months[i] for i in range(0, len(months))},
                        min = 0,
                        max = len(months)-1,
                        value = [0, len(months)-1])
            ], style = {'width' : '80%',
                        'fontSize' : '20px',
                        'padding-left' : '100px',
                        'display': 'inline-block'})
    
graph1 = dcc.Graph(id="positive", style={'display': 'inline-block'})
graph2 = dcc.Graph(id="curhospital", style={'display': 'inline-block'})
graph3 = dcc.Graph(id="newdeaths", style={'display': 'inline-block'})
graph4 = dcc.Graph(id="totdeaths", style={'display': 'inline-block'})

row1 = html.Div(children=[graph1, graph2])
row2 = html.Div(children=[graph3, graph4])

layout = html.Div(children=[header, markdown, dropdown, slider, row1, row2], style={"text-align": "center"})
app.layout = layout


# In[12]:


@app.callback(
    Output('positive', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# STEP 5: Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = df.copy()
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='positive')        
    elif state_values[0]=="All States":
        dff = df.copy()
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='positive')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        dff = temp.pivot(index='date',columns='state',values='positive')
    
    # Filter by months
    dff = dff[(dff.index >= dt.datetime.strptime(months[month_values[0]],"%B %Y")) & (dff.index <= dt.datetime.strptime(months[month_values[1]],"%B %Y"))]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:,}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        title={
                'text': "New Cases",
                'x':0.5,'xanchor': 'center',
                'font':{'size': 18}
                },
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        width=700,   
        xaxis=dict(
            title="Date",
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False  # Removes Y-axis grid lines
            )
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# In[13]:


@app.callback(
    Output('curhospital', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# STEP 5: Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = df.copy()
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='hospitalizedCurrently')        
    elif state_values[0]=="All States":
        dff = df.copy()
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='hospitalizedCurrently')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        dff = temp.pivot(index='date',columns='state',values='hospitalizedCurrently')
    
    # Filter by months
    dff = dff[(dff.index >= dt.datetime.strptime(months[month_values[0]],"%B %Y")) & (dff.index <= dt.datetime.strptime(months[month_values[1]],"%B %Y"))]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:,}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        title={
                'text': "Current Hospitalizations",
                'x':0.5,'xanchor': 'center',
                'font':{'size': 18}
                },
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        width=700,  
        xaxis=dict(
            title="Date",
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False  # Removes Y-axis grid lines
            )
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# In[14]:


@app.callback(
    Output('newdeaths', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# STEP 5: Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = df.copy()
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='deathIncrease')        
    elif state_values[0]=="All States":
        dff = df.copy()
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='deathIncrease')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        dff = temp.pivot(index='date',columns='state',values='deathIncrease')
    
    # Filter by months
    dff = dff[(dff.index >= dt.datetime.strptime(months[month_values[0]],"%B %Y")) & (dff.index <= dt.datetime.strptime(months[month_values[1]],"%B %Y"))]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:,}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        title={
                'text': "Daily Deaths",
                'x':0.5,'xanchor': 'center',
                'font':{'size': 18}
                },
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        width=700,  
        xaxis=dict(
            title="Date",
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False  # Removes Y-axis grid lines
            )
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# In[15]:


@app.callback(
    Output('totdeaths', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('slider', 'value')])
    
# STEP 5: Update Figure
def update_figure(state_values,month_values):

    if state_values is None:
        dff = df.copy()
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='death')        
    elif state_values[0]=="All States":
        dff = df.copy()
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='death')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        dff = temp.pivot(index='date',columns='state',values='death')
    
    # Filter by months
    dff = dff[(dff.index >= dt.datetime.strptime(months[month_values[0]],"%B %Y")) & (dff.index <= dt.datetime.strptime(months[month_values[1]],"%B %Y"))]
    
    fig = go.Figure()
    for column in dff.columns.to_list():
        fig.add_trace(
            go.Scatter(
                x = dff.index,
                y = dff[column],
                name = column,
                mode='lines',
                opacity=0.8,
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:,}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        title={
                'text': "Total Deaths",
                'x':0.5,'xanchor': 'center',
                'font':{'size': 18}
                },
        hovermode='closest',plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor = 'white',
            font_size=16),
        width=700,  
        xaxis=dict(
            title="Date",
            zeroline=True,
            showgrid=False,  # Removes X-axis grid lines 
            ),
        yaxis=dict(
            zeroline=True, 
            showgrid=False  # Removes Y-axis grid lines
            ),
    )
        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig


# In[16]:


server = app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
