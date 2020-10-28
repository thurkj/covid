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
import plotly.io as pio

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# In[2]:


today = dt.datetime.now().strftime('%B %d, %Y')  # today's date. this will be useful when sourcing results 


# In[3]:


url = 'https://api.census.gov/data/2019/pep/population?get=NAME,POP&for=state:*'
    
response = requests.get(url)
population = pd.read_json(response.text)  # convert to dataframe
population.head()

population.rename(columns=population.iloc[0],inplace=True)
population.drop(0,inplace=True)
population.drop(['state'], axis=1,inplace=True)

abbrev = pd.read_csv('state_abbrev.csv',header=None)  # convert to dataframe
abbrev.rename(columns={0:'NAME',1:'state'},inplace=True)

population = population.merge(abbrev,on='NAME',how='outer')
population.drop(['NAME'], axis=1,inplace=True)
population['POP'] = population.POP.astype(float)


# In[4]:


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


# In[5]:


covid.dtypes


# In[6]:


covid['date'] = pd.to_datetime(covid['date'], format='%Y%m%d')
covid['month'] = covid['date'].dt.strftime('%B %Y') 

covid.drop(covid[covid['state'] == "AS"].index, inplace=True)         # Drop American Samoa
covid.drop(covid[covid['date'] < '2020-03-01'].index, inplace=True)   # Drop January and February when there are few cases

months = covid['month'].unique().tolist()
months.reverse()


# In[7]:


# Prepare for Dash
df = covid[['date','state','positiveIncrease','hospitalizedIncrease','deathIncrease','death']]
df.set_index('date',inplace=True)
df = df.sort_index()

# convert some data to 7-day moving averages
df_newcases = pd.DataFrame(df.groupby('state')['positiveIncrease'].rolling(7).mean())
df_newhospitalized = pd.DataFrame(df.groupby('state')['hospitalizedIncrease'].rolling(7).mean())
df_newdeaths = pd.DataFrame(df.groupby('state')['deathIncrease'].rolling(7).mean())

df_newcases = df_newcases.rename(columns={"positiveIncrease":"new_cases"})
df_newhospitalized = df_newhospitalized.rename(columns={"hospitalizedIncrease":"new_hospitalized"})
df_newdeaths = df_newdeaths.rename(columns={"deathIncrease":"new_deaths"})

# Merge results back into original df
df.reset_index(level=0, drop=False, inplace=True)
df.set_index(['state','date'],inplace=True)
df = df.sort_index()
df = df.merge(df_newcases,left_index=True, right_index=True)
df = df.merge(df_newhospitalized,left_index=True, right_index=True)
df = df.merge(df_newdeaths,left_index=True, right_index=True)
df = df.rename(columns={"death":"total_deaths"})

df.reset_index(level=[0,1], drop=False, inplace=True)
df = df.merge(population,on='state',how='outer')
df.dropna(subset=['state'],inplace=True)

df2 = df.copy()
df2.state = 'All States'
df = df.append(df2, ignore_index=True)

df.tail()


# In[8]:


app = dash.Dash()
server = app.server

header = html.H1(children="United States Covid-19 Trends (as of " + today + ")")

markdown_text = '''
The following graphs depict select Covid-19 trends by state. The graphs are interactive; e.g., hover your cursor over a data-series to observe specific values.

You can also use the filters below to choose a subset of states, periods, and/or standardize the data
to sharpen your analysis. 

Data Source: The Atlantic [Covid-19 Tracking Project](https://covidtracking.com) 
'''
markdown = dcc.Markdown(children=markdown_text)

# Dropdown
dropdown1 =  html.P([
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
    
dropdown2 =  html.P([
            html.Label("Present Raw Data or Population-adjusted (per 10,000 residents)"),
            dcc.Dropdown(
            id='normalization-dropdown',
             options=[
            {'label': 'Raw', 'value': 'Yes'},
            {'label': 'Per 10,000 Residents', 'value': 'No'},
            ],
            value='Yes',
            multi=False,
            searchable= True)
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

layout = html.Div(children=[header, markdown, dropdown1, slider, dropdown2, row1, row2], style={"text-align": "center","width":"95%"})
app.layout = layout


# In[9]:


#===========================================
# Daily Positive Cases
#===========================================

@app.callback(
    Output('positive', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('normalization-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,normalization_values,month_values):
        
    if state_values is None:
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_cases'] = 1e+4*dff['new_cases']/dff['POP'] 
        
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_cases')        
    elif state_values[0]=="All States":
        dff = df.copy()
        
        # Check for normalization
        if normalization_values[0]=="No":
            dff['new_cases'] = 1e+4*dff['new_cases']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_cases')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        
        # Check for normalization
        if normalization_values[0]=="No":
            temp['new_cases'] = 1e+4*temp['new_cases']/temp['POP'] 
            
        dff = temp.pivot(index='date',columns='state',values='new_cases')        

        
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
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:.1f}'
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


# In[10]:


#===========================================
# Currently Hospitalized
#===========================================

@app.callback(
    Output('curhospital', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('normalization-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,normalization_values,month_values):

    if state_values is None:
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_hospitalized'] = 1e+4*dff['new_hospitalized']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_hospitalized')        
    elif state_values[0]=="All States":
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_hospitalized'] = 1e+4*dff['new_hospitalized']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_hospitalized')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        
        # Check for normalization
        if normalization_values=='No':
            temp['new_hospitalized'] = 1e+4*temp['new_hospitalized']/temp['POP'] 
            
        dff = temp.pivot(index='date',columns='state',values='new_hospitalized')
    
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
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:.1f}'
            )
        )
      
    # Update remaining layout properties
    fig.update_layout(
        title={
                'text': "New Hospitalizations",
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


# In[11]:


#===========================================
# Daily Deaths
#===========================================

@app.callback(
    Output('newdeaths', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('normalization-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,normalization_values,month_values):

    if state_values is None:
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_deaths'] = 1e+4*dff['new_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_deaths')        
    elif state_values[0]=="All States":
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['new_deaths'] = 1e+4*dff['new_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='new_deaths')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        
        # Check for normalization
        if normalization_values=='No':
            temp['new_deaths'] = 1e+4*temp['new_deaths']/temp['POP'] 
            
        dff = temp.pivot(index='date',columns='state',values='new_deaths')
    
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
                hovertemplate = '<extra></extra>State: ' + column + '<br>Date: ' + dff.index.strftime('%m/%d') +'<br>Value: %{y:.1f}'
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


# In[12]:


#===========================================
# Total Number of Deaths
#===========================================

@app.callback(
    Output('totdeaths', 'figure'),
    [Input('state-dropdown', 'value')],
    [Input('normalization-dropdown', 'value')],
    [Input('slider', 'value')])
    
# Update Figure
def update_figure(state_values,normalization_values,month_values):

    if state_values is None:
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['total_deaths'] = 1e+4*dff['total_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='total_deaths')        
    elif state_values[0]=="All States":
        dff = df.copy()
        
        # Check for normalization
        if normalization_values=='No':
            dff['total_deaths'] = 1e+4*dff['total_deaths']/dff['POP'] 
            
        dff.drop(dff[dff['state'] == "All States"].index, inplace=True)
        dff = dff.pivot(index='date',columns='state',values='total_deaths')
    else:
        if not isinstance(state_values, list): state_values = [state_values]
        temp = df.loc[df['state'].isin(state_values)]
        
        # Check for normalization
        if normalization_values=='No':
            temp['total_deaths'] = 1e+4*temp['total_deaths']/temp['POP'] 
            
        dff = temp.pivot(index='date',columns='state',values='total_deaths')
    
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


# In[13]:


if __name__ == '__main__':
    app.run_server(debug=True)





