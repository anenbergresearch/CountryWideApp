import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import plotly.io as pio
import country_converter as coco

df = pd.read_csv('updated_unified_data.csv')
c40 = pd.read_excel('cityid-c40_crosswalk.xlsx')
ID_pop = df[df['Year']==2019][['ID','Population']]
c40_p = c40.merge(ID_pop, how = 'left', left_on = 'city_id', right_on ='ID')
total = c40_p[['ID','c40','continent']].merge(df, how = 'right', on ='ID')
df = total[['ID','City','c40','Country','continent','Year','Population','NO2','PM','O3']].copy()
df['CityCountry'] = df.City + ', ' + df.Country + ' (' +df.ID.apply(int).apply(str) +')'
pol = ['NO2','O3','PM']
import dash.dependencies
pio.templates.default = "plotly_white"

def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()
## Filter df
ds= total.query('Year <2005')
da = total.query('Year>=2005')
##Find 0 values in 2000
s =total.query('Year ==2000 & NO2==0')
ds.loc[(ds['ID'].isin(s.ID)),('NO2')] =np.nan
ds.loc[(ds.ID ==923),('NO2')]=np.nan
dfilt = pd.concat([ds,da])

dfilt['CityCountry'] = dfilt.City + ', ' + dfilt.Country + ' (' +dfilt.ID.apply(int).apply(str) +')'


def find_stats(dataframe):
    me = dataframe.groupby(['Country','Year']).mean(numeric_only=True)[['Population','PM','O3','NO2','Latitude','Longitude']].round(decimals= 2)
    dd = dataframe[['Country','Year','Population','O3','NO2','PM']].dropna()
    me['w_NO2']=dd.groupby(['Country','Year']).apply(w_avg,'NO2','Population')
    me['w_PM']=dd.groupby(['Country','Year']).apply(w_avg,'PM','Population')
    me['w_O3']=dd.groupby(['Country','Year']).apply(w_avg,'O3','Population')
    me.Population = me.Population.round(decimals=-3)
    me = me.reset_index()
    me['iso'] = coco.convert(names=me.Country,to='ISO3')

    _ma = dataframe.groupby(['Country','Year']).max(numeric_only=True)[['Population','PM','O3','NO2','Latitude','Longitude']].round(decimals = 2)
    _ma.Population = me.Population
    _ma = _ma.reset_index()

    _mi = dataframe.groupby(['Country','Year']).min(numeric_only=True)[['Population','PM','O3','NO2','Latitude','Longitude']].round(decimals = 2)
    _mi.Population = me.Population
    _mi = _mi.reset_index()
    return me,_ma,_mi

mean, _max, _min = find_stats(total)
fmean,fmax,fmin = find_stats(dfilt)

pd.options.plotting.backend = "plotly"

app = Dash(__name__)#, external_stylesheets=external_stylesheets)
server=app.server
colors = {
    'background': 'white',
    'text': '#468570'
}


available_indicators = ['O3','PM','NO2','w_O3','w_PM','w_NO2']

app.layout = html.Div([
    html.Div([
        html.Div(style={'backgroundColor': colors['background']}, children=[
            html.H1(
                children='13,000 Cities',
                style={
                    'textAlign': 'center',
                    'color': 'black'
                }
            ),

            html.Div(children='Exploring Countrywide Trends', style={
                'textAlign': 'center',
                'color': 'lightgray'
            })]),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='NO2'
            ),
            dcc.RadioItems(
                id='crossfilter-data-type',
                options=[{'label': i, 'value': i} for i in ['Filtered', 'Raw']],
                value='Filtered',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': 'Population', 'value': 'Population'}],
                value='Population'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],                
                value='Log',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'white',
        'backgroundColor': 'white',
        'padding': '10px 5px'
    }),
    

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'hovertext': 'United States'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series', hoverData={'points': [{'hovertext': 'Washington D.C., United States (860)'}]}),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Slider(
        id='crossfilter-year--slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=df['Year'].max(),
        marks={str(year): str(year) for year in df['Year'].unique()},
        step=None
    ), style={'width': '95%', 'padding': '0px 20px 20px 20px'})
])

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-data-type', 'value'),
     dash.dependencies.Input('crossfilter-year--slider', 'value'),
     ])


def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, data_type,
                 year_value):
    if data_type == 'Raw':
        m = mean.query('Year == @year_value')
    else:
        m = fmean.query('Year == @year_value')
    #m = m.query('@pop_limit[0] < Population <@pop_limit[1]')
    
    fig = px.choropleth(m, locations = 'iso',
            hover_name='Country',
            color = yaxis_column_name,hover_data={'Latitude':False,'Longitude':False,'iso':False}, color_continuous_scale='OrRd')

    #fig.update_layout(legend=dict(groupclick="toggleitem"))

        
    fig.update_layout(legend_title_text='')


    fig.update_traces(customdata=m['Country'])
    
    #fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


def create_time_series(city,means, title, cityname, axiscol_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= means.Year, y=means.Maximum, name = 'Maximum', 
                             marker = {'color':'lightgray'},line= {'color':'lightgray'},
        showlegend=False))
    fig.add_trace(go.Scatter(x= means.Year, y=means[axiscol_name], name = 'Mean '+axiscol_name, 
                             marker = {'color':'#4CB391'},line= {'color':'#4CB391'},
        showlegend=False))
    fig.add_trace(go.Scatter(x= means.Year, y=means['w_'+axiscol_name].round(decimals= 2), name = 'Pop-W Mean '+axiscol_name,opacity=0.7, 
                             marker = {'color':'#4CB391'},line= {'color':'#4CB391','dash':'dash'},
        showlegend=False))
    fig.add_trace(go.Scatter(x= means.Year, y=means.Minimum, name = 'Minimum', 
                             marker = {'color':'lightgray'},line= {'color':'lightgray'},
        showlegend=False))
    fig.add_trace(go.Scatter(x= means.Year, y=city.round(decimals= 2), name = cityname, 
                             marker = {'color':'#CC5500'},line= {'color':'#CC5500'},
        showlegend=False))
    
    # px.scatter(means, x= 'Year',y= ['Maximum',axiscol_name,'Minimum'],
    #                  color_discrete_sequence=['lightgray','red','lightgray'])

    fig.update_traces(mode='lines+markers')
    fig.update_layout(hovermode="x unified")
    #fig.update_xaxes(showgrid=False)

    #fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig

@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-data-type', 'value'),
     Input('crossfilter-year--slider','value')])
def update_y_timeseries(hoverData, yaxis_column_name, xaxis_type,data_type,year_value):
    
    for i in pol:
        if i in yaxis_column_name:
            yaxis_column_name = i
    if data_type == "Raw":
        dff = df[df['Country'] == hoverData['points'][0]['hovertext']]
    else:
        dff = dfilt[dfilt['Country']==hoverData['points'][0]['hovertext']]
    dff = dff.query('Year ==@year_value')
    country_name = dff['Country'].iloc[0]
    title = '<b>{}</b><br>{}'.format(country_name, yaxis_column_name)
    fig = px.scatter(dff, x='Population',
            y=yaxis_column_name,
            hover_name='CityCountry',
            color = 'c40',
            #opacity = 0.4,
            color_discrete_map= {'not_c40':'rgba(76, 179, 145,0.4)','c40':'rgba(30, 49, 133,0.9)'}
            )    
    fig.update_xaxes(title='Population', type='linear' if xaxis_type == 'Linear' else 'log')

    fig.update_yaxes(title=yaxis_column_name)
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=title)
    fig.update_layout(height = 225, margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_layout(showlegend=False)

    return fig

@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('x-time-series','hoverData'),
    dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-data-type', 'value')])
def update_x_timeseries(cityName, hoverData, yaxis_column_name, data_type):
    for i in pol:
        if i in yaxis_column_name:
            yaxis_column_name = i
    if data_type == 'Raw':
        _df = mean[mean['Country'] == hoverData['points'][0]['hovertext']][['Year',yaxis_column_name,'w_'+yaxis_column_name]]
        _df['Minimum'] = _min[_min['Country'] == hoverData['points'][0]['hovertext']][yaxis_column_name]
        _df['Maximum'] = _max[_max['Country'] == hoverData['points'][0]['hovertext']][yaxis_column_name]
        city = df[df.CityCountry ==cityName['points'][0]['hovertext']][yaxis_column_name]
    else:
        _df = fmean[fmean['Country'] == hoverData['points'][0]['hovertext']][['Year',yaxis_column_name,'w_'+yaxis_column_name]]
        _df['Minimum'] = fmin[fmin['Country'] == hoverData['points'][0]['hovertext']][yaxis_column_name]
        _df['Maximum'] = fmax[fmax['Country'] == hoverData['points'][0]['hovertext']][yaxis_column_name]
        city = dfilt[dfilt.CityCountry ==cityName['points'][0]['hovertext']][yaxis_column_name]
    country_name = hoverData['points'][0]['hovertext']
    city_name = cityName['points'][0]['hovertext']
    return create_time_series(city,_df, country_name,city_name,yaxis_column_name)

if __name__== '__main__':
    app.run_server(debug=False)