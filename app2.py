#Building my app for the Hospital Finder

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash_table

import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import os

#Load data and define functions for analysis:
master_df = pd.read_csv('./Data/Master_df.csv',dtype={'Provider ID':str})
master_df_transformed = pd.read_csv('./Data/master_df_transformed.csv',dtype={'Provider ID':str})
zip_df = pd.read_csv('./Data/Zip_Code_data_cleaned.csv')

#This token is specific to this app and will not work for other applications:
MAPBOX_KEY = 'sk.eyJ1IjoiemJwdmFydW4iLCJhIjoiY2p4NG9zYW84MGMwajQ5dDZpcmZ6bTNzYSJ9.gLDZ-K0JIJGgokXeofgbCw'

def get_distance_haversine(lat1,lat2,long1,long2):
    import math
    del_lat = math.radians(lat1 - lat2)
    del_long = math.radians(long1 - long2)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    
    R = 6371 #Average radius of the earth in km
    a = (math.sin(del_lat/2))**2 + math.cos(lat1)*math.cos(lat2)*((math.sin(del_long/2))**2)
    c = 2*math.atan2(np.sqrt(a),np.sqrt(1 - a))
    dist = R*c #This is the distance in km
    dist_mi = dist/1.609
    return dist_mi

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
                        'https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
#                        'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
#                        'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js']

def get_relevant_hosps(zipc, dist, specialty):
  #Utility function: Get the relevant hospital shortlist for given specialty and ZIP code.
  temp = zip_df.loc[zip_df['Zip']==int(zipc)]
  if np.shape(temp)[0] != 0:
    lat1 = temp['Latitude'].iloc[0]
    long1 = temp['Longitude'].iloc[0]

    if specialty == 'OB/GYN':
      reduced_hosps = master_df_transformed[master_df_transformed['Num_OB/GYN'] > 0].copy()
    elif specialty == 'Orthopedic Surgery':
      reduced_hosps = master_df_transformed[master_df_transformed['Num_Ortho'] > 0].copy()
    else:
      reduced_hosps = master_df_transformed[master_df_transformed['Num_Pediatric'] > 0].copy()
    

    reduced_hosps = reduced_hosps[(reduced_hosps['Latitude'].astype(float) < (lat1 + 1).astype(float))]
    reduced_hosps = reduced_hosps[(reduced_hosps['Latitude'].astype(float) > (lat1 - 1).astype(float))]
    reduced_hosps = reduced_hosps[(reduced_hosps['Longitude'] < (long1 + 1))]
    reduced_hosps = reduced_hosps[(reduced_hosps['Longitude'] > (long1 - 1))].copy()

    if specialty == 'OB/GYN':
      bins = [0, 10, 50, 100, np.inf]
      labels=[-0.5,0,0.5,1]
      reduced_hosps['Dept_size_transformed'] = pd.cut(reduced_hosps['Num_OB/GYN'], bins=bins, labels=labels)
    elif specialty == 'Orthopedic Surgery':
      bins = [0, 10, 50, 100, np.inf]
      labels=[-0.5,0,0.5,1]
      reduced_hosps['Dept_size_transformed'] = pd.cut(reduced_hosps['Num_Ortho'], bins=bins, labels=labels)
    else:
      bins = [0, 2, 5, 10, 50, np.inf]
      labels=[-1, -0.5, 0, 0.5, 1]
      reduced_hosps['Dept_size_transformed'] = pd.cut(reduced_hosps['Num_Pediatric'], bins=bins, labels=labels)
      bins = [0, 10, 50, 100, 500, np.inf]
      labels=[-1, -0.5, 0, 0.5, 1]
      reduced_hosps['Num_Assist Transformed'] = pd.cut(reduced_hosps['Num_Assist'], bins=bins, labels=labels)

    def get_distance(row):
      row['Distance'] = get_distance_haversine(lat1,float(row['Latitude']),long1,float(row['Longitude']))
      return row

    reduced_hosps = reduced_hosps.apply(get_distance,1)
    reduced_hosps = reduced_hosps[reduced_hosps['Distance'] < int(dist)]

    reduced_hosps['Distance transformed'] = -1*(reduced_hosps['Distance'] - np.mean(reduced_hosps['Distance']))/np.std(reduced_hosps['Distance'])
    return reduced_hosps

def compute_final_score(reduced_hosps,specialty, dept_size_wt, dist_wt, var_wt_1, var_wt_2, var_wt_3, var_wt_4):
  #Utility function: Compute the final score for the relevant hospitals from all other parameters.
  if specialty == 'OB/GYN':
    reduced_hosps['Final Score'] = dept_size_wt*reduced_hosps['Dept_size_transformed'] + dist_wt*reduced_hosps['Distance transformed'] + var_wt_1*reduced_hosps['Blood clots post surgery Score'] + var_wt_2*reduced_hosps['Post surgery Sepsis Score'] + var_wt_3*reduced_hosps['Abdomen Open Wound Score'] + var_wt_4*reduced_hosps['Accidental Lacerations Score']
    max_score = dept_size_wt*max(reduced_hosps['Dept_size_transformed']) + dist_wt*max(reduced_hosps['Distance transformed']) + var_wt_1*max(reduced_hosps['Blood clots post surgery Score']) + var_wt_2*max(reduced_hosps['Post surgery Sepsis Score']) + var_wt_3*max(reduced_hosps['Abdomen Open Wound Score']) + var_wt_4*max(reduced_hosps['Accidental Lacerations Score'])
    min_score = dept_size_wt*min(reduced_hosps['Dept_size_transformed']) + dist_wt*min(reduced_hosps['Distance transformed']) + var_wt_1*min(reduced_hosps['Blood clots post surgery Score']) + var_wt_2*min(reduced_hosps['Post surgery Sepsis Score']) + var_wt_3*min(reduced_hosps['Abdomen Open Wound Score']) + var_wt_4*min(reduced_hosps['Accidental Lacerations Score'])
  
  elif specialty == 'Orthopedic Surgery':
    reduced_hosps['Final Score'] = dept_size_wt*reduced_hosps['Dept_size_transformed'] + dist_wt*reduced_hosps['Distance transformed'] + var_wt_1*reduced_hosps['Hip and Knee Replacement Score'] + var_wt_2*reduced_hosps['Postop Dialysis need Score'] + var_wt_3*reduced_hosps['Blood clots post surgery Score'] + var_wt_4*reduced_hosps['Post surgery Sepsis Score']
    max_score = dept_size_wt*max(reduced_hosps['Dept_size_transformed']) + dist_wt*max(reduced_hosps['Distance transformed']) + var_wt_1*max(reduced_hosps['Hip and Knee Replacement Score']) + var_wt_2*max(reduced_hosps['Postop Dialysis need Score']) + var_wt_3*max(reduced_hosps['Blood clots post surgery Score']) + var_wt_4*max(reduced_hosps['Post surgery Sepsis Score'])
    min_score = dept_size_wt*min(reduced_hosps['Dept_size_transformed']) + dist_wt*min(reduced_hosps['Distance transformed']) + var_wt_1*min(reduced_hosps['Hip and Knee Replacement Score']) + var_wt_2*min(reduced_hosps['Postop Dialysis need Score']) + var_wt_3*min(reduced_hosps['Blood clots post surgery Score']) + var_wt_4*min(reduced_hosps['Post surgery Sepsis Score'])

  else:
    reduced_hosps['Final Score'] = dept_size_wt*reduced_hosps['Dept_size_transformed'] + dist_wt*reduced_hosps['Distance transformed'] + var_wt_1*reduced_hosps['Pneumonia death Score'] + var_wt_2*reduced_hosps['Num_Assist Transformed']
    max_score = dept_size_wt*max(reduced_hosps['Dept_size_transformed']) + dist_wt*max(reduced_hosps['Distance transformed']) + var_wt_1*max(reduced_hosps['Pneumonia death Score']) + var_wt_2*max(reduced_hosps['Num_Assist Transformed'])
    min_score = dept_size_wt*min(reduced_hosps['Dept_size_transformed']) + dist_wt*min(reduced_hosps['Distance transformed']) + var_wt_1*min(reduced_hosps['Pneumonia death Score']) + var_wt_2*min(reduced_hosps['Num_Assist Transformed'])

  reduced_hosps['Final Score'] = (reduced_hosps['Final Score'] - min_score)/(max_score - min_score)

  reduced_hosps = reduced_hosps.sort_values('Final Score',ascending=False)
  return reduced_hosps

def get_final_table(reduced_hosps,specialty):
  #Utility function to compute the final table to be displayed:
  if specialty == 'OB/GYN':
    sel_cols = ['Hospital name','Num_OB/GYN','Blood clots post surgery Score','Post surgery Sepsis Score','Abdomen Open Wound Score','Accidental Lacerations Score']
    df1 = master_df.loc[reduced_hosps.index[:5],sel_cols]
    df1.rename(columns={'Num_OB/GYN':'Number of Specialists',
                        'Blood clots post surgery Score':'Rate of blood clot complications',
                        'Post surgery Sepsis Score':'Rate of Sepsis post surgery',
                        'Abdomen Open Wound Score':'Rate of wound opening in abdomen post surgery',
                        'Accidental Lacerations Score':'Rate of accidental lacerations during surgery'}, inplace=True)
    df2 = reduced_hosps[['Hospital name','Distance','Final Score']]
    df2 = df2.iloc[:5,:]
    final_table = df1.merge(df2,left_on='Hospital name',right_on='Hospital name')
    final_cols = ['Hospital name','Distance','Number of Specialists','Rate of blood clot complications','Rate of Sepsis post surgery','Rate of wound opening in abdomen post surgery','Rate of accidental lacerations during surgery','Final Score']
    final_table = final_table[final_cols]
  elif specialty == 'Orthopedic Surgery':
    sel_cols = ['Hospital name','Num_Ortho','Hip and Knee Replacement Score','Postop Dialysis need Score','Blood clots post surgery Score','Post surgery Sepsis Score']
    df1 = master_df.loc[reduced_hosps.index[:5],sel_cols]
    df1.rename(columns={'Num_Ortho':'Number of Specialists',
                        'Blood clots post surgery Score':'Rate of blood clot complications',
                        'Post surgery Sepsis Score':'Rate of Sepsis post surgery',
                        'Hip and Knee Replacement Score':'Rate of complications for hip/knee replacements',
                        'Postop Dialysis need Score':'Rate of kidney injury requiring Dialysis post surgery'}, inplace=True)
    df2 = reduced_hosps[['Hospital name','Distance','Final Score']]
    df2 = df2.iloc[:5,:]
    final_table = df1.merge(df2,left_on='Hospital name',right_on='Hospital name')
    final_cols = ['Hospital name','Distance','Number of Specialists','Rate of complications for hip/knee replacements','Rate of kidney injury requiring Dialysis post surgery','Rate of blood clot complications','Rate of Sepsis post surgery','Final Score']
    final_table = final_table[final_cols]
  else:
    sel_cols = ['Hospital name','Num_Pediatric','Pneumonia death Score','Num_Assist']
    df1 = master_df.loc[reduced_hosps.index[:5],sel_cols]
    df1.rename(columns={'Num_Pediatric':'Number of Specialists',
                        'Pneumonia death Score':'Mortality rate for Pneumonia',
                        'Num_Assist':'Number of Assistants on-site'}, inplace=True)
    df2 = reduced_hosps[['Hospital name','Distance','Final Score']]
    df2 = df2.iloc[:5,:]
    final_table = df1.merge(df2,left_on='Hospital name',right_on='Hospital name')
    final_cols = ['Hospital name','Distance','Number of Specialists','Mortality rate for Pneumonia','Number of Assistants on-site','Final Score']
    final_table = final_table[final_cols]

  final_table['Final Score'] = round(final_table['Final Score']*100)
  final_table['Distance'] = round(final_table['Distance'],2)
  return final_table

def display_map_and_hosps(reduced_hosps,zipc,dist):
  
  if zipc is not None:
    temp = zip_df.loc[zip_df['Zip']==int(zipc)]
    if np.shape(temp)[0] != 0:
      lat1 = temp['Latitude'].iloc[0]
      long1 = temp['Longitude'].iloc[0]
      if int(dist) == 5:
        marker_size = 10
        zoom = 10
      elif int(dist) == 10:
        marker_size = 10
        zoom = 9
      elif int(dist) == 20:
        marker_size = 8
        zoom = 7
      elif int(dist) == 50:
        marker_size = 6
        zoom = 6
      return {
          "data": [
                  {
                      "type": "scattermapbox",
                      "lat": reduced_hosps['Latitude'],
                      "lon": reduced_hosps['Longitude'],
                      "text": reduced_hosps['Hospital name'],
                      "mode": "markers",
                      "marker": {
                          "size": marker_size,
                          "opacity": 1.0
                      }
                  }
                  ],
          "layout": {
              "autosize": True,
              "hovermode": "closest",
              "mapbox": {
                  "accesstoken": MAPBOX_KEY,
                  "bearing": 0,
                  "center": {
                      "lat": lat1,
                      "lon": long1
                  },
                  "pitch": 0,
                  "zoom": zoom,
                  "style": "outdoors"
              }
          }
      }
  else:
      return {
          "data": [
                  {
                      "type": "scattermapbox",
                      "lat": master_df['Latitude'],
                      "lon": master_df['Longitude'],
                      "text": master_df['Hospital name'],
                      "mode": "markers",
                      "marker": {
                          "size": 3,
                          "opacity": 1.0
                      }
                  }
                  ],
          "layout": {
              "autosize": True,
              "hovermode": "closest",
              "mapbox": {
                  "accesstoken": os.environ.get("MAPBOX_KEY"),
                  "bearing": 0,
                  "center": {
                      "lat": 40,
                      "lon": -98.5
                  },
                  "pitch": 0,
                  "zoom": 2,
                  "style": "outdoors"
              }
          }
      }

def make_comparison_graphs(final_table,specialty):
  if (specialty == 'OB/GYN') | (specialty == 'Orthopedic Surgery'):
    num_plots = 6
    temp = final_table.columns[1:-1].to_list()
    fig = tools.make_subplots(rows=2,cols=3,subplot_titles=temp)
    for i in np.arange(num_plots):
      trace = go.Bar(
        x=final_table.iloc[:,0],
        y=final_table.iloc[:,(i+1)]
        )
      row_var = int(np.floor(i/3) + 1)
      col_var = int(i%3 + 1)
      fig.append_trace(trace,row_var,col_var)
      fig['layout'].update(xaxis = dict(showticklabels=False), xaxis2 = dict(showticklabels=False),
                        xaxis3 = dict(showticklabels=False), xaxis4 = dict(showticklabels=False),
                        xaxis5 = dict(showticklabels=False), xaxis6 = dict(showticklabels=False))
  else:
    num_plots = 4
    fig = tools.make_subplots(rows=2,cols=2,subplot_titles=final_table.columns[1:-2].to_list())
    for i in np.arange(num_plots):
      trace = go.Bar(
        x=final_table.iloc[:,0],
        y=final_table.iloc[:,(i+1)]
        )
      row_var = int(np.floor(i/2) + 1)
      col_var = int(i%2 + 1)
      fig.append_trace(trace,row_var,col_var)
      fig['layout'].update(xaxis = dict(showticklabels=False), xaxis2 = dict(showticklabels=False),
                        xaxis3 = dict(showticklabels=False), xaxis4 = dict(showticklabels=False))

  fig['layout'].update(title = 'Metrics for each hospital')

  return fig


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
server.secret_key = os.environ.get("SECRET_KEY", "secret")

app.layout = html.Div(children=[
  html.Div([
    html.Div([
    html.H1(
    children='HospFinder',
    style={'text-align':'center'}
    )],className="col-md-12")
    ],className='row')
  ,

  #Start of 2nd row:
  html.Div([
    html.Div([
    html.P([
        html.Label('Enter the specialty you are looking for:'),
        # dcc.Input(id='mother_birth', value=1952, type='number'),
        dcc.Dropdown(
            id='specialty',
            options=[{'label': i, 'value':i} for i in ['OB/GYN','Orthopedic Surgery','Pediatric Care']],\
            value='OB/GYN'
        )
    ],
    style={'width': '250px'}),
    html.P([
      html.Label('Enter a ZIP code to search around:'),
      dcc.Input(id='ZIP-code-state', type='text', placeholder='Enter a ZIP code')]),
    html.P([
      html.Label('Specify your distance limit (in miles)'),
      dcc.RadioItems(
            id='distance-limit',
            options=[{'label': i, 'value': i} for i in [str(5), str(10), str(20), str(50)]],
            value=str(5),
            labelStyle={'display': 'inline-block'}
        )])],className="col-md-2"),
    html.Div([
      html.Button(id='submit-button', children='Submit')],className="col-md-2"),
    html.Div([
      html.Div(children=[
        html.Div(id='output-check'),
        html.H2(id='hosp-output')])
      ],className='col-md-8')
    ],className="row")
  ,
  #Start next row with sliders and radar chart:
  html.Div([
    html.Div([
    html.Div('Please specify how important the following are to you (higher number is better outcomes):'),
    html.P([
      html.Label('Distance'),
      daq.Slider(
        id='distance-weight',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10}),
    html.P([
      html.Label('Size of Department'),
      daq.Slider(
        id='dept-size-weight',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10}),
    html.P([
      html.Label(id='slider-head-1'),
      daq.Slider(
        id='weight-1',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10}),
    html.P([
      html.Label(id='slider-head-2'),
      daq.Slider(
        id='weight-2',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10}),
    html.P([
      html.Label(id='slider-head-3'),
      daq.Slider(
        id='weight-3',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10}),
    html.P([
      html.Label(id='slider-head-4'),
      daq.Slider(
        id='weight-4',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10})
    ],className="col-md-4"),
    html.Div([
      dcc.Graph(id='map-output')],className="col-md-8")
    ],className="row"),
  html.Hr(),
  #Final row with table for now:
  html.Div([html.Div([
    dcc.Graph(id='charts-output')
    ],className="col-md-12")
  ],className="row")
  
  ],className="container-fluid")

#Update the weight labels based on Department selected
@app.callback(
    [Output('slider-head-1', 'children'),
     Output('slider-head-2', 'children'),
     Output('slider-head-3', 'children'),
     Output('slider-head-4', 'children')],
    [Input('specialty', 'value')])
def get_weight_labels(specialty):
  if specialty == 'OB/GYN':
    wt_3 = 'Rate of blood clot complications'
    wt_4 = 'Rate of sepsis complications'
    wt_5 = 'Rate of wound splitting open on abdomen or pelvis'
    wt_6 = 'Rate of accidental cuts during Surgery'
  elif specialty == 'Orthopedic Surgery':
    wt_3 = 'Rate of surgical complications'
    wt_4 = 'Rate of kidney complications requiring dialysis'
    wt_5 = 'Rate of blood clot complications'
    wt_6 = 'Rate of sepsis complications'
  else:
    wt_3 = 'Pneumonia mortality rate'
    wt_4 = 'Number of assistants available'
    wt_5 = 'N/A'
    wt_6 = 'N/A' 
  return wt_3, wt_4, wt_5, wt_6

#Map callback:
@app.callback(Output('map-output', 'figure'),
              [Input('submit-button','n_clicks')],
              [State('ZIP-code-state','value'),
               State('specialty','value'),
               State('distance-limit','value')])
def populate_map(n_clicks,zipc,specialty,dist):
  if zipc is not None:
    reduced_hosps = get_relevant_hosps(zipc,dist,specialty)
    return display_map_and_hosps(reduced_hosps,zipc,dist)
  else:
    return display_map_and_hosps(master_df,zipc,dist)

@app.callback(Output('output-check', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('specialty','value'),
               State('ZIP-code-state', 'value'),
               State('distance-limit', 'value')])
def check_zip(n_clicks, specialty, zipc, dist):
  if n_clicks is not None:
      temp = zip_df.loc[zip_df['Zip']==int(zipc)]
      if np.shape(temp)[0] == 0:
        return u'''
          Please check the ZIP code you entered.
          Alternatively enter a different ZIP code.
        '''
      else:
        return u'''
          You searched for {} departments in ZIP code {} with a distance limit of {}.
          The recommended hospitals are as follows:
      '''.format(specialty, zipc,int(dist))


@app.callback(Output('hosp-output', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('specialty', 'value'),
               State('ZIP-code-state', 'value'),
               State('distance-limit', 'value'),
               State('distance-weight','value'),
               State('dept-size-weight','value'),
               State('weight-1','value'),
               State('weight-2','value'),
               State('weight-3','value'),
               State('weight-4','value')])
def get_hospital(n_clicks, specialty, zipc, dist, dist_wt, dept_size_wt, var_wt_1, var_wt_2, var_wt_3, var_wt_4):
    if n_clicks is not None:
      reduced_hosps = get_relevant_hosps(zipc,dist,specialty)
      #If all weights are set to zero:
      if (dist_wt + dept_size_wt + var_wt_1 + var_wt_2 + var_wt_3 + var_wt_4) == 0:
        dist_wt = 1
        dept_size_wt = 1
        var_wt_1 = 1
        var_wt_2 = 1
        var_wt_3 = 1
        var_wt_4 = 1
      #Weight all the rows for a final score:
      reduced_hosps = compute_final_score(reduced_hosps,specialty, dept_size_wt, dist_wt, var_wt_1, var_wt_2, var_wt_3, var_wt_4)
      hosp_name = reduced_hosps['Hospital name'].iloc[0]
      hosp_score = round(reduced_hosps['Final Score'].iloc[0]*100)
      return u'''
          {} with a composite score of {}/100

      '''.format(hosp_name, hosp_score)

@app.callback(Output('charts-output','figure'),
              [Input('submit-button', 'n_clicks')],
              [State('specialty', 'value'),
               State('ZIP-code-state', 'value'),
               State('distance-limit', 'value'),
               State('distance-weight','value'),
               State('dept-size-weight','value'),
               State('weight-1','value'),
               State('weight-2','value'),
               State('weight-3','value'),
               State('weight-4','value')])
def display_table(n_clicks, specialty, zipc, dist, dist_wt, dept_size_wt, var_wt_1, var_wt_2, var_wt_3, var_wt_4):
    if n_clicks is not None:
      reduced_hosps = get_relevant_hosps(zipc,dist,specialty)
    
      #If all weights are set to zero:
      if (dist_wt + dept_size_wt + var_wt_1 + var_wt_2) == 0:
        dist_wt = 1
        dept_size_wt = 1
        var_wt_1 = 1
        var_wt_2 = 1
      #Weight all the rows for a final score:
      reduced_hosps = compute_final_score(reduced_hosps,specialty, dept_size_wt, dist_wt, var_wt_1, var_wt_2, var_wt_3, var_wt_4)

      final_table = get_final_table(reduced_hosps,specialty)
      
      return make_comparison_graphs(final_table,specialty)
    else:
      return {
        "data": [
            {
                "type": "bar",
                "x": ['A','B','C'],
                "y": [1,2,3]
            }
        ],
        "layout": {
            "title": "Enter ZIP code to continue",
        }
    }



if __name__ == '__main__':
    app.run_server(debug=True)