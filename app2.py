#Building my app for the Hospital Finder

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash_table

import numpy as np
import pandas as pd

#Load data and define functions for analysis:
master_df = pd.read_csv('./Data/Master_df.csv',dtype={'Provider ID':str})
master_df_week2 = pd.read_csv('./Data/master_df_week2.csv',dtype={'Provider ID':str})
zip_df = pd.read_csv('./Data/Zip_Code_data_cleaned.csv')

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
#                        'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css',
#                        'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
  html.H1(
    children='HospFinder',
    style={'text-align':'center'}
    ),
  html.Div([
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
        )]),
    html.P([
      html.Label('Weight of distance'),
      daq.Slider(
        id='distance-weight',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10}),
    html.P([
      html.Label('Weight of Recommendation Score'),
      daq.Slider(
        id='reco-weight',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10}),
    html.P([
      html.Label('Weight of Mortality Rates for ER procedures'),
      daq.Slider(
        id='mort-weight',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10}),
    html.P([
      html.Label('Weight of complications for elective procedures'),
      daq.Slider(
        id='compli-weight',
        min=0,
        max=10,
        value=5,
        marks={str(i):str(i) for i in range(11)},
        step=1)],style={'padding':10})
    ],style={'padding':10}),
  html.Button(id='submit-button', children='Submit'),
  html.Div(id='output-check'),
  html.Div(children=[
    html.H2(id='hosp-output')]),
  html.Div(id='table-output')

  ],style={'padding':10})
    
@app.callback(Output('output-check', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('ZIP-code-state', 'value'),
               State('distance-limit', 'value'),
               State('distance-weight','value'),
               State('reco-weight','value'),
               State('mort-weight','value'),
               State('compli-weight','value')])
def update_output(n_clicks, zipc, dist, dist_wt, reco_wt, mort_wt, compli_wt):
  if n_clicks is not None:
      temp = zip_df.loc[zip_df['Zip']==int(zipc)]
      if np.shape(temp)[0] == 0:
        return u'''
          Please check the ZIP code you entered.
          Alternatively enter a different ZIP code.
        '''
      else:
        return u'''
          The ZIP code entered was {},
          Distance limit was {},
          and the weights for Distance, Recommendation scores, Mortality and complications were {}, {}, {}, {}.
          For these values, the recommended hospital is:
      '''.format(zipc,int(dist),dist_wt,reco_wt,mort_wt,compli_wt)


@app.callback(Output('hosp-output', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('ZIP-code-state', 'value'),
               State('distance-limit', 'value'),
               State('distance-weight','value'),
               State('reco-weight','value'),
               State('mort-weight','value'),
               State('compli-weight','value')])
def get_hospital(n_clicks, zipc, dist, dist_wt, reco_wt, mort_wt, compli_wt):
    if n_clicks is not None:
      temp = zip_df.loc[zip_df['Zip']==int(zipc)]
      if np.shape(temp)[0] != 0:
        lat1 = temp['Latitude'].iloc[0]
        long1 = temp['Longitude'].iloc[0]

        reduced_hosps = master_df_week2[(master_df_week2['Latitude'].astype(float) < (lat1 + 1).astype(float))]
        reduced_hosps = reduced_hosps[(reduced_hosps['Latitude'].astype(float) > (lat1 - 1).astype(float))]
        reduced_hosps = reduced_hosps[(reduced_hosps['Longitude'] < (long1 + 1))]
        reduced_hosps = reduced_hosps[(reduced_hosps['Longitude'] > (long1 - 1))]

        def get_distance(row):
          row['Distance'] = get_distance_haversine(lat1,float(row['Latitude']),long1,float(row['Longitude']))
          return row

        reduced_hosps = reduced_hosps.apply(get_distance,1)
        reduced_hosps = reduced_hosps[reduced_hosps['Distance'] < int(dist)]

        reduced_hosps['Distance transformed'] = -1*(reduced_hosps['Distance'] - np.mean(reduced_hosps['Distance']))/np.std(reduced_hosps['Distance'])

        #If all weights are set to zero:
        if (dist_wt + reco_wt + mort_wt + compli_wt) == 0:
          dist_wt = 1
          reco_wt = 1
          mort_wt = 1
          compli_wt = 1
        #Weight all the rows for a final score:
        reduced_hosps['Final Score'] = reco_wt*reduced_hosps['Review Score Transformed'] + dist_wt*reduced_hosps['Distance transformed'] + mort_wt*reduced_hosps['Mortality transformed'] + compli_wt*reduced_hosps['Complications transformed']
        max_score = reco_wt*max(reduced_hosps['Review Score Transformed']) + dist_wt*max(reduced_hosps['Distance transformed']) + mort_wt*max(reduced_hosps['Mortality transformed']) + compli_wt*max(reduced_hosps['Complications transformed'])
        min_score = reco_wt*min(reduced_hosps['Review Score Transformed']) + dist_wt*min(reduced_hosps['Distance transformed']) + mort_wt*min(reduced_hosps['Mortality transformed']) + compli_wt*min(reduced_hosps['Complications transformed'])

        reduced_hosps['Final Score'] = (reduced_hosps['Final Score'] - min_score)/(max_score - min_score)

        reduced_hosps = reduced_hosps.sort_values('Final Score',ascending=False)
        hosp_name = reduced_hosps['Hospital name'].iloc[0]
        hosp_score = round(reduced_hosps['Final Score'].iloc[0]*100)
        return u'''
            {} with a composite score of {}/100

        '''.format(hosp_name, hosp_score)

@app.callback(Output('table-output','children'),
              [Input('submit-button', 'n_clicks')],
              [State('ZIP-code-state', 'value'),
               State('distance-limit', 'value'),
               State('distance-weight','value'),
               State('reco-weight','value'),
               State('mort-weight','value'),
               State('compli-weight','value')])
def display_table(n_clicks, zipc, dist, dist_wt, reco_wt, mort_wt, compli_wt):
    if n_clicks is not None:
      temp = zip_df.loc[zip_df['Zip']==int(zipc)]
      if np.shape(temp)[0] != 0:
        lat1 = temp['Latitude'].iloc[0]
        long1 = temp['Longitude'].iloc[0]

        reduced_hosps = master_df_week2[(master_df_week2['Latitude'].astype(float) < (lat1 + 1).astype(float))]
        reduced_hosps = reduced_hosps[(reduced_hosps['Latitude'].astype(float) > (lat1 - 1).astype(float))]
        reduced_hosps = reduced_hosps[(reduced_hosps['Longitude'] < (long1 + 1))]
        reduced_hosps = reduced_hosps[(reduced_hosps['Longitude'] > (long1 - 1))]

        def get_distance(row):
          row['Distance'] = get_distance_haversine(lat1,float(row['Latitude']),long1,float(row['Longitude']))
          return row

        reduced_hosps = reduced_hosps.apply(get_distance,1)
        reduced_hosps = reduced_hosps[reduced_hosps['Distance'] < int(dist)]

        reduced_hosps['Distance transformed'] = -1*(reduced_hosps['Distance'] - np.mean(reduced_hosps['Distance']))/np.std(reduced_hosps['Distance'])

        #If all weights are set to zero:
        if (dist_wt + reco_wt + mort_wt + compli_wt) == 0:
          dist_wt = 1
          reco_wt = 1
          mort_wt = 1
          compli_wt = 1
        #Weight all the rows for a final score:
        reduced_hosps['Final Score'] = reco_wt*reduced_hosps['Review Score Transformed'] + dist_wt*reduced_hosps['Distance transformed'] + mort_wt*reduced_hosps['Mortality transformed'] + compli_wt*reduced_hosps['Complications transformed']
        max_score = reco_wt*max(reduced_hosps['Review Score Transformed']) + dist_wt*max(reduced_hosps['Distance transformed']) + mort_wt*max(reduced_hosps['Mortality transformed']) + compli_wt*max(reduced_hosps['Complications transformed'])
        min_score = reco_wt*min(reduced_hosps['Review Score Transformed']) + dist_wt*min(reduced_hosps['Distance transformed']) + mort_wt*min(reduced_hosps['Mortality transformed']) + compli_wt*min(reduced_hosps['Complications transformed'])

        reduced_hosps['Final Score'] = (reduced_hosps['Final Score'] - min_score)/(max_score - min_score)

        reduced_hosps = reduced_hosps.sort_values('Final Score',ascending=False)

        sel_cols = ['Hospital name','Distance','Review Score','Mortality transformed','Complications transformed','Final Score']
        final_table = reduced_hosps[sel_cols]
        final_table['Final Score'] = round(final_table['Final Score']*100)
        final_table['Distance'] = round(final_table['Distance'],2)
        final_table['Mortality transformed'] = round(final_table['Mortality transformed'],2)
        final_table['Complications transformed'] = round(final_table['Complications transformed'],2)
        
        if np.shape(final_table)[0] > 5:
          final_table = final_table.iloc[:5,:]    
        return dash_table.DataTable(
          columns=[{"name": i, "id": i} for i in final_table.columns],
          data=final_table.to_dict('records')
          )


if __name__ == '__main__':
    app.run_server(debug=True)