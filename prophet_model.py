# -*- coding: utf-8 -*-
"""Prophet_Model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14LnlIIChfWyTB5AjgYGq6yLLvvdTQ1yo
"""

import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from prophet import Prophet

#Hyper Parameters
epochs = 1

array = []

county_dict = pd.read_csv('https://github.com/duckie22/CBOE/blob/a508a34b13c48904d337bd6c51b34ee8671e449c/Use4.csv?raw=true')
for i in county_dict['State']:
  county_dict['State'] = county_dict['State'].replace('ID','Idaho')
  county_dict['State'] = county_dict['State'].replace('ND','North Dakota')
  county_dict['State'] = county_dict['State'].replace('SD','South Dakota')
  county_dict['State'] = county_dict['State'].replace('WY','Wyoming')
  county_dict['State'] = county_dict['State'].replace('MT','Montana')
for i in county_dict['Area_Name']:
  x = i.replace('County','')
  y = x.replace('Parish','')
  z = y.replace('city','')
  a = z.replace('City','')
  b = a.rstrip(' ')
  array.append(b)
county_dict = county_dict.replace(to_replace ="gop",
                value = 1)
county_dict = county_dict.replace(to_replace ="dem",
                value = 2)
county_dict = county_dict.replace(to_replace ="nan",
                value = 0)
county_dict['Combo1'] = array
county_dict['Combo'] = county_dict['Combo1'] + ',' + ' ' + county_dict['State']
county_dict['POP_ESTIMATE_2019'] = (county_dict['POP_ESTIMATE_2019'] - county_dict['POP_ESTIMATE_2019'].min()) / (county_dict['POP_ESTIMATE_2019'].max() - county_dict['POP_ESTIMATE_2019'].min())
county_dict['land_area'] = (county_dict['land_area'] - county_dict['land_area'].min()) / (county_dict['land_area'].max() - county_dict['land_area'].min())
county_dict['# in poverty'] = (county_dict['# in poverty'] - county_dict['# in poverty'].min()) / (county_dict['# in poverty'].max() - county_dict['# in poverty'].min())
county_dict['poverty_rate'] = (county_dict['poverty_rate'] - county_dict['poverty_rate'].min()) / (county_dict['poverty_rate'].max() - county_dict['poverty_rate'].min())
county_dict['Unemployment Rate'] = (county_dict['Unemployment Rate'] - county_dict['Unemployment Rate'].min()) / (county_dict['Unemployment Rate'].max() - county_dict['Unemployment Rate'].min())
county_dict['population_density'] = (county_dict['population_density'] - county_dict['population_density'].min()) / (county_dict['population_density'].max() - county_dict['population_density'].min())
county_dict['Less than HS Diploma'] = (county_dict['Less than HS Diploma'] - county_dict['Less than HS Diploma'].min()) / (county_dict['Less than HS Diploma'].max() - county_dict['Less than HS Diploma'].min())
county_dict['HS Diploma'] = (county_dict['HS Diploma'] - county_dict['HS Diploma'].min()) / (county_dict['HS Diploma'].max() - county_dict['HS Diploma'].min())
county_dict['Some College'] = (county_dict['Some College'] - county_dict['Some College'].min()) / (county_dict['Some College'].max() - county_dict['Some College'].min())
county_dict['Bachelors'] = (county_dict['Bachelors'] - county_dict['Bachelors'].min()) / (county_dict['Bachelors'].max() - county_dict['Bachelors'].min())
county_dict = county_dict.drop(columns=['FIPStxt', 'State', 'Area_Name','Combo1'])
county_dict = county_dict.set_index('Combo').T.to_dict('list')

csse_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
wy_df = csse_df[csse_df['Province_State'] == 'Wyoming']
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
nd_df = csse_df[csse_df['Province_State'] == 'North Dakota']
mt_df = csse_df[csse_df['Province_State'] == 'Montana']
sd_df = csse_df[csse_df['Province_State'] == 'South Dakota']
id_df = csse_df[csse_df['Province_State'] == 'Idaho']
df = pd.concat([wy_df,nd_df,mt_df,sd_df,id_df], ignore_index=True)
df['Location'] = df['Admin2'].str.cat(df['Province_State'], sep=', ')
df = df.drop(columns=['UID','code3','iso2','iso3','FIPS','Country_Region','Lat','Long_','Combined_Key','Admin2','Province_State'])
headers_vals = list(df['Location'])
headers_vals.insert(0,'Date')
df = df.drop(columns=['Location'])
df = df.transpose().reset_index()



countess = 0

for i in df['index']:
  #a_string[:1] + "b" + a_string[1:]
  df['index'][countess] = i[:-2] + str(20) + i[-2:]
  countess = countess + 1

def conv_dates_series(df, col, old_date_format, new_date_format):

    df[col] = pd.to_datetime(df[col], format=old_date_format).dt.strftime(new_date_format)
    
    return df



new_date_format='%Y-%m-%d'
old_date_format='%m/%d/%Y'

conv_dates_series(df, 'index', old_date_format, new_date_format)


df.columns = headers_vals
df = df.set_index('Date')
df = df.drop(columns=['Out of WY, Wyoming','Unassigned, Wyoming','Out of ND, North Dakota','Unassigned, North Dakota','Out of MT, Montana',
                      'Unassigned, Montana','Out of SD, South Dakota','Unassigned, South Dakota','Out of ID, Idaho','Unassigned, Idaho'])

# Convert to new cases only
df_new_cases_only = pd.DataFrame()
for i in list(df):
  array = [second - first for first, second in zip(df[i], df[i][1:])]
  # Make all negative values 0
  pos_array = []
  for j in array:
      if j < 0:
          a = 0
      else:
          a = j
      pos_array.append(a)
  df_new_cases_only[i] = pos_array
    
# Convert all to floats
df = df[1:]
df_new_cases_only = df_new_cases_only.astype(float)
df2 = df_new_cases_only
df2 = df2.set_index(df.index)

predictions = []
real = []

for i in range(242):

  x = pd.DataFrame(df2.iloc[:, i]) 

  x = x.reset_index()

  x = pd.DataFrame(x) 

  x.columns = ['ds', 'y']

  m = Prophet()
  m.fit(x)

  future = m.make_future_dataframe(periods=307)

  forecast = m.predict(future)

  real.append(np.array(df2.iloc[:, i][0:403]))
  predictions.append(forecast['yhat'][0:403].values)

mae_test = mean_absolute_error(real, predictions)

mse_test = mean_squared_error(real, predictions)