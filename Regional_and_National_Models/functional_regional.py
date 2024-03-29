#Imports
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from google.colab import files
from datetime import datetime,timedelta,date
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input,Dense,Flatten,Conv1D,MaxPooling1D,concatenate,TimeDistributed,LSTM

#Hyper Parameters
epochs = 150                                             
gap = abs(date.today()-date(2021, 2, 28)).days                          #This sets the last day in the training data.
optimizer = keras.optimizers.Adam(learning_rate=0.0008)
LOSS = "huber_loss"
interval = 14                                                           #Length of training slices

'''
Importing and formating both the static and dynamic data.
'''

#Statics
county = pd.read_csv('https://github.com/BMW-lab-MSU/MSUBC-COVID-19/blob/4f52afcb0c4618ee681caf24d490edf8b47f090f/County_data_folder/data.csv?raw=true')
county = county.replace(regex={'WY': 'Wyoming', 'ID': 'Idaho', 'MT': 'Montana', 'SD':'South Dakota', 'ND': 'North Dakota', ' County':'', 'gop':1,'dem':0})
county_dict = county.drop(['Area_Name', 'State','FIPStxt','Unnamed: 0'], axis = 1)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(county_dict.to_numpy())
county_dict = pd.DataFrame(df_scaled, columns=['POP_ESTIMATE_2019', 'land_area', 'population_desnsity', 'Less than HS Diploma','HS Diploma','Some College','Bachelors','# in poverty','poverty_rate','Unemployment Rate','party'])
county_dict['County'] = county['Area_Name'] + ',' + ' '+ county['State']
#dynamic data
csse_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
df = pd.concat([csse_df[csse_df['Province_State'] == 'Wyoming'],csse_df[csse_df['Province_State'] == 'North Dakota'],csse_df[csse_df['Province_State'] == 'Montana'],csse_df[csse_df['Province_State'] == 'South Dakota'],csse_df[csse_df['Province_State'] == 'Idaho']], ignore_index=True)
df['County'] = df['Admin2'].str.cat(df['Province_State'], sep=', ')
x = pd.merge(df,county_dict, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
df = df.drop(list(x.index.values))
df = df.drop(columns=['UID','code3','iso2','iso3','FIPS','Country_Region','Lat','Long_','Combined_Key','Admin2','Province_State'])
headers_vals = list(df['County'])
headers_vals.insert(0,'Date') 
df = df.drop(columns=['County']).transpose().reset_index()
df.columns = headers_vals
df = df.set_index('Date') 
df_new_cases_only = pd.DataFrame()

#Convert to new case by day
for i in list(df):
  array = np.array([second - first for first, second in zip(df[i], df[i][1:])])
  array[array<0] = 0
  df_new_cases_only[i] = array
df = df_new_cases_only.astype(float)
county_dict = county_dict.set_index('County').T.to_dict('list') #Statics Data

'''
Train test split and data shaping
'''
day1mae, day3mae, day7mae, day1mse, day3mse, day7mse, day12mae, day32mae, day72mae, day12mse, day32mse, day72mse = ([] for i in range(12))

labels, features, statics_train,statics_test, counties = ([] for i in range(5))
for county in list(df):
    seq = list(df[county])
    for i in range(len(seq)-(interval)+2-gap):
        counties.append(county)
        array = seq[i:i+(interval+7)]
        features.append(np.array(array[:-7]))
        labels.append(np.array([array[-7],array[-6],array[-5],array[-4],array[-3],array[-2],array[-1]]))
features, labels, counties = shuffle(features, labels, counties)
features_train, features_test, labels_train, labels_test, counties_train, counties_test = train_test_split(features, labels, counties, test_size=0.2, random_state=10)
for i in counties_train:
    statics_train.append(np.array(county_dict[i]))
for i in counties_test:
    statics_test.append(np.array(county_dict[i]))

X_train,X_test,y_train,y_test,X_2_train,X_2_test = np.array([np.array(i) for i in features_train]),np.array([np.array(i) for i in features_test]),np.array(labels_train),np.array(labels_test),np.array([np.array(i) for i in statics_train]),np.array([np.array(i) for i in statics_test])
X_train_reshaped,X_test_reshaped,X_2_train_reshaped,X_2_test_reshaped = X_train.reshape((X_train.shape[0], 1, interval, 1)),X_test.reshape((X_test.shape[0], 1, interval, 1)),X_2_train.reshape((X_2_train.shape[0], X_2_train.shape[1], 1)),X_2_test.reshape((X_2_test.shape[0], X_2_train.shape[1], 1))

#Model Architecture
visible1 = Input(shape=(None, interval, 1))
conv11 = TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'))(visible1)
pool11 = TimeDistributed(MaxPooling1D(pool_size=2))(conv11)
flat1 = TimeDistributed(Flatten())(pool11)
hidden11 = LSTM(30, activation='relu', dropout=0.1)(flat1)
hidden12 = Dense(10, activation='relu')(hidden11)
hidden13 = Dense(10, activation='relu')(hidden12)
hidden14 = Dense(5, activation='relu')(hidden13)
hidden15 = Dense(5, activation='relu')(hidden14)
output1 = Dense(7)(hidden15)
visible2 = Input(shape=(11,1))
hidden21 = Dense(10, activation='relu')(visible2)
hidden22 = Dense(10, activation='relu')(hidden21)
flat2 = Flatten()(hidden22)
hidden23 = Dense(5, activation='relu')(flat2)
output2 = Dense(7)(hidden23)
merge = concatenate([hidden12, flat2])
finalhidden1 = Dense(130, activation='relu')(merge)
finalhidden2 = Dense(110, activation='relu')(finalhidden1)
hidden31 = layers.add([finalhidden2, flat2])
finalhidden3 = Dense(5)(hidden31)
finaloutput = Dense(7)(finalhidden3)
finalmodel = Model(inputs=[visible1,visible2], outputs=finaloutput)
finalmodel.compile(loss=LOSS, metrics=["mean_absolute_error"], optimizer=optimizer)
finalmodel.fit(x=[X_train_reshaped, X_2_train_reshaped],y=y_train,epochs=epochs,verbose=1)

globals()["count"] = 0
def evaluate(y_test_pred, y_test, y_train_pred, y_train):
    globals()["count"] = globals()["count"] + 1
    r2_train, mae_train,mse_train,r2_test,mae_test,mse_test = r2_score(y_train, y_train_pred),mean_absolute_error(y_train, y_train_pred),mean_squared_error(y_train, y_train_pred),r2_score(y_test, y_test_pred),mean_absolute_error(y_test, y_test_pred),mean_squared_error(y_test, y_test_pred)
    if count == 1:
      day1mae.append(mae_test)
      day1mse.append(mse_test)
    if count == 3:
      day3mae.append(mae_test)
      day3mse.append(mse_test)
    if count == 7:
      day7mae.append(mae_test)
      day7mse.append(mse_test)

y_test_pred_day1,y_test_pred_day2,y_test_pred_day3,y_test_pred_day4,y_test_pred_day5,y_test_pred_day6,y_test_pred_day7 = np.array([i[0] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]),np.array([i[1] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]),np.array([i[2] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]),np.array([i[3] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]),np.array([i[4] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]),np.array([i[5] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]),np.array([i[6] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])])
y_test_pred_day1,y_test_pred_day2,y_test_pred_day3,y_test_pred_day4,y_test_pred_day5,y_test_pred_day6,y_test_pred_day7 = [round(i) for i in y_test_pred_day1], [round(i) for i in y_test_pred_day2],[round(i) for i in y_test_pred_day3],[round(i) for i in y_test_pred_day4],[round(i) for i in y_test_pred_day5],[round(i) for i in y_test_pred_day6],[round(i) for i in y_test_pred_day7]
y_train_pred_day1,y_train_pred_day2,y_train_pred_day3,y_train_pred_day4,y_train_pred_day5,y_train_pred_day6,y_train_pred_day7 = [i[0] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])],[i[1] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])],[i[2] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])],[i[3] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])],[i[4] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])],[i[5] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])],[i[6] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])]
y_train_pred_day1,y_train_pred_day2,y_train_pred_day3,y_train_pred_day4,y_train_pred_day5,y_train_pred_day6,y_train_pred_day7 = [round(i) for i in y_train_pred_day1],[round(i) for i in y_train_pred_day2],[round(i) for i in y_train_pred_day3],[round(i) for i in y_train_pred_day4],[round(i) for i in y_train_pred_day5],[round(i) for i in y_train_pred_day6],[round(i) for i in y_train_pred_day7]
y_test_day1,y_test_day2,y_test_day3,y_test_day4,y_test_day5,y_test_day6,y_test_day7 = np.array([i[0] for i in y_test]),np.array([i[1] for i in y_test]),np.array([i[2] for i in y_test]),np.array([i[3] for i in y_test]),np.array([i[4] for i in y_test]),np.array([i[5] for i in y_test]),np.array([i[6] for i in y_test])
y_train_day1,y_train_day2,y_train_day3,y_train_day4,y_train_day5,y_train_day6,y_train_day7 = [i[0] for i in y_train],[i[1] for i in y_train],[i[2] for i in y_train],[i[3] for i in y_train],[i[4] for i in y_train],[i[5] for i in y_train],[i[6] for i in y_train]

# Evaluation
evaluate(y_test_pred_day1, y_test_day1, y_train_pred_day1, y_train_day1)
evaluate(y_test_pred_day2, y_test_day2, y_train_pred_day2, y_train_day2)
evaluate(y_test_pred_day3, y_test_day3, y_train_pred_day3, y_train_day3)
evaluate(y_test_pred_day4, y_test_day4, y_train_pred_day4, y_train_day4)
evaluate(y_test_pred_day5, y_test_day5, y_train_pred_day5, y_train_day5)
evaluate(y_test_pred_day6, y_test_day6, y_train_pred_day6, y_train_day6)
evaluate(y_test_pred_day7, y_test_day7, y_train_pred_day7, y_train_day7)

#Download Files

df22   = pd.DataFrame()
df22['a1'],df22['a3'],df22['a7'],df22['s1'],df22['s3'],df22['s7'] = day1mae,day3mae,day7mae,day1mse,day3mse,day7mse
df22.to_excel(excel_writer="regstats.xlsx")

from google.colab import files
files.download("/content/regstats.xlsx")

