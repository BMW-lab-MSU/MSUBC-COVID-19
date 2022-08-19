#Imports
from __future__ import absolute_import, division, print_function
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

gap = abs(date.today()-date(2021, 2, 28)).days                          #This sets the last day in the training data.
interval = 14
optimizer = keras.optimizers.Adam(learning_rate=0.0008)
LOSS = "huber_loss"
epochs = 150

average_mae_test1 = []
average_mae_test2 = []
average_mae_test3 = []
average_mae_test4 = []
average_mae_test5 = []
average_mae_test6 = []
average_mae_test7 = []
average_mse_test1 = []
average_mse_test2 = []
average_mse_test3 = []
average_mse_test4 = []
average_mse_test5 = []
average_mse_test6 = []
average_mse_test7 = []
average_r2_test1 = []
average_r2_test2 = []
average_r2_test3 = []
average_r2_test4 = []
average_r2_test5 = []
average_r2_test6 = []
average_r2_test7 = []



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

# New way to generate arrays of length X
features = []
labels = []
counties = []

# Iterate across dataframe, one county at a time
for county in list(df):
    seq = list(df[county])

    
    # Create multiple arrays from each county
    for i in range(len(seq)-(interval+7)-gap):
        array = seq[i:i+(interval+7)]
        x_array = array[:-7]
        y_array = np.array([array[-7],array[-6],array[-5],array[-4],array[-3],array[-2],array[-1]])
        counties.append(county)
        features.append(x_array)
        labels.append(y_array)

# Shuffle dataset
features, labels, counties = shuffle(features, labels, counties)

# Train Test Split
features_train, features_test, labels_train, labels_test, counties_train, counties_test = train_test_split(features, labels, counties, test_size=0.2, random_state=10)

# Sync up static variables
statics_train = []
statics_test = []
#statics_validation = [] 

for i in counties_train:
    array = np.array(county_dict[i])
    statics_train.append(array)
for i in counties_test:
    array = np.array(county_dict[i])
    statics_test.append(array)



X_train,X_test,y_train,y_test,X_2_train,X_2_test = np.array([np.array(i) for i in features_train]),np.array([np.array(i) for i in features_test]),np.array(labels_train),np.array(labels_test),np.array([np.array(i) for i in statics_train]),np.array([np.array(i) for i in statics_test])
X_train_reshaped,X_test_reshaped,X_2_train_reshaped,X_2_test_reshaped = X_train.reshape((X_train.shape[0], 1, interval, 1)),X_test.reshape((X_test.shape[0], 1, interval, 1)),X_2_train.reshape((X_2_train.shape[0], X_2_train.shape[1], 1)),X_2_test.reshape((X_2_test.shape[0], X_2_train.shape[1], 1))


checkpoint_path = "training_1/1.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)

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
finalmodel.fit(x=[X_train_reshaped, X_2_train_reshaped],y=y_train,epochs=epochs,callbacks = [cp_callback],verbose=1)

y_1_graph = []
y_3_graph = []
y_7_graph = []

y_1_graph_real = []

z = 0

while z <= 68: #Change to 91 if zeros are included (for apprx. three month predictoins)


  gap = abs(date.today()-date(2021, 2, 28)).days                          #This sets the last day in the training data.

  average_mae_test1 = []
  average_mae_test2 = []
  average_mae_test3 = []
  average_mae_test4 = []
  average_mae_test5 = []
  average_mae_test6 = []
  average_mae_test7 = []
  average_mse_test1 = []
  average_mse_test2 = []
  average_mse_test3 = []
  average_mse_test4 = []
  average_mse_test5 = []
  average_mse_test6 = []
  average_mse_test7 = []
  average_r2_test1 = []
  average_r2_test2 = []
  average_r2_test3 = []
  average_r2_test4 = []
  average_r2_test5 = []
  average_r2_test6 = []
  average_r2_test7 = []


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

  # New way to generate arrays of length X
  interval = 14
  gap = gap - z
  features_test = []
  features_train = []
  labels_test = []
  labels_train = []
  counties_test = []
  counties_trian = []
  counties = []
  features = []
  labels = []

  j = 0

  for county in list(df):
    seq = list(df[county])

    #Can be changed to any County 
    target_county = 'Ada, Idaho' #Format: County, State

    if seq == list(df[target_county]):
      j = j + 1

    else:
      for i in range(len(seq)-(interval+7)-gap):
          array = seq[i:i+(interval+7)]
          x_array = array[:-7]
          y_array = np.array([array[-7],array[-6],array[-5],array[-4],array[-3],array[-2],array[-1]])
          counties.append(county)
          features.append(x_array)
          labels.append(y_array)

  features, labels, counties = shuffle(features, labels, counties)

  # Train Test Split
  features_train, features_tests, labels_train, labels_tests, counties_train, counties_tests = train_test_split(features, labels, counties, test_size=0.2, random_state=10)


  seq = list(df[target_county])     

  gap2 = gap
    
  #Remove Zeros

  #-----------------#    Commnt out this block to have the network predict with zeros included
  
  gap2 = gap + 4
  for i in range(len(seq)-(interval+7)-gap,len(seq)):
    if seq[i] == 0:
      gap2 = gap2 -1 

  
  flag = True

  while flag==True:                                         
    try:
      seq.remove(0)
    except Exception:
      flag = False
  #-----------------#

  x_array = seq[(len(seq)-gap2-interval):(len(seq)-gap2)]
  print(x_array)
  y_array = seq[(len(seq)-gap2):len(seq)-gap2+7]
  print(y_array)
  counties_test.append(target_county)
  features_test.append(x_array)
  labels_test.append(y_array)

  statics_train = []
  statics_test = []

  for i in counties_train:
      array = np.array(county_dict[i])
      statics_train.append(array)
  for i in counties_test:
      array = np.array(county_dict[i])
      statics_test.append(array)

  X_train,X_test,y_train,y_test,X_2_train,X_2_test = np.array([np.array(i) for i in features_train]),np.array([np.array(i) for i in features_test]),np.array(labels_train),np.array(labels_test),np.array([np.array(i) for i in statics_train]),np.array([np.array(i) for i in statics_test])
  X_train_reshaped,X_test_reshaped,X_2_train_reshaped,X_2_test_reshaped = X_train.reshape((X_train.shape[0], 1, interval, 1)),X_test.reshape((X_test.shape[0], 1, interval, 1)),X_2_train.reshape((X_2_train.shape[0], X_2_train.shape[1], 1)),X_2_test.reshape((X_2_test.shape[0], X_2_train.shape[1], 1))



  def create_model():

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

    return finalmodel

  finalmodel = create_model()
  finalmodel.load_weights("/content/training_1/1.ckpt")


  array22 = []

  def evaluate1(y_test_pred, y_test, y_train_pred, y_train):
      '''
      Sorry, haven't written a real doc string yet.
      '''
      r2_train = r2_score(y_train, y_train_pred)
      mae_train = mean_absolute_error(y_train, y_train_pred)
      mse_train = mean_squared_error(y_train, y_train_pred)

      r2_test = r2_score(y_test, y_test_pred)
      mae_test = mean_absolute_error(y_test, y_test_pred)
      mse_test = mean_squared_error(y_test, y_test_pred)

      average_mae_test1.append(mae_test)
      average_mse_test1.append(mse_test)
      average_r2_test1.append(r2_test)

      print('  R-squared Train: ' + str(r2_train))
      print('  R-squared Test: ' + str(r2_test))
      print('  MAE Train: ' + str(mae_train))
      print('  MAE Test: ' + str(mae_test))
      print('  MSE Train: ' + str(mse_train))
      print('  MSE Test: ' + str(mse_test))

      array22.append(r2_train)
      array22.append(r2_test)
      array22.append(mae_train)
      array22.append(mae_test)
      array22.append(mse_train)
      array22.append(mse_test)
    

  def evaluate2(y_test_pred, y_test, y_train_pred, y_train):
      '''
      Sorry, haven't written a real doc string yet.
      '''
      r2_train = r2_score(y_train, y_train_pred)
      mae_train = mean_absolute_error(y_train, y_train_pred)
      mse_train = mean_squared_error(y_train, y_train_pred)

      r2_test = r2_score(y_test, y_test_pred)
      mae_test = mean_absolute_error(y_test, y_test_pred)
      mse_test = mean_squared_error(y_test, y_test_pred)

      average_mae_test2.append(mae_test)
      average_mse_test2.append(mse_test)
      average_r2_test2.append(r2_test)

      print('  R-squared Train: ' + str(r2_train))
      print('  R-squared Test: ' + str(r2_test))
      print('  MAE Train: ' + str(mae_train))
      print('  MAE Test: ' + str(mae_test))
      print('  MSE Train: ' + str(mse_train))
      print('  MSE Test: ' + str(mse_test))

  def evaluate3(y_test_pred, y_test, y_train_pred, y_train):
      '''
      Sorry, haven't written a real doc string yet.
      '''
      r2_train = r2_score(y_train, y_train_pred)
      mae_train = mean_absolute_error(y_train, y_train_pred)
      mse_train = mean_squared_error(y_train, y_train_pred)

      r2_test = r2_score(y_test, y_test_pred)
      mae_test = mean_absolute_error(y_test, y_test_pred)
      mse_test = mean_squared_error(y_test, y_test_pred)

      average_mae_test3.append(mae_test)
      average_mse_test3.append(mse_test)
      average_r2_test3.append(r2_test)

      print('  R-squared Train: ' + str(r2_train))
      print('  R-squared Test: ' + str(r2_test))
      print('  MAE Train: ' + str(mae_train))
      print('  MAE Test: ' + str(mae_test))
      print('  MSE Train: ' + str(mse_train))
      print('  MSE Test: ' + str(mse_test))

      array22.append(r2_train)
      array22.append(r2_test)
      array22.append(mae_train)
  def evaluate4(y_test_pred, y_test, y_train_pred, y_train):
      '''
      Sorry, haven't written a real doc string yet.
      '''
      r2_train = r2_score(y_train, y_train_pred)
      mae_train = mean_absolute_error(y_train, y_train_pred)
      mse_train = mean_squared_error(y_train, y_train_pred)

      r2_test = r2_score(y_test, y_test_pred)
      mae_test = mean_absolute_error(y_test, y_test_pred)
      mse_test = mean_squared_error(y_test, y_test_pred)

      average_mae_test4.append(mae_test)
      average_mse_test4.append(mse_test)
      average_r2_test4.append(r2_test)

      print('  R-squared Train: ' + str(r2_train))
      print('  R-squared Test: ' + str(r2_test))
      print('  MAE Train: ' + str(mae_train))
      print('  MAE Test: ' + str(mae_test))
      print('  MSE Train: ' + str(mse_train))
      print('  MSE Test: ' + str(mse_test))

  def evaluate5(y_test_pred, y_test, y_train_pred, y_train):
      '''
      Sorry, haven't written a real doc string yet.
      '''
      r2_train = r2_score(y_train, y_train_pred)
      mae_train = mean_absolute_error(y_train, y_train_pred)
      mse_train = mean_squared_error(y_train, y_train_pred)

      r2_test = r2_score(y_test, y_test_pred)
      mae_test = mean_absolute_error(y_test, y_test_pred)
      mse_test = mean_squared_error(y_test, y_test_pred)

      average_mae_test5.append(mae_test)
      average_mse_test5.append(mse_test)
      average_r2_test5.append(r2_test)

      print('  R-squared Train: ' + str(r2_train))
      print('  R-squared Test: ' + str(r2_test))
      print('  MAE Train: ' + str(mae_train))
      print('  MAE Test: ' + str(mae_test))
      print('  MSE Train: ' + str(mse_train))
      print('  MSE Test: ' + str(mse_test))


  def evaluate6(y_test_pred, y_test, y_train_pred, y_train):
      '''
      Sorry, haven't written a real doc string yet.
      '''
      r2_train = r2_score(y_train, y_train_pred)
      mae_train = mean_absolute_error(y_train, y_train_pred)
      mse_train = mean_squared_error(y_train, y_train_pred)

      r2_test = r2_score(y_test, y_test_pred)
      mae_test = mean_absolute_error(y_test, y_test_pred)
      mse_test = mean_squared_error(y_test, y_test_pred)

      average_mae_test6.append(mae_test)
      average_mse_test6.append(mse_test)
      average_r2_test6.append(r2_test)

      print('  R-squared Train: ' + str(r2_train))
      print('  R-squared Test: ' + str(r2_test))
      print('  MAE Train: ' + str(mae_train))
      print('  MAE Test: ' + str(mae_test))
      print('  MSE Train: ' + str(mse_train))
      print('  MSE Test: ' + str(mse_test))

  def evaluate7(y_test_pred, y_test, y_train_pred, y_train):
      '''
      Sorry, haven't written a real doc string yet.
      '''
      r2_train = r2_score(y_train, y_train_pred)
      mae_train = mean_absolute_error(y_train, y_train_pred)
      mse_train = mean_squared_error(y_train, y_train_pred)

      r2_test = r2_score(y_test, y_test_pred)
      mae_test = mean_absolute_error(y_test, y_test_pred)
      mse_test = mean_squared_error(y_test, y_test_pred)

      average_mae_test7.append(mae_test)
      average_mse_test7.append(mse_test)
      average_r2_test7.append(r2_test)

      print('  R-squared Train: ' + str(r2_train))
      print('  R-squared Test: ' + str(r2_test))
      print('  MAE Train: ' + str(mae_train))
      print('  MAE Test: ' + str(mae_test))
      print('  MSE Train: ' + str(mse_train))
      print('  MSE Test: ' + str(mse_test))

      array22.append(r2_train)
      array22.append(r2_test)
      array22.append(mae_train)
      array22.append(mae_test)
      array22.append(mse_train)
      array22.append(mse_test)

  y_test_pred_day1 = [i[0] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]
  y_test_pred_day2 = [i[1] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]
  y_test_pred_day3 = [i[2] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]
  y_test_pred_day4 = [i[3] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]
  y_test_pred_day5 = [i[4] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]
  y_test_pred_day6 = [i[5] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]
  y_test_pred_day7 = [i[6] for i in finalmodel.predict([X_test_reshaped,X_2_test_reshaped])]

  y_test_pred_day1 = [round(i) for i in y_test_pred_day1]
  y_test_pred_day2 = [round(i) for i in y_test_pred_day2]
  y_test_pred_day3 = [round(i) for i in y_test_pred_day3]
  y_test_pred_day4 = [round(i) for i in y_test_pred_day4]
  y_test_pred_day5 = [round(i) for i in y_test_pred_day5]
  y_test_pred_day6 = [round(i) for i in y_test_pred_day6]
  y_test_pred_day7 = [round(i) for i in y_test_pred_day7]

  y_train_pred_day1 = [i[0] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])]
  y_train_pred_day2 = [i[1] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])]
  y_train_pred_day3 = [i[2] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])]
  y_train_pred_day4 = [i[3] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])]
  y_train_pred_day5 = [i[4] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])]
  y_train_pred_day6 = [i[5] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])]
  y_train_pred_day7 = [i[6] for i in finalmodel.predict([X_train_reshaped,X_2_train_reshaped])]

  y_train_pred_day1 = [round(i) for i in y_train_pred_day1]
  y_train_pred_day2 = [round(i) for i in y_train_pred_day2]
  y_train_pred_day3 = [round(i) for i in y_train_pred_day3]
  y_train_pred_day4 = [round(i) for i in y_train_pred_day4]
  y_train_pred_day5 = [round(i) for i in y_train_pred_day5]
  y_train_pred_day6 = [round(i) for i in y_train_pred_day6]
  y_train_pred_day7 = [round(i) for i in y_train_pred_day7]


    # Split y arrays into individual arrays for analysis
  y_test_day1 = [i[0] for i in y_test]
  y_test_day2 = [i[1] for i in y_test]
  y_test_day3 = [i[2] for i in y_test]
  y_test_day4 = [i[3] for i in y_test]
  y_test_day5 = [i[4] for i in y_test]
  y_test_day6 = [i[5] for i in y_test]
  y_test_day7 = [i[6] for i in y_test]   

  y_train_day1 = [i[0] for i in y_train]
  y_train_day2 = [i[1] for i in y_train]
  y_train_day3 = [i[2] for i in y_train]
  y_train_day4 = [i[3] for i in y_train]
  y_train_day5 = [i[4] for i in y_train]
  y_train_day6 = [i[5] for i in y_train]
  y_train_day7 = [i[6] for i in y_train]


  # Evaluation
  # Evaluation
  print('Day 1')
  evaluate1(y_test_pred_day1, y_test_day1, y_train_pred_day1, y_train_day1)
  print('')
  print('****************************************************')
  print('Day 2')
  evaluate2(y_test_pred_day2, y_test_day2, y_train_pred_day2, y_train_day2)
  print('')
  print('****************************************************')
  print('Day 3')
  evaluate3(y_test_pred_day3, y_test_day3, y_train_pred_day3, y_train_day3)
  print('')
  print('****************************************************')
  print('Day 4')
  evaluate4(y_test_pred_day4, y_test_day4, y_train_pred_day4, y_train_day4)
  print('')
  print('****************************************************')
  print('Day 5')
  evaluate5(y_test_pred_day5, y_test_day5, y_train_pred_day5, y_train_day5)
  print('')
  print('****************************************************')
  print('Day 6')
  evaluate6(y_test_pred_day6, y_test_day6, y_train_pred_day6, y_train_day6)
  print('')
  print('****************************************************')
  print('Day 7')
  evaluate7(y_test_pred_day7, y_test_day7, y_train_pred_day7, y_train_day7)
  print('')
  print('****************************************************')



  y_1_graph.append(y_test_pred_day1)
  y_3_graph.append(y_test_pred_day3)
  y_7_graph.append(y_test_pred_day7)


  y_1_graph_real.append(y_test_day1)

  z = z + 1

abs_df = pd.DataFrame()
abs1_df = pd.DataFrame()
abs2_df = pd.DataFrame()
o_df = pd.DataFrame()
o1_df = pd.DataFrame()
o2_df = pd.DataFrame()
import matplotlib.pyplot as plt

y1 = y_1_graph
y2 = y_3_graph
y3 = y_7_graph
y4 = y_1_graph_real

i = 1
x1 = []
while i < 70:
  x1.append(i)
  i = i + 1


#To switch back to three month zero predction 68->91 add 22 to the large numbers

abs_df['Actual Cases y'] = y4
abs_df['Days x'] = x1
abs_df['Actual Cases 1 day y'] = y1
abs_df['Days 1 x'] = x1 
abs1_df['Actual Cases 3 day y'] = y2[0:67]
abs1_df['Days 3 x'] = x1[2:69] 
abs2_df['Actual Cases 7 day y'] = y3[0:63]
abs2_df['Days 7 x'] = x1[6:69]



plt.plot(x1, y1, label = "1 day")

plt.plot(x1[2:69], y2[0:67], label = "3 day")

plt.plot(x1[6:69], y3[0:63], label = "7 day")
  
plt.plot(x1, y4, label = "Actual")
  
# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('Case Count')
# giving a title to my graph
plt.title('Actual vs Predicted [Custer, Montana]')
  
# show a legend on the plot
plt.legend()
  
# function to show the plot
plt.show()



  
plt.plot(x1[2:69], y2[0:67], label = "3 day")


  
plt.plot(x1, y4, label = "Actual")
  
# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('Case Count')
# giving a title to my graph
plt.title('Actual vs Predicted [Custer, Montana]')
  
# show a legend on the plot
plt.legend()
  
# function to show the plot
plt.show()

plt.plot(x1, y1, label = "1 day")
  



  
plt.plot(x1, y4, label = "Actual")
  
# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('Case Count')
# giving a title to my graph
plt.title('Actual vs Predicted [Custer, Montana]')
  
# show a legend on the plot
plt.legend()
  
# function to show the plot
plt.show()



plt.plot(x1[6:69], y3[0:63], label = "7 day")
  
plt.plot(x1, y4, label = "Actual")
  
# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('Case Count')
# giving a title to my graph
plt.title('Actual vs Predicted [Custer, Montana]')
  
# show a legend on the plot
plt.legend()
  
# function to show the plot
plt.show()




mse_test1 = mean_squared_error(y1, y4)
mse_test2 = mean_squared_error(y2[0:67], y4[2:69])
mse_test3 = mean_squared_error(y3[0:63], y4[6:69])
mse_test4 = mean_squared_error(y4[0:68], y4[1:69])
mse_test5 = mean_squared_error(y4[0:66], y4[3:69])
mse_test6 = mean_squared_error(y4[0:62], y4[7:69])

o_df['Actual Cases 1 day y'] = y4[1:69]
o_df['Days 1 x'] =  y4[0:68]
o1_df['Actual Cases 3 day y'] = y4[3:69]
o1_df['Days 3 x'] = y4[0:66] 
o2_df['Actual Cases 7 day y'] = y4[7:69]
o2_df['Days 7 x'] = y4[0:62]


mae_test1 = mean_absolute_error(y1, y4)
mae_test2 = mean_absolute_error(y2[0:67], y4[2:69])
mae_test3 = mean_absolute_error(y3[0:63], y4[6:69])
mae_test4 = mean_absolute_error(y4[0:68], y4[1:69])
mae_test5 = mean_absolute_error(y4[0:66], y4[3:69])
mae_test6 = mean_absolute_error(y4[0:62], y4[7:69])

print(mse_test1)
print(mse_test2)
print(mse_test3)
print(mse_test4)
print(mse_test5)
print(mse_test6)
print(mae_test1)
print(mae_test2)
print(mae_test3)
print(mae_test4)
print(mae_test5)
print(mae_test6)



plt.plot(x1[7:69], y4[0:62], label = "7 day")
  
plt.plot(x1, y4, label = "Actual")
  
# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('Case Count')
# giving a title to my graph
plt.title('Actual vs Predicted [Custer,Montana]')
  
# show a legend on the plot
plt.legend()
  
# function to show the plot
plt.show()


plt.plot(x1[3:69], y4[0:66], label = "3 day")
  
plt.plot(x1, y4, label = "Actual")
  
# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('Case Count')
# giving a title to my graph
plt.title('Actual vs Predicted [Custer, Montana]')
  
# show a legend on the plot
plt.legend()
  
# function to show the plot
plt.show()


plt.plot(x1[1:69], y4[0:68], label = "1 day")
  
plt.plot(x1, y4, label = "Actual")
  
# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('Case Count')
# giving a title to my graph
plt.title('Actual vs Predicted [Custer,Montana]')
  
# show a legend on the plot
plt.legend()
  
# function to show the plot
plt.show()



abs_df.to_excel(excel_writer="Predictions.xlsx")
abs1_df.to_excel(excel_writer="Predictions1.xlsx")
abs2_df.to_excel(excel_writer="Predictions2.xlsx")
o_df.to_excel(excel_writer="StepForwards.xlsx")
o1_df.to_excel(excel_writer="StepForwards1.xlsx")
o2_df.to_excel(excel_writer="StepForwards2.xlsx")


from google.colab import files
files.download("/content/Predictions.xlsx")
files.download("/content/Predictions1.xlsx")
files.download("/content/Predictions2.xlsx")
files.download("/content/StepForwards.xlsx")
files.download("/content/StepForwards1.xlsx")
files.download("/content/StepForwards2.xlsx")
