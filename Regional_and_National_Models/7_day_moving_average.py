import os
import tensorflow as tf
import pandas as pd
import numpy as np
from google.colab import files
from datetime import datetime
from datetime import timedelta
from datetime import date
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

interval = 14                                                         #Length of training data slices
gap = abs(date.today()-date(2021, 2, 28)).days                        #Last day of training
optimizer = keras.optimizers.Adam(learning_rate=0.0008)
LOSS = tf.keras.losses.Huber(delta=1.0)
epochs = 100


#Pre-Processing

csse_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
df = pd.concat([csse_df[csse_df['Province_State'] == 'Wyoming'],csse_df[csse_df['Province_State'] == 'North Dakota'],csse_df[csse_df['Province_State'] == 'Montana'],csse_df[csse_df['Province_State'] == 'South Dakota'],csse_df[csse_df['Province_State'] == 'Idaho']], ignore_index=True)
df['County'] = df['Admin2'].str.cat(df['Province_State'], sep=', ')
df = df.drop(columns=['UID','code3','iso2','iso3','FIPS','Country_Region','Lat','Long_','Combined_Key','Admin2','Province_State'])
headers_vals = list(df['County'])
headers_vals.insert(0,'Date') 
df = df.drop(columns=['County']).transpose().reset_index()
df.columns = headers_vals
df = df.set_index('Date') 
df_new_cases_only = pd.DataFrame()

#Convert Data into new cases by day

for i in list(df):
  array = np.array([second - first for first, second in zip(df[i], df[i][1:])])
  array[array<0] = 0
  df_new_cases_only[i] = array
df = df_new_cases_only.astype(float)
p = 0

#Convert data into seven day MA

for i in df:
  df[i] = df.iloc[:,p].rolling(window=7).mean()
  p = p + 1
df = df[7:len(df)].reset_index(drop=True)

mult = int(df.max().max()) 
df = df/(df.max().max())

day1mae, day3mae, day7mae, day1mse, day3mse, day7mse, day12mae, day32mae, day72mae, day12mse, day32mse, day72mse = ([] for i in range(12))


#########################################################################################################################

#Create Training and testing data

labels, features, counties, statics_train, statics_test, day1mae, day3mae, day7mae, day1mse, day3mse, day7mse, day12mae, day32mae, day72mae, day12mse, day32mse, day72mse = ([] for i in range(17))
for county in list(df):
    seq = list(df[county])
    for i in range(len(seq)-(interval+7)-gap):
        array = seq[i:i+(interval+7)]
        counties.append(county)
        features.append(np.array(array[:-7]))
        labels.append(np.array([array[-7],array[-6],array[-5],array[-4],array[-3],array[-2],array[-1]]))
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle = True)
X_train, X_test, y_train, y_test = np.array([np.array(i) for i in X_train]), np.array([np.array(i) for i in X_test]), np.array(y_train), np.array(y_test)
X_train = np.array([np.array(i) for i in X_train])

X_train_reshaped, X_test_reshaped = X_train.reshape((X_train.shape[0], 1, interval, 1)), X_test.reshape((X_test.shape[0], 1, interval, 1))

#Network Architecture

def Learner():
  model = Sequential()
  model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, interval, 1)))
  model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
  model.add(TimeDistributed(Flatten()))
  model.add(LSTM(20, activation='relu'))
  model.add(Dense(10))
  model.add(Dense(10))
  model.add(Dense(5))
  model.add(Dense(5))
  model.add(Dense(3))
  model.add(Dense(7))
  model.compile(optimizer=optimizer, loss=LOSS)
  return model

#Evaluate

globals()["count"] = 0  
def evaluate(y_test_pred, y_test):
  globals()["count"] = globals()["count"] + 1
  r2_test = r2_score(y_test, y_test_pred)
  mae_test = mean_absolute_error(y_test, y_test_pred)
  mse_test = mean_squared_error(y_test, y_test_pred)
  print('  R-squared Test: ' + str(r2_test))
  print('  MAE Test: ' + str(mae_test))
  print('  MSE Test: ' + str(mse_test))
  print(count)
  if count == 1:
    day1mae.append(mae_test)
    day1mse.append(mse_test)
  if count == 3:
    day3mae.append(mae_test)
    day3mse.append(mse_test)
  if count == 7:
    day7mae.append(mae_test)
    day7mse.append(mse_test)
  if count == 8:
    day12mae.append(mae_test)
    day12mse.append(mse_test)
  if count == 10:
    day32mae.append(mae_test)
    day32mse.append(mse_test)
  if count == 14:
    day72mae.append(mae_test)
    day72mse.append(mse_test)

finalmodel = Learner()
finalmodel.fit(x=[X_train_reshaped], y=y_train, epochs=epochs, verbose=1)

#Predictions

y_test_pred_day1,y_test_pred_day2,y_test_pred_day3,y_test_pred_day4,y_test_pred_day5,y_test_pred_day6,y_test_pred_day7 = np.array([i[0] for i in finalmodel.predict([X_test_reshaped])])*mult,np.array([i[1] for i in finalmodel.predict([X_test_reshaped])])*mult,np.array([i[2] for i in finalmodel.predict([X_test_reshaped])])*mult,np.array([i[3] for i in finalmodel.predict([X_test_reshaped])])*mult,np.array([i[4] for i in finalmodel.predict([X_test_reshaped])])*mult,np.array([i[5] for i in finalmodel.predict([X_test_reshaped])])*mult,np.array([i[6] for i in finalmodel.predict([X_test_reshaped])])*mult
y_test_pred_day1,y_test_pred_day2,y_test_pred_day3,y_test_pred_day4,y_test_pred_day5,y_test_pred_day6,y_test_pred_day7 = [round(i) for i in y_test_pred_day1], [round(i) for i in y_test_pred_day2],[round(i) for i in y_test_pred_day3],[round(i) for i in y_test_pred_day4],[round(i) for i in y_test_pred_day5],[round(i) for i in y_test_pred_day6],[round(i) for i in y_test_pred_day7]
y_train_pred_day1,y_train_pred_day2,y_train_pred_day3,y_train_pred_day4,y_train_pred_day5,y_train_pred_day6,y_train_pred_day7 = [i[0] for i in finalmodel.predict([X_train_reshaped])],[i[1] for i in finalmodel.predict([X_train_reshaped])],[i[2] for i in finalmodel.predict([X_train_reshaped])],[i[3] for i in finalmodel.predict([X_train_reshaped])],[i[4] for i in finalmodel.predict([X_train_reshaped])],[i[5] for i in finalmodel.predict([X_train_reshaped])],[i[6] for i in finalmodel.predict([X_train_reshaped])]
y_train_pred_day1,y_train_pred_day2,y_train_pred_day3,y_train_pred_day4,y_train_pred_day5,y_train_pred_day6,y_train_pred_day7 = [round(i) for i in y_train_pred_day1],[round(i) for i in y_train_pred_day2],[round(i) for i in y_train_pred_day3],[round(i) for i in y_train_pred_day4],[round(i) for i in y_train_pred_day5],[round(i) for i in y_train_pred_day6],[round(i) for i in y_train_pred_day7]
y_test_day1,y_test_day2,y_test_day3,y_test_day4,y_test_day5,y_test_day6,y_test_day7 = np.array([i[0] for i in y_test])*mult,np.array([i[1] for i in y_test])*mult,np.array([i[2] for i in y_test])*mult,np.array([i[3] for i in y_test])*mult,np.array([i[4] for i in y_test])*mult,np.array([i[5] for i in y_test])*mult,np.array([i[6] for i in y_test])*mult
y_train_day1,y_train_day2,y_train_day3,y_train_day4,y_train_day5,y_train_day6,y_train_day7 = [i[0] for i in y_train],[i[1] for i in y_train],[i[2] for i in y_train],[i[3] for i in y_train],[i[4] for i in y_train],[i[5] for i in y_train],[i[6] for i in y_train]

print('*****************************************************************************')
print(y_test_pred_day1)
print('*****************************************************************************')
print('Day 1')
evaluate(y_test_pred_day1, y_test_day1)
print('Day 2')
evaluate(y_test_pred_day2, y_test_day2)
print('Day 3')
evaluate(y_test_pred_day3, y_test_day3)
print('Day 4')
evaluate(y_test_pred_day4, y_test_day4)
print('Day 5')
evaluate(y_test_pred_day5, y_test_day5)
print('Day 6')
evaluate(y_test_pred_day6, y_test_day6)
print('Day 7')
evaluate(y_test_pred_day7, y_test_day7)


#Download Results

df22 = pd.DataFrame()
df22['a1'], df22['a3'], df22['a7'], df22['s1'], df22['s3'], df22['s7']  = day1mae, day3mae, day7mae, day1mse, day3mse, day7mse
df22.to_excel(excel_writer="7ma.xlsx")
files.download("/content/7ma.xlsx")
