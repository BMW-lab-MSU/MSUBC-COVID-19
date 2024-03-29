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
epochs = 50
gap = abs(date.today()-date(2021, 2, 28)).days
optimizer = keras.optimizers.Adam(learning_rate=0.0008)
LOSS = "huber_loss"
interval = 14

featuresSmall = []
labelsSmall = []
featuresBig = []
labelsBig = []
array = []
day1mae = []
day3mae = []
day7mae = []
day1mse = []
day3mse = []
day7mse = []

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
for i in list(df):
  array = np.array([second - first for first, second in zip(df[i], df[i][1:])])
  array[array<0] = 0
  df_new_cases_only[i] = array
df = df_new_cases_only.astype(float)

labels, features, day1mae, day3mae, day7mae, day1mse, day3mse, day7mse, day12mae, day32mae, day72mae, day12mse, day32mse, day72mse = ([] for i in range(14))
for county in list(df):
    seq = list(df[county])
    for i in range(len(seq)-(interval)+2-gap):
        array = seq[i:i+(interval+7)]
        featuresSmall.append(np.array(array[:-7]))
        labelsSmall.append(np.array([array[-7],array[-6],array[-5],array[-4],array[-3],array[-2],array[-1]]))




csse_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
# (put these all in one line when you get a chance!) df = pd.concat([csse_df[csse_df['Province_State'] == 'Wyoming'],csse_df[csse_df['Province_State'] == 'North Dakota']])



wy_df = csse_df[csse_df['Province_State'] == 'Wyoming']
nd_df = csse_df[csse_df['Province_State'] == 'North Dakota']
mt_df = csse_df[csse_df['Province_State'] == 'Montana']
sd_df = csse_df[csse_df['Province_State'] == 'South Dakota']
id_df = csse_df[csse_df['Province_State'] == 'Idaho']
wa_df = csse_df[csse_df['Province_State'] == 'Washington']
or_df = csse_df[csse_df['Province_State'] == 'Oregon']
nv_df = csse_df[csse_df['Province_State'] == 'Nevada']
ut_df = csse_df[csse_df['Province_State'] == 'Utah']
co_df = csse_df[csse_df['Province_State'] == 'Colorado']
ne_df = csse_df[csse_df['Province_State'] == 'Nebraska']
ia_df = csse_df[csse_df['Province_State'] == 'Iowa']
mn_df = csse_df[csse_df['Province_State'] == 'Minnesota']
ca_df = csse_df[csse_df['Province_State'] == 'California']
az_df = csse_df[csse_df['Province_State'] == 'Arizona']
nm_df = csse_df[csse_df['Province_State'] == 'New Mexico']
tx_df = csse_df[csse_df['Province_State'] == 'Texas']
ok_df = csse_df[csse_df['Province_State'] == 'Oklahoma']
ks_df = csse_df[csse_df['Province_State'] == 'Kansas']
mo_df = csse_df[csse_df['Province_State'] == 'Missouri']
il_df = csse_df[csse_df['Province_State'] == 'Illinois']
wi_df = csse_df[csse_df['Province_State'] == 'Wisconsin']
mi_df = csse_df[csse_df['Province_State'] == 'Michigan']
in_df = csse_df[csse_df['Province_State'] == 'Indiana']
ky_df = csse_df[csse_df['Province_State'] == 'Kentucky']
tn_df = csse_df[csse_df['Province_State'] == 'Tennessee']
ms_df = csse_df[csse_df['Province_State'] == 'Mississippi']
ar_df = csse_df[csse_df['Province_State'] == 'Arkansas']
la_df = csse_df[csse_df['Province_State'] == 'Louisiana']
al_df = csse_df[csse_df['Province_State'] == 'Alabama']
ga_df = csse_df[csse_df['Province_State'] == 'Georgia']
fl_df = csse_df[csse_df['Province_State'] == 'Florida']
sc_df = csse_df[csse_df['Province_State'] == 'South Carolina']
nc_df = csse_df[csse_df['Province_State'] == 'North Carolina']
va_df = csse_df[csse_df['Province_State'] == 'Virginia']
wv_df = csse_df[csse_df['Province_State'] == 'West Virginia']
oh_df = csse_df[csse_df['Province_State'] == 'Ohio']
pa_df = csse_df[csse_df['Province_State'] == 'Pennsylvania']
md_df = csse_df[csse_df['Province_State'] == 'Maryland']
de_df = csse_df[csse_df['Province_State'] == 'Delaware']
nj_df = csse_df[csse_df['Province_State'] == 'New Jersey']
ny_df = csse_df[csse_df['Province_State'] == 'New York']
vt_df = csse_df[csse_df['Province_State'] == 'Vermont']
ct_df = csse_df[csse_df['Province_State'] == 'Connecticut']
ri_df = csse_df[csse_df['Province_State'] == 'Rhode Island']
ma_df = csse_df[csse_df['Province_State'] == 'Massachusetts']
nh_df = csse_df[csse_df['Province_State'] == 'New Hampshire']
me_df = csse_df[csse_df['Province_State'] == 'Maine']

df = pd.concat([wy_df,nd_df,mt_df,sd_df,id_df,wa_df,or_df,nv_df,ut_df,co_df,ne_df,ia_df,mn_df,ca_df,az_df,nm_df,tx_df,ok_df,la_df,ar_df,mo_df,ks_df,wi_df,mi_df,in_df,il_df,ky_df,tn_df,ms_df,al_df,ga_df,fl_df,sc_df,nc_df,va_df,wv_df,oh_df,pa_df,md_df,de_df,nj_df,ny_df,ct_df,ri_df,ma_df,vt_df,nh_df,me_df], ignore_index=True)

df['Location'] = df['Admin2'].str.cat(df['Province_State'], sep=', ')

df = df.drop(columns=['UID','code3','iso2','iso3','FIPS','Country_Region','Lat','Long_','Combined_Key',
                      #'2/1/20','2/2/20','2/3/20','2/4/20','2/5/20',
                      #'2/6/20','2/7/20','2/8/20','2/9/20','2/10/20','2/11/20','2/12/20','2/13/20','2/14/20','2/15/20','2/16/20','2/17/20','2/18/20',
                      #'2/19/20','2/20/20','2/21/20','2/22/20','2/23/20','2/24/20','2/25/20','2/26/20','2/27/20','2/28/20','2/29/20','3/1/20','3/2/20',
                      #'3/3/20','3/4/20','3/5/20','3/6/20','3/7/20','3/8/20','3/9/20','3/10/20',
                      'Admin2','Province_State'])


headers_vals = list(df['Location'])

headers_vals.insert(0,'Date')

df = df.drop(columns=['Location'])

df = df.transpose().reset_index()

df.columns = headers_vals

df = df.set_index('Date')

df = df.drop(columns=['Out of WY, Wyoming','Unassigned, Wyoming','Out of ND, North Dakota','Unassigned, North Dakota','Out of MT, Montana',
                      'Unassigned, Montana','Out of SD, South Dakota','Unassigned, South Dakota','Out of ID, Idaho','Unassigned, Idaho','Out of WA, Washington','Unassigned, Washington','Out of OR, Oregon','Unassigned, Oregon','Out of NV, Nevada',
                      'Unassigned, Nevada','Out of UT, Utah','Unassigned, Utah','Out of CO, Colorado','Unassigned, Colorado',
                      'Out of NE, Nebraska', 'Unassigned, Nebraska','Out of IA, Iowa','Unassigned, Iowa','Out of MN, Minnesota', 
                      'Unassigned, Minnesota','Out of CA, California','Unassigned, California','Out of AZ, Arizona','Unassigned, Arizona','Out of NM, New Mexico',
                      'Unassigned, New Mexico','Out of TX, Texas','Unassigned, Texas','Out of OK, Oklahoma','Unassigned, Oklahoma',
                      'Out of LA, Louisiana','Unassigned, Louisiana','Out of AR, Arkansas','Unassigned, Arkansas','Out of MO, Missouri',
                      'Unassigned, Missouri','Out of KS, Kansas','Unassigned, Kansas',
                      'Out of WI, Wisconsin', 'Unassigned, Wisconsin','Out of MI, Michigan','Unassigned, Michigan','Out of IN, Indiana', 
                      'Unassigned, Indiana','Out of IL, Illinois','Unassigned, Illinois','Out of KY, Kentucky','Unassigned, Kentucky','Out of TN, Tennessee',
                      'Unassigned, Tennessee','Out of MS, Mississippi','Unassigned, Mississippi','Out of AL, Alabama','Unassigned, Alabama',
                      'Out of GA, Georgia','Unassigned, Georgia','Out of FL, Florida','Unassigned, Florida','Out of SC, South Carolina',
                      'Unassigned, South Carolina','Out of NC, North Carolina','Unassigned, North Carolina','Out of VA, Virginia','Unassigned, Virginia',
                      'Out of WV, West Virginia', 'Unassigned, West Virginia','Out of OH, Ohio','Unassigned, Ohio','Out of PA, Pennsylvania', 
                      'Unassigned, Pennsylvania','Out of MD, Maryland','Unassigned, Maryland','Out of DE, Delaware','Unassigned, Delaware','Out of NJ, New Jersey',
                      'Unassigned, New Jersey','Out of NY, New York','Unassigned, New York','Out of VT, Vermont','Unassigned, Vermont',
                      'Out of CT, Connecticut','Unassigned, Connecticut','Out of RI, Rhode Island','Unassigned, Rhode Island','Out of MA, Massachusetts',
                      'Unassigned, Massachusetts','Out of NH, New Hampshire','Unassigned, New Hampshire','Out of ME, Maine','Unassigned, Maine','Southeast Utah, Utah','Southwest Utah, Utah',
                      'Franklin City, Virginia','Weber-Morgan, Utah','Roanoke City, Virginia','Dona Ana, New Mexico','Central Utah, Utah',
                      'Kansas City, Missouri','St. Louis City, Missouri','James City, Virginia','Michigan Department of Corrections (MDOC), Michigan',
                      'TriCounty, Utah','Dukes and Nantucket, Massachusetts','Charles City, Virginia','Carson City, Nevada',
                      'Federal Correctional Institution (FCI), Michigan','Baltimore City, Maryland','Fairfax City, Virginia','Richmond City, Virginia',
                      'Bear River, Utah'])

#df = df.rename(columns={'Weber-Morgan, Utah':'Weber, Utah', 'Roanoke City, Virginia':'Roanoke, Virginia','Dona Ana, New Mexico':'Doña Ana, New Mexico'})


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
df_new_cases_only = df_new_cases_only.astype(float)
df = df_new_cases_only

d0 = date(2021, 2, 28)
d1 = date.today()

gap = abs(d0-d1)
gap = gap.days


for county in list(df):
    seq = list(df[county])

    
    # Create multiple arrays from each county
    for i in range(len(seq)-(interval)+2-gap):
        array = seq[i:i+(interval+7)]
        x_array = array[:-7]
        y_array = np.array([array[-7],array[-6],array[-5],array[-4],array[-3],array[-2],array[-1]])
        featuresBig.append(x_array)
        labelsBig.append(y_array)

X_train2, X_test, y_train2, y_test = train_test_split(featuresSmall, labelsSmall, test_size=0.2, shuffle = True)
X_train, X_test2, y_train, y_test2 = train_test_split(featuresBig, labelsBig, test_size=0.2, shuffle = True)
X_train, X_test, y_train, y_test = np.array([np.array(i) for i in X_train]), np.array([np.array(i) for i in X_test]), np.array(y_train), np.array(y_test)

X_train_reshaped = X_train.reshape((X_train.shape[0], 1, interval, 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, interval, 1))

# Initialize and Scaffold the Model
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

model.fit(X_train_reshaped, y_train, epochs=epochs, verbose=1)

globals()["count"] = 0  
def evaluate(y_test_pred, y_test, y_train_pred, y_train):
  globals()["count"] = globals()["count"] + 1
  r2_train = r2_score(y_train, y_train_pred)
  mae_train = mean_absolute_error(y_train, y_train_pred)
  mse_train = mean_squared_error(y_train, y_train_pred)
  r2_test = r2_score(y_test, y_test_pred)
  mae_test = mean_absolute_error(y_test, y_test_pred)
  mse_test = mean_squared_error(y_test, y_test_pred)
  print('  R-squared Train: ' + str(r2_train))
  print('  R-squared Test: ' + str(r2_test))
  print('  MAE Train: ' + str(mae_train))
  print('  MAE Test: ' + str(mae_test))
  print('  MSE Train: ' + str(mse_train))
  print('  MSE Test: ' + str(mse_test))
  if count == 1:
    day1mae.append(mae_test)
    day1mse.append(mse_test)
  if count == 3:
    day3mae.append(mae_test)
    day3mse.append(mse_test)
  if count == 7:
    day7mae.append(mae_test)
    day7mse.append(mse_test)

y_test_pred_day1,y_test_pred_day2,y_test_pred_day3,y_test_pred_day4,y_test_pred_day5,y_test_pred_day6,y_test_pred_day7 = np.array([i[0] for i in model.predict([X_test_reshaped])]),np.array([i[1] for i in model.predict([X_test_reshaped])]),np.array([i[2] for i in model.predict([X_test_reshaped])]),np.array([i[3] for i in model.predict([X_test_reshaped])]),np.array([i[4] for i in model.predict([X_test_reshaped])]),np.array([i[5] for i in model.predict([X_test_reshaped])]),np.array([i[6] for i in model.predict([X_test_reshaped])])
y_test_pred_day1,y_test_pred_day2,y_test_pred_day3,y_test_pred_day4,y_test_pred_day5,y_test_pred_day6,y_test_pred_day7 = [round(i) for i in y_test_pred_day1], [round(i) for i in y_test_pred_day2],[round(i) for i in y_test_pred_day3],[round(i) for i in y_test_pred_day4],[round(i) for i in y_test_pred_day5],[round(i) for i in y_test_pred_day6],[round(i) for i in y_test_pred_day7]
y_train_pred_day1,y_train_pred_day2,y_train_pred_day3,y_train_pred_day4,y_train_pred_day5,y_train_pred_day6,y_train_pred_day7 = [i[0] for i in model.predict([X_train_reshaped])],[i[1] for i in model.predict([X_train_reshaped])],[i[2] for i in model.predict([X_train_reshaped])],[i[3] for i in model.predict([X_train_reshaped])],[i[4] for i in model.predict([X_train_reshaped])],[i[5] for i in model.predict([X_train_reshaped])],[i[6] for i in model.predict([X_train_reshaped])]
y_train_pred_day1,y_train_pred_day2,y_train_pred_day3,y_train_pred_day4,y_train_pred_day5,y_train_pred_day6,y_train_pred_day7 = [round(i) for i in y_train_pred_day1],[round(i) for i in y_train_pred_day2],[round(i) for i in y_train_pred_day3],[round(i) for i in y_train_pred_day4],[round(i) for i in y_train_pred_day5],[round(i) for i in y_train_pred_day6],[round(i) for i in y_train_pred_day7]
y_test_day1,y_test_day2,y_test_day3,y_test_day4,y_test_day5,y_test_day6,y_test_day7 = np.array([i[0] for i in y_test]),np.array([i[1] for i in y_test]),np.array([i[2] for i in y_test]),np.array([i[3] for i in y_test]),np.array([i[4] for i in y_test]),np.array([i[5] for i in y_test]),np.array([i[6] for i in y_test])
y_train_day1,y_train_day2,y_train_day3,y_train_day4,y_train_day5,y_train_day6,y_train_day7 = [i[0] for i in y_train],[i[1] for i in y_train],[i[2] for i in y_train],[i[3] for i in y_train],[i[4] for i in y_train],[i[5] for i in y_train],[i[6] for i in y_train]

print('Day 1')
evaluate(y_test_pred_day1, y_test_day1, y_train_pred_day1, y_train_day1)
print('Day 2')
evaluate(y_test_pred_day2, y_test_day2, y_train_pred_day2, y_train_day2)
print('Day 3')
evaluate(y_test_pred_day3, y_test_day3, y_train_pred_day3, y_train_day3)
print('Day 4')
evaluate(y_test_pred_day4, y_test_day4, y_train_pred_day4, y_train_day4)
print('Day 5')
evaluate(y_test_pred_day5, y_test_day5, y_train_pred_day5, y_train_day5)
print('Day 6')
evaluate(y_test_pred_day6, y_test_day6, y_train_pred_day6, y_train_day6)
print('Day 7')
evaluate(y_test_pred_day7, y_test_day7, y_train_pred_day7, y_train_day7)

df22 = pd.DataFrame()
df22['a1'], df22['a3'], df22['a7'], df22['s1'], df22['s3'], df22['s7']  = day1mae, day3mae, day7mae, day1mse, day3mse, day7mse
df22.to_excel(excel_writer="train50.xlsx")
files.download("/content/train50.xlsx")
