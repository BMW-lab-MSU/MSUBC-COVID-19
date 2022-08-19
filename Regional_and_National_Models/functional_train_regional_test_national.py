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
epochs = 150
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

##############################################################################################

'''
Importing both the static and dynamic data.  REGIONAL
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

day1mae, day3mae, day7mae, day1mse, day3mse, day7mse, day12mae, day32mae, day72mae, day12mse, day32mse, day72mse = ([] for i in range(12))
labelssmall, featuressmall, statics_train,statics_test, countiessmall = ([] for i in range(5))
for county in list(df):
    seq = list(df[county])
    for i in range(len(seq)-(interval)+2-gap):
        countiessmall.append(county)
        array = seq[i:i+(interval+7)]
        featuressmall.append(np.array(array[:-7]))
        labelssmall.append(np.array([array[-7],array[-6],array[-5],array[-4],array[-3],array[-2],array[-1]]))

########################################################################################################

'''
IMPORT NATIONAL
'''
array = []
county_dict = pd.read_csv('https://github.com/BMW-lab-MSU/MSUBC-COVID-19/blob/08b66cc8bd1692a3f6e2bd0ca96d61686bc41520/County_data_folder/National_data.csv?raw=true')
for i in county_dict['State']:
  county_dict['State'] = county_dict['State'].replace('AL','Alabama')
  county_dict['State'] = county_dict['State'].replace('AK','Alaska')
  county_dict['State'] = county_dict['State'].replace('AZ','Arizona')
  county_dict['State'] = county_dict['State'].replace('AR','Arkansas')
  county_dict['State'] = county_dict['State'].replace('CA','California')
  county_dict['State'] = county_dict['State'].replace('CO','Colorado')
  county_dict['State'] = county_dict['State'].replace('CT','Connecticut')
  county_dict['State'] = county_dict['State'].replace('DE','Delaware')
  county_dict['State'] = county_dict['State'].replace('FL','Florida')
  county_dict['State'] = county_dict['State'].replace('GA','Georgia')
  county_dict['State'] = county_dict['State'].replace('HI','Hawaii')
  county_dict['State'] = county_dict['State'].replace('ID','Idaho')
  county_dict['State'] = county_dict['State'].replace('IL','Illinois')
  county_dict['State'] = county_dict['State'].replace('IN','Indiana')
  county_dict['State'] = county_dict['State'].replace('IA','Iowa')
  county_dict['State'] = county_dict['State'].replace('KS','Kansas')
  county_dict['State'] = county_dict['State'].replace('KY','Kentucky')
  county_dict['State'] = county_dict['State'].replace('LA','Louisiana')
  county_dict['State'] = county_dict['State'].replace('ME','Maine')
  county_dict['State'] = county_dict['State'].replace('MD','Maryland')
  county_dict['State'] = county_dict['State'].replace('MA','Massachusetts')
  county_dict['State'] = county_dict['State'].replace('MI','Michigan')
  county_dict['State'] = county_dict['State'].replace('MN','Minnesota')
  county_dict['State'] = county_dict['State'].replace('MS','Mississippi')
  county_dict['State'] = county_dict['State'].replace('MO','Missouri')
  county_dict['State'] = county_dict['State'].replace('NE','Nebraska')
  county_dict['State'] = county_dict['State'].replace('NV','Nevada')
  county_dict['State'] = county_dict['State'].replace('NH','New Hampshire')
  county_dict['State'] = county_dict['State'].replace('NJ','New Jersey')
  county_dict['State'] = county_dict['State'].replace('NM','New Mexico')
  county_dict['State'] = county_dict['State'].replace('NY','New York')
  county_dict['State'] = county_dict['State'].replace('NC','North Carolina')
  county_dict['State'] = county_dict['State'].replace('ND','North Dakota')
  county_dict['State'] = county_dict['State'].replace('OH','Ohio')
  county_dict['State'] = county_dict['State'].replace('OK','Oklahoma')
  county_dict['State'] = county_dict['State'].replace('OR','Oregon')
  county_dict['State'] = county_dict['State'].replace('PA','Pennsylvania')
  county_dict['State'] = county_dict['State'].replace('RI','Rhode Island')
  county_dict['State'] = county_dict['State'].replace('SC','South Carolina')
  county_dict['State'] = county_dict['State'].replace('SD','South Dakota')
  county_dict['State'] = county_dict['State'].replace('TN','Tennessee')
  county_dict['State'] = county_dict['State'].replace('TX','Texas')
  county_dict['State'] = county_dict['State'].replace('UT','Utah')
  county_dict['State'] = county_dict['State'].replace('VT','Vermont')
  county_dict['State'] = county_dict['State'].replace('VA','Virginia')
  county_dict['State'] = county_dict['State'].replace('WA','Washington')
  county_dict['State'] = county_dict['State'].replace('WV','West Virginia')
  county_dict['State'] = county_dict['State'].replace('WI','Wisconsin')
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
county_dict = county_dict.drop(columns=['FIPStxt', 'State', 'Area_Name','Combo1','Unnamed: 0'])
county_dict = county_dict.set_index('Combo').T.to_dict('list')

csse_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
wy_df,nd_df,mt_df,sd_df,id_df,wa_df,or_df,nv_df,ut_df,co_df,ne_df,ia_df,mn_df,ca_df,az_df,nm_df = csse_df[csse_df['Province_State'] == 'Wyoming'], csse_df[csse_df['Province_State'] == 'North Dakota'],csse_df[csse_df['Province_State'] == 'Montana'],csse_df[csse_df['Province_State'] == 'South Dakota'],csse_df[csse_df['Province_State'] == 'Idaho'],csse_df[csse_df['Province_State'] == 'Washington'],csse_df[csse_df['Province_State'] == 'Oregon'],csse_df[csse_df['Province_State'] == 'Nevada'],csse_df[csse_df['Province_State'] == 'Utah'],csse_df[csse_df['Province_State'] == 'Colorado'],csse_df[csse_df['Province_State'] == 'Nebraska'],csse_df[csse_df['Province_State'] == 'Iowa'],csse_df[csse_df['Province_State'] == 'Minnesota'],csse_df[csse_df['Province_State'] == 'California'],csse_df[csse_df['Province_State'] == 'Arizona'],csse_df[csse_df['Province_State'] == 'New Mexico']

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
df = df.drop(columns=['UID','code3','iso2','iso3','FIPS','Country_Region','Lat','Long_','Combined_Key','Admin2','Province_State'])
headers_vals = list(df['Location'])
headers_vals.insert(0,'Date')
df = df.drop(columns=['Location'])
df = df.transpose().reset_index()
df.columns = headers_vals
df = df.set_index('Date')
df = df.drop(columns=['Out of WY, Wyoming','Unassigned, Wyoming','Out of ND, North Dakota','Unassigned, North Dakota','Out of MT, Montana','Unassigned, Montana','Out of SD, South Dakota','Unassigned, South Dakota','Out of ID, Idaho','Unassigned, Idaho','Out of WA, Washington','Unassigned, Washington','Out of OR, Oregon','Unassigned, Oregon','Out of NV, Nevada','Unassigned, Nevada','Out of UT, Utah','Unassigned, Utah','Out of CO, Colorado','Unassigned, Colorado','Out of NE, Nebraska', 'Unassigned, Nebraska','Out of IA, Iowa','Unassigned, Iowa','Out of MN, Minnesota', 'Unassigned, Minnesota','Out of CA, California','Unassigned, California','Out of AZ, Arizona','Unassigned, Arizona','Out of NM, New Mexico','Unassigned, New Mexico','Out of TX, Texas','Unassigned, Texas','Out of OK, Oklahoma','Unassigned, Oklahoma','Out of LA, Louisiana','Unassigned, Louisiana','Out of AR, Arkansas','Unassigned, Arkansas','Out of MO, Missouri','Unassigned, Missouri','Out of KS, Kansas','Unassigned, Kansas','Out of WI, Wisconsin', 'Unassigned, Wisconsin','Out of MI, Michigan','Unassigned, Michigan','Out of IN, Indiana', 'Unassigned, Indiana','Out of IL, Illinois','Unassigned, Illinois','Out of KY, Kentucky','Unassigned, Kentucky','Out of TN, Tennessee','Unassigned, Tennessee','Out of MS, Mississippi','Unassigned, Mississippi','Out of AL, Alabama','Unassigned, Alabama','Out of GA, Georgia','Unassigned, Georgia','Out of FL, Florida','Unassigned, Florida','Out of SC, South Carolina','Unassigned, South Carolina','Out of NC, North Carolina','Unassigned, North Carolina','Out of VA, Virginia','Unassigned, Virginia','Out of WV, West Virginia', 'Unassigned, West Virginia','Out of OH, Ohio','Unassigned, Ohio','Out of PA, Pennsylvania', 'Unassigned, Pennsylvania','Out of MD, Maryland','Unassigned, Maryland','Out of DE, Delaware','Unassigned, Delaware','Out of NJ, New Jersey','Unassigned, New Jersey','Out of NY, New York','Unassigned, New York','Out of VT, Vermont','Unassigned, Vermont','Out of CT, Connecticut','Unassigned, Connecticut','Out of RI, Rhode Island','Unassigned, Rhode Island','Out of MA, Massachusetts','Unassigned, Massachusetts','Out of NH, New Hampshire','Unassigned, New Hampshire','Out of ME, Maine','Unassigned, Maine','Southeast Utah, Utah','Southwest Utah, Utah','Franklin City, Virginia','Weber-Morgan, Utah','Roanoke City, Virginia','Dona Ana, New Mexico','Central Utah, Utah','Kansas City, Missouri','St. Louis City, Missouri','James City, Virginia','Michigan Department of Corrections (MDOC), Michigan','TriCounty, Utah','Dukes and Nantucket, Massachusetts','Charles City, Virginia','Carson City, Nevada','Federal Correctional Institution (FCI), Michigan','Baltimore City, Maryland','Fairfax City, Virginia','Richmond City, Virginia','Bear River, Utah'])
df_new_cases_only = pd.DataFrame()




##############################################################################################################################################################################################################################


#Convert to new case by day
for i in list(df):
  array = np.array([second - first for first, second in zip(df[i], df[i][1:])])
  array[array<0] = 0
  df_new_cases_only[i] = array
df = df_new_cases_only.astype(float)
#county_dict = county_dict.set_index('County').T.to_dict('list') #Statics Data

labelsbig, featuresbig, statics_train,statics_test, countiesbig = ([] for i in range(5))
for county in list(df):
    seq = list(df[county])
    for i in range(len(seq)-(interval)+2-gap):
        countiesbig.append(county)
        array = seq[i:i+(interval+7)]
        featuresbig.append(np.array(array[:-7]))
        labelsbig.append(np.array([array[-7],array[-6],array[-5],array[-4],array[-3],array[-2],array[-1]]))
 
X_train, X_test2, y_train, y_test2,X_2_train,X_2_test2  = train_test_split(featuressmall, labelssmall,countiessmall, test_size=0.2, shuffle = True)
X_train2, X_test, y_train2, y_test,X_2_train2,X_2_test  = train_test_split(featuresbig, labelsbig,countiesbig, test_size=0.2, shuffle = True)


for i in X_2_train:
    statics_train.append(np.array(county_dict[i]))
for i in X_2_test:
    statics_test.append(np.array(county_dict[i]))

X_train, X_test, y_train, y_test,X_2_train,X_2_test = np.array([np.array(i) for i in X_train]), np.array([np.array(i) for i in X_test]), np.array(y_train), np.array(y_test),np.array([np.array(i) for i in statics_train]),np.array([np.array(i) for i in statics_test])

X_train_reshaped = X_train.reshape((X_train.shape[0], 1, interval, 1))
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
  
df22   = pd.DataFrame()
df22['a1'],df22['a3'],df22['a7'],df22['s1'],df22['s3'],df22['s7'] = day1mae,day3mae,day7mae,day1mse,day3mse,day7mse
df22.to_excel(excel_writer="regnatstats.xlsx")

from google.colab import files
files.download("/content/regnatstats.xlsx")
