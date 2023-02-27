import pandas as pd
stations = [18722, 18326, 18300, 15568, 18956, 18300, 18328, 18722, 15330, 18304, 15916, 24212, 18328, 18724, 15568, 18332, 18732, 18302, 16298, 18726, 17720, 18732, 18724, 18728, 18302, 18304, 18728, 18956, 18968, 18742, 18744, 18728, 18732, 18332, 18954, 24226, 12412, 18726, 18740, 18300, 18730, 15400, 18330, 12412, 18326, 18740, 24224, 18686, 18942, 18930, 15572, 24170, 16024, 16010, 19036, 14332, 18694, 18934, 18716, 16016, 18944, 18718, 18920, 13788, 16028, 15826, 18942, 18934, 16012, 18944, 16020, 13524, 15916, 18928, 18922, 24230, 18730, 18932, 18712, 12412, 18720, 16036, 13644, 18926, 19036, 16020, 18694, 15826, 18924, 24480, 19036, 18730, 24480, 18934, 18712, 18720, 16018, 14014, 18920, 12412, 18918, 16028, 15858, 18712, 18906, 15902, 15864, 24492, 16062, 18680, 24498, 18910, 18886, 14414, 18664, 15868, 18908, 15808, 15932, 15808, 15858, 18904, 15888, 15902, 18906, 24388, 18890, 24388, 16038, 24386, 15880, 18916, 15876, 15884, 15900, 18902, 15876, 12412, 15860, 15878, 18902, 15546, 16060, 16094, 15856, 15546, 16038, 15884, 24492, 18908, 15902, 18890, 15864, 18900, 12412, 15874, 18914, 15808, 15870, 18590, 18586, 18254, 18580, 18570, 18664, 18582, 18582, 18672, 18700, 18232, 18592, 18580, 18592, 18700, 18568, 18590, 18592, 16472, 18700, 18590, 24386, 18568, 24386, 18882, 18582, 18572, 18592, 18586, 18590, 18680, 18616, 18590, 18582, 18572, 18586, 24500, 15696, 15696, 18672, 18566, 18576, 15546, 15696, 24386, 18676, 24500, 18590, 18566, 13702, 18574, 12412, 18674, 18906, 18700, 18578, 18590, 18664, 18616, 18616, 18266, 18576, 18566, 12412, 18168, 18700, 18580, 18584, 18258, 18692, 18682, 15574, 18268, 24170, 18708, 18294, 16076, 18684, 18256, 15570, 18270, 18268, 12412, 18704, 15766, 18686, 15362, 18284, 24220, 18296, 12412, 18690, 24502]
start_date = '1/22/20'                   #start_date = '1/22/20'    For now shorter date range   #If we set this 1 day farther back in the code we can fix the code just fyi right now we are removing the first day that we say here in the shaped_df thing I think
end_date = '12/31/21'


the_list = []
for i in range(len(stations)):
  the_list.append(names.iloc[stations[i]+2][0])
csse_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
target = ["Montana", "Idaho", "Wyoming", "North Dakota", "South Dakota"]
df = csse_df[csse_df['Province_State'].isin(target)].reset_index()  
csse_df = df

df2 = df.loc[:, start_date:end_date]
df4 = pd.DataFrame(columns=df2.columns, index = list(csse_df['Combined_Key']))

for count,jj in enumerate(the_list):

  #For all of the csvs in the filenames list
  df3 = pd.read_csv(f'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2020/{jj}')   #Last 11 of the csv

  for i in range(len(df3)):
    x = df3.iloc[i]['DATE']
    year = x[2:4]
    month = x[5:7]
    day = x[8:10]
    if month[0] == '0':
      month = month[1]
    if day[0] == '0':
      day = day[1]
    date = month + '/' + day + '/' + year

    if (date in df2.columns):
      df4.iloc[count][date] = df3.iloc[i]['TEMP']
