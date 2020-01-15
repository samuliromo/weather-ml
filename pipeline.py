import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#read data and remove unnecessary variables

df = pd.read_csv('minute_weather.csv')
df.rename(columns={'hpwren_timestamp': 'time'}, inplace=True)
df.drop(['rowID', 'avg_wind_direction', 'max_wind_direction', 'min_wind_direction', 'max_wind_speed', 'min_wind_speed','rain_duration'], axis=1, inplace=True)
df.dropna(inplace=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
#print(df.info())
#print(df.head())


"""
#show the correlation matrix
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
#plt.show()
"""

#graph temperature over time
"""
ax = plt.gca()
df.plot(kind='line',y='air_temp',ax=ax)
plt.show()
"""
#=======================================
"""
times = sorted(df.index.values)
last_5pct = times[-int(0.05*len(times))]
print(last_5pct)

validation_df = df[(df.index >= last_5pct)]
print(validation_df.head, validation_df.tail)
"""
#=======================================


SEQ_LEN = 60
PREDICT_PERIOD_LEN = 60
TARGET_VAR = 'air_temp'
TARGET_INDEX = df.columns.get_loc(TARGET_VAR)

def classify(current_seq, future_seq):
  data = []
  for i in current_seq:
    data.append(i[TARGET_INDEX])
  if np.mean(future_seq) > np.mean(data):
    return 1
  else:
    return 0

def make_sequences(data):
  print('Creating sequences.....')
  sequential_data = []
  sequence = []
  n_seq = 0
  index = 0
  for i in data.values:
    index +=1
    if (n_seq == SEQ_LEN):
      n_seq = 0
      pred_seq = []
      for n in range(index, index+PREDICT_PERIOD_LEN):
        pred_seq.append(data[TARGET_VAR][n])
      sequence.append(classify(sequence, pred_seq))
      sequential_data.append(sequence)
      sequence = []
      print(index, len(sequential_data))
    sequence.append(i)
    n_seq +=1
    if len(sequential_data) == 10:
      break
  return sequential_data


data = make_sequences(df)