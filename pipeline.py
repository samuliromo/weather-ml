import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy import save

#read data and remove unnecessary variables

df = pd.read_csv('minute_weather.csv')
df.rename(columns={'hpwren_timestamp': 'time'}, inplace=True)
df.drop(['rowID', 'avg_wind_direction', 'max_wind_direction', 'min_wind_direction', 'max_wind_speed', 'min_wind_speed','rain_duration'], axis=1, inplace=True)
df.dropna(inplace=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)


#show the correlation matrix
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.show()


#graph temperature over time
ax = plt.gca()
df.plot(kind='line',y='air_temp',ax=ax)
#plt.show()

#============================================================

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
    try:
      index +=1
      if (n_seq == SEQ_LEN):
        n_seq = 0
        pred_seq = []
        for n in range(index, index+PREDICT_PERIOD_LEN):
          pred_seq.append(data[TARGET_VAR][n])
        sequential_data.append([np.array(sequence), classify(sequence, pred_seq)])
        sequence = []
        if (index-1)%6000 == 0:
          print(index)
      sequence.append(i)
      n_seq +=1
    except Exception as e:
      print(e)
  return sequential_data


def balance(data):
  class_0 = []
  class_1 = []
  for seq, target in data:
    try:
      if target == 0:
        class_0.append([seq, target])
      elif target == 1:
        class_1.append([seq, target])
    except Exception as e:
      print(e)

  random.shuffle(class_0)
  random.shuffle(class_1)
  lower = min(len(class_0), len(class_1))
  class_0 = class_0[:lower]
  class_1 = class_1[:lower]
  sequential_data = class_0 + class_1
  random.shuffle(sequential_data)
  X = []
  y = []
  for seq, target in sequential_data:
    X.append(seq)
    y.append(target)
  return np.array(X), y


def normalize(data):
  normalized = np.reshape(data, ((len(data)*SEQ_LEN), 5))
  scaler = preprocessing.MinMaxScaler()
  scaler.fit(normalized)
  normalized = scaler.transform(normalized)
  normalized = np.reshape(normalized, (int((len(normalized)/SEQ_LEN)), SEQ_LEN, 5))
  return normalized


def preprocess_and_save(data, filename):
  df = make_sequences(data)
  x, y = balance(df)
  x = normalize(x)
  save(f'{filename}_x', x)
  save(f'{filename}_y', y)
  print(f"{filename} length: {len(x)}")
  print(f"negatives: {y.count(0)}, positives: {y.count(1)}")

#===============================================================================
train, val, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

preprocess_and_save(train, 'train')
preprocess_and_save(val, 'val')
preprocess_and_save(test, 'test')