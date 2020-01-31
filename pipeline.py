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
df['time'] = pd.to_datetime(df['time']) #set the index to datetime
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
plt.show()

#============================================================

SEQ_LEN = 60 #set the length of the PAST sequence we want to use for prediction
PREDICT_PERIOD_LEN = 60 #set the length of the FUTURE sequence we want to gain some information about by predicting
TARGET_VAR = 'air_temp' #the target variable the positive or negative change of which we try to predict
TARGET_INDEX = df.columns.get_loc(TARGET_VAR) #index of the target variable in the main data frame


#assign the labels for sequences. 0 for negative change, 1 for positive
def classify(current_seq, future_seq):
  data = []
  for i in current_seq:
    data.append(i[TARGET_INDEX])
  if np.mean(future_seq) > np.mean(data):
    return 1
  else:
    return 0


#split the main sequence into "past" and "future" sequences, the length of which are determined separately
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


#balance the data to have an equal amount of both classes, and shuffle it
#method taken from: https://pythonprogramming.net/balancing-rnn-data-deep-learning-python-tensorflow-keras/
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


#normalize the data using min-max scaling
def normalize(data):
  normalized = np.reshape(data, ((len(data)*SEQ_LEN), 5))
  scaler = preprocessing.MinMaxScaler()
  scaler.fit(normalized)
  normalized = scaler.transform(normalized)
  normalized = np.reshape(normalized, (int((len(normalized)/SEQ_LEN)), SEQ_LEN, 5))
  return normalized


#combine the whole pipeline into a single step, save the array and print some quick statistics
def preprocess_and_save(data, filename):
  filename = f'past-{SEQ_LEN}-future-{PREDICT_PERIOD_LEN}-{TARGET_VAR}/'
  df = make_sequences(data)
  x, y = balance(df)
  x = normalize(x)
  save(f'{filename}_x', x)
  save(f'{filename}_y', y)
  print(f"{filename} length: {len(x)}")
  print(f"negatives: {y.count(0)}, positives: {y.count(1)}")

#===============================================================================
#Split the main dataframe into training/validation/testing data with a ratio of 60/20/20
train, val, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

#complete pipeline for train, validation and test datasets
preprocess_and_save(train, 'train')
preprocess_and_save(val, 'val')
preprocess_and_save(test, 'test')