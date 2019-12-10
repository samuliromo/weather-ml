import pandas as pd
import matplotlib.pyplot as plt

#read data and remove unnecessary variables
df = pd.read_csv('minute_weather.csv')
df.drop(['rowID', 'avg_wind_direction', 'max_wind_direction', 'min_wind_direction', 'max_wind_speed', 'min_wind_speed','rain_duration'], axis=1, inplace=True)
print(df.head())
print(df.info())
df.dropna(inplace=True)
print(df.info())
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
ax = plt.gca()
df.plot(kind='line',x='hpwren_timestamp',y='air_temp',ax=ax)
plt.show()