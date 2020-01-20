import time
from numpy import load
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Flatten

EPOCHS = 5
BATCH_SIZE = 32

train_x = load('train_x.npy')
train_y = load('train_y.npy')
val_x = load('val_x.npy')
val_y = load('val_y.npy')

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_x, val_y)
)
