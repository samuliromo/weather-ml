import time
from numpy import load
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

EPOCHS = 5
BATCH_SIZE = 32
#NAME = f"RNN-PRED-{int(time.time())}"

train_x = load('train_x.npy')
train_y = load('train_y.npy')
val_x = load('val_x.npy')
val_y = load('val_y.npy')

print(train_x.shape)

#print(len(train_x), len(train_y), len(val_x), len(val_y))
print(train_x.shape[1:])

model = Sequential()

model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu'))
#model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))

model.add(Dense(2, activation='relu'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_x, val_y)
)
