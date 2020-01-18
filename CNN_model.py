import tensorflow as tf
from numpy import load
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

EPOCHS = 15
BATCH_SIZE = 64
#NAME = f"RNN-PRED-{int(time.time())}"

train_x = load('train_x.npy')
train_y = load('train_y.npy')
val_x = load('val_x.npy')
val_y = load('val_y.npy')

print(train_x.shape)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(train_x.shape[1:])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit model
model.fit(train_x, train_y, 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE,
          validation_data=(val_x, val_y),
          verbose=1)
