import tensorflow as tf
from numpy import load, array
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D

#set the amount of training epochs and batch size
EPOCHS = 5
BATCH_SIZE = 32

#load the training and validation data
train_x = load('train_x.npy')
train_y = load('train_y.npy')
val_x = load('val_x.npy')
val_y = load('val_y.npy')

#build the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(train_x.shape[1:]), dilation_rate=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', dilation_rate=2))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #sigmoid activation function works well for binary classification
 #add the optimizer
opt = tf.keras.optimizers.Adam(lr=0.0008, decay=1e-6)

#compile the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

#do the actual training
model.fit(train_x, train_y, 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE,
          validation_data=(val_x, val_y),
          verbose=1)
