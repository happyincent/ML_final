from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard

from PIL import Image
import numpy as np

MODEL_FILENAME = 'model_cnn.h5'
VCODE_PATH = 'code_auto/code'
LABEL_PATH = 'label_auto.txt'
LOG_PATH = '/home/ddl/Desktop/cnn_log'

TEST_THRESHOLD = 8.0/10.0

batch_size = 128
num_classes = 10
epochs = 50

img_rows, img_cols = 24, 31

y = []
with open(LABEL_PATH) as f:
    for line in f:
        y.append(line[0])

_max = len(y)
_test = int(_max*TEST_THRESHOLD)

x = []
for i in range(1,_max+1):
    im = Image.open(VCODE_PATH + str(i) + '.bmp')
    x.append(np.asarray(im, dtype="uint8"))
x = np.asarray(x)

x_train = x[0:_test]
x_test = x[_test:_max]

y_train = y[0:_test]
y_test = y[_test:_max]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=[TensorBoard(log_dir=LOG_PATH)])

model.save(MODEL_FILENAME)
print("Saved model to disk")