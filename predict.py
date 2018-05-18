import keras
import re

num_classes = 10
img_rows, img_cols = 24, 31

NN_FILENAME = "model_nn*"
CNN_FILENAME = "model_cnn*"

def predict(vcode, model_name, model):

    # if model_name == NN_FILENAME:
    if re.compile(NN_FILENAME).match(model_name):
        vcode = vcode.reshape(vcode.shape[0], img_rows*img_cols)

    # elif model_name == CNN_FILENAME:
    elif re.compile(CNN_FILENAME).match(model_name):
        from keras import backend as K
        if K.image_data_format() == 'channels_first':
            vcode = vcode.reshape(vcode.shape[0], 1, img_rows, img_cols)
        else:
            vcode = vcode.reshape(vcode.shape[0], img_rows, img_cols, 1)


    # vcode = vcode.astype('float32')
    vcode /= 255
    
    vcode_arr = model.predict(vcode[0:5]).argmax(1)
    vcode_str = str(vcode_arr[0]) + str(vcode_arr[1]) + str(vcode_arr[2]) + str(vcode_arr[3])

    return vcode_str