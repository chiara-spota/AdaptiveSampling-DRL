import numpy as np
import time
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pickle
import pickle5 as p
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.python.framework import tensor_shape
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from evaluate_me import compute_AAMI_performance_measures
from evaluate_me import write_AAMI_results


def inception_module_1(layer_in):
    conv1 = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='GlorotNormal',
                   kernel_regularizer=l2(0.0002))(layer_in)
    conv4 = Conv2D(16, kernel_size=(4, 1), padding='same', activation='relu', kernel_initializer='GlorotNormal',
                   kernel_regularizer=l2(0.0002))(layer_in)
    conv16 = Conv2D(16, kernel_size=(16, 1), padding='same', activation='relu', kernel_initializer='GlorotNormal',
                    kernel_regularizer=l2(0.0002))(layer_in)
    layer_out = concatenate([conv1, conv4, conv16], axis=-1)
    return layer_out


def conv_block(input_data, filters, conv_size):
    x1 = Conv2D(filters, kernel_size=(conv_size, 1), strides=(1, 1), padding='same', activation=None)(input_data)
    x1 = BatchNormalization()(x1)
    x = Conv2D(filters, kernel_size=(conv_size, 1), strides=(1, 1), padding='same', activation=None)(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=(conv_size, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = Add()([x, x1])
    x = Activation('relu')(x)
    return x

def iden_block(input_data, filters, conv_size):
    x = Conv2D(filters, kernel_size=(conv_size, 1), strides=(1, 1), padding='same', activation=None)(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([x, input_data])
    x = Activation('relu')(x)
    return x

def compact_module(input_data, conv_size, filters):
    x = conv_block(input_data, filters, conv_size)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    # x = iden_block(x, filters, conv_size)
    return x

def list_to_array(data):
    return_data = np.zeros((1,256))
    return_labels = np.zeros((1,),dtype=int)
    for i in data:
        data1 = i[0]
        label1 = i[1]
        return_data = np.concatenate((return_data,data1))
        return_labels = np.concatenate((return_labels,label1))
    return_data = np.delete(return_data, 0, 0)
    return_labels = np.delete(return_labels, 0)
    return_data = np.expand_dims(return_data, 2)
    return return_data,return_labels

def define_my_model_dense(inp_shape, out_shape):
    drp_out_dns = .5
    nb_dense = 129
    kernel_regularizer = regularizers.l2(0.00005)
    inp = Input(inp_shape)
    layer_0 = Conv2D(32, kernel_size=(9, 1), strides=(2, 1), padding='same', activation=None)(inp)
    x = BatchNormalization()(layer_0)
    x = Activation('relu')(x)
    x = compact_module(x, 7, 48)
    x = inception_module_1(x)
    x = compact_module(x, 5, 128)
    #### DAP Layer
    x = MaxPooling2D(pool_size=(4, 1), strides=(4, 1))(x)
    x = Flatten()(x)
    act = Dense(nb_dense, kernel_regularizer=kernel_regularizer, activation='relu', name="act_dns")(x)
    act = Dropout(drp_out_dns, name="act_drp_out")(act)
    out_act = Dense(out_shape, activation='softmax', name="act_smx")(act)
    BerkenLeNet = Model(inputs=inp, outputs=out_act)
    opt = Adam(learning_rate=0.001)
    cce = tf.keras.losses.CategoricalCrossentropy()
    BerkenLeNet.compile(optimizer=opt, loss=cce,
                        metrics=['accuracy', tfa.metrics.F1Score(num_classes=3, threshold=0.5, average='macro')])
    BerkenLeNet.summary()
    return BerkenLeNet


print("TensorFlow Version: ", tf.__version__)

with open('all_data.pickle', 'rb') as handle:
    all_data = p.load(handle)

training_data = all_data[0]
valid_data = all_data[1]
test_data = all_data[2]

X_train_new,Y_train = list_to_array(training_data)
X_train = np.expand_dims(X_train_new, 3)

X_valid_new,Y_valid = list_to_array(valid_data)
X_valid = np.expand_dims(X_valid_new, 3)

X_test_new,Y_test = list_to_array(test_data)
X_test = np.expand_dims(X_test_new, 3)

data_class_names = ["N", "S", "V"]

w = X_train.shape[1]
h = X_train.shape[2]

## Computing the class weight for each label (in case that dataset is not balanced)
data_class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                       classes=np.arange(len(data_class_names)),
                                                       y=Y_train)
data_class_weights = dict(zip(range(len(data_class_weights)), data_class_weights.round(2)))
print("- Data Shape:\n -- Training:  Data {} Labels {} \n -- Testing: Data {} Labels {}".
      format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
print("- Activity Weights", dict(zip(data_class_names, data_class_weights.values())))

### These are a subset of feasible situations in both dimensions
W_combinations = [32, 64, 128, 256]
normal_size = 256
for w in W_combinations:
    decimate_ratio = int(normal_size / w)
    X_train_l = X_train[:, 0::decimate_ratio, :]
    X_valid_l = X_valid[:, 0::decimate_ratio, :]
    X_test_l = X_test[:, 0::decimate_ratio, :]

    ytrain = to_categorical(Y_train, num_classes=3, dtype='int')
    yval = to_categorical(Y_valid, num_classes=3, dtype='int')

    model = define_my_model_dense(inp_shape=(w, 1, 1), out_shape=3)
    checkpoint_filepath = 'checkpoints_weight'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=False,
                                                                   monitor='val_f1_score',
                                                                   mode='max',
                                                                   save_best_only=True)

    stop_me = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', min_delta=0, patience=100,
                                               verbose=1, mode='max', baseline=None, restore_best_weights=True)

    history = model.fit(X_train_l, ytrain, epochs=500, batch_size=80, verbose=1,
                        validation_data=(X_valid_l, yval), callbacks=[stop_me, model_checkpoint_callback],
                        class_weight=data_class_weights)

    model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={"F1Score": tfa.metrics.F1Score})

    y_pred = model.predict(X_test_l)

    prediction = np.argmax(y_pred, 1)
    fs = compute_AAMI_performance_measures(prediction, Y_test)
    write_AAMI_results(fs, str(w) + 'constant_new_weight' + '.csv')

    print('exit')
