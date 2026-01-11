import numpy as np
import time
import requests

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.python.framework import tensor_shape
from keras.regularizers import l2


class DimensionAdaptivePooling(layers.Layer):
    """ Dimension Adaptive Pooling layer for 2D inputs.
    # Arguments
        pool_list: a tuple (W,H)
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each tuple in the list is the number of regions in that pool. For example [(8,6),(4,3)] would be 2
            regions with 1, 8x6 and 4x3 max pools, so 48+12 outputs per feature map.
        forRNN: binary
            Determines wheterh the layer after this is a recurrent layer (LSTM) or not (it is Dense)
        operation: string
            Either `max` or `avg`.
    # Input shape
        4D tensor with shape: `(samples, w, h, M)` .
    # Output shape
        2D or 3D tensor with shape: `(samples,  W*H*M)` or `(samples,  W, H*M)`.
    """

    def __init__(self, pooling_parameters, forRNN=False, operation="max", name=None, **kwargs):
        super(DimensionAdaptivePooling, self).__init__(name=name, **kwargs)
        self.pool_list = np.array(pooling_parameters)
        self.forRNN = forRNN
        self.W = self.pool_list[0]
        self.H = self.pool_list[1]
        self.num_outputs_per_feature_map = self.W * self.H
        if operation == "max":
            self.operation = tf.math.reduce_max
        elif operation == "avg":
            self.operation = tf.math.reduce_mean

    def build(self, input_shape):
        self.M = input_shape[3]

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.forRNN:
            return tensor_shape.TensorShape([input_shape[0], self.W, self.H * self.M])
        else:
            return tensor_shape.TensorShape([input_shape[0], self.W * self.H * self.M])

    def get_config(self):
        config = {'dap pooling parameters': self.pool_list, 'forRNN': self.forRNN}
        base_config = super(DimensionAdaptivePooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DimensionAdaptivePoolingForSensors(DimensionAdaptivePooling):
    def __init__(self, pooling_parameters, forRNN=False, operation="max", name=None, **kwargs):
        super(DimensionAdaptivePoolingForSensors, self).__init__(pooling_parameters=pooling_parameters,
                                                                 forRNN=forRNN,
                                                                 operation=operation,
                                                                 name=name, **kwargs)

    def call(self, xp, mask=None):
        xp_dtype = xp.dtype
        input_shape = tf.shape(xp)
        wp = input_shape[1]  ## This is the number of sample points in each time-window (w')
        hp = input_shape[2]  ## This is the number of sensor channels (h')

        xpp = tf.identity(xp)
        try:
            A = tf.cast(tf.math.maximum(tf.math.ceil((self.H - hp) / 3), 0), dtype=xp_dtype)
            for ia in range(tf.cast(A, tf.int32)):
                xpp = tf.concat([xpp, xp], 2)
            xpp = xpp[:, :wp, :tf.math.maximum(hp, self.H), :]
        except:
            A = tf.Variable(0, dtype=xp_dtype)
        p_w = tf.cast(wp / self.W, dtype=xp_dtype)
        p_h = tf.cast(hp / self.H, dtype=xp_dtype)
        Zp = []
        for iw in range(self.W):
            for ih in range(self.H):
                r1 = tf.cast(tf.math.round(iw * p_w), tf.int32)
                r2 = tf.cast(tf.math.round((iw + 1) * p_w), tf.int32)
                if A == 0:
                    c1 = tf.cast(tf.math.round(ih * p_h), tf.int32)
                    c2 = tf.cast(tf.math.round((ih + 1) * p_h), tf.int32)
                else:
                    c1 = tf.cast(tf.math.round(ih * tf.math.floor((A + 1) * p_h)), tf.int32)
                    c2 = tf.cast(tf.math.round((ih + 1) * tf.math.floor((A + 1) * p_h)), tf.int32)
                try:
                    Zp.append(self.operation(xpp[:, r1:r2, c1:c2, :], axis=(1, 2)))
                except:
                    Zp = []
        Zp = tf.concat(Zp, axis=-1)
        if self.forRNN:
            Zp = tf.reshape(Zp, (input_shape[0], self.W, self.H * self.M))
        else:
            Zp = tf.reshape(Zp, (input_shape[0], self.W * self.H * self.M))
        return Zp


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


def define_my_model_dense(inp_shape, out_shape):
    nb_filters = 64
    drp_out_dns = .5
    nb_dense = 256
    kernel_regularizer = regularizers.l2(0.00005)
    inp = Input(inp_shape)
    layer_0 = Conv2D(64, kernel_size=(5, 1), strides=(1, 1), padding='same', activation=None)(inp)
    x = BatchNormalization()(layer_0)
    x = Activation('relu')(x)
    x = compact_module(x, 9, 64)
    x = inception_module_1(x)
    x = compact_module(x, 5, 128)
    x = compact_module(x, 3, 256)
    #### DAP Layer
    x = DimensionAdaptivePoolingForSensors((4, 1), operation="max", name="DAP", forRNN=False)(x)

    act = Dense(nb_dense, kernel_regularizer=kernel_regularizer, activation='relu', name="act_dns")(x)
    act = Dropout(drp_out_dns, name="act_drp_out")(act)
    out_act = Dense(out_shape, activation='softmax', name="act_smx")(act)
    model = keras.models.Model(inputs=inp, outputs=out_act)
    return model


def define_my_model_dense_2(inp_shape, out_shape):
    nb_filters = 64
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
    x = DimensionAdaptivePoolingForSensors((4, 1), operation="max", name="DAP", forRNN=False)(x)

    act = Dense(nb_dense, kernel_regularizer=kernel_regularizer, activation='relu', name="act_dns")(x)
    act = Dropout(drp_out_dns, name="act_drp_out")(act)
    out_act = Dense(out_shape, activation='softmax', name="act_smx")(act)
    model = keras.models.Model(inputs=inp, outputs=out_act)
    return model

def define_my_dense(inp_shape,out_shape):
    inp = Input(inp_shape)
    x1 = Dense(128, activation='relu')(inp)
    x2 = Dense(17, activation='relu')(x1)
    x3 = Dense(5, activation='relu')(x2)
    model = keras.models.Model(inputs=inp, outputs=x3)
    return model