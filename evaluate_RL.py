import numpy as np
import time
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import tensorflow_addons as tfa
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.python.framework import tensor_shape
from models import *
from evaluate_me import compute_AAMI_performance_measures
from evaluate_me import write_AAMI_results


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


def label_encoding(labels):
    encoded_labels = np.empty((len(labels), 3), dtype=int)
    for i in range(len(labels)):
        if labels[i] == 0:
            encoded_labels[i] = [1, 0, 0]
        elif labels[i] == 1:
            encoded_labels[i] = [0, 1, 0]
        elif labels[i] == 2:
            encoded_labels[i] = [0, 0, 1]
    return np.array(encoded_labels)

def action_to_decimate(action):
    if action == 0:
        return 32
    if action == 1:
        return 64
    if action == 2:
        return 128
    if action == 3:
        return 256

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

def dimension_adaptive_training(model, X_train, Y_train, X_val, Y_val, data_class_weights,
                                batch_size=128, num_epochs=128, save_dir=None,
                                W_combinations=None, H_combinations=None,
                                n_batch_per_train_setp=1):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()
    best_val_accuracy = 0.
    for epoch in range(num_epochs):
        ## Training
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = iter(train_dataset.shuffle(len(X_train)).batch(batch_size))
        n_iterations_per_epoch = len(X_train) // (batch_size * n_batch_per_train_setp)
        epoch_loss_avg = tf.keras.metrics.Mean()
        for i in range(n_iterations_per_epoch):
            rnd_order_H = np.random.permutation(len(H_combinations))
            rnd_order_W = np.random.permutation(len(W_combinations))
            n_samples = 0.
            with tf.GradientTape() as tape:
                accum_loss = tf.Variable(0.)
                for j in range(n_batch_per_train_setp):
                    try:
                        X, Y = next(train_dataset)
                    except:
                        break
                    X = X.numpy()
                    sample_weight = [data_class_weights[y] for y in Y.numpy()]
                    ### Dimension Randomization
                    ####### Random Sensor Selection
                    rnd_H = H_combinations[rnd_order_H[j % len(rnd_order_H)]]
                    X = X[:, :, rnd_H, :]
                    ####### Random Sampling Rate Selection
                    rnd_W = W_combinations[rnd_order_W[j % len(rnd_order_W)]]
                    X = tf.image.resize(X, (rnd_W, len(rnd_H)))
                    logits = model(X)
                    accum_loss = accum_loss + loss_fn(Y, logits, sample_weight)
                    n_samples = n_samples + 1.
            gradients = tape.gradient(accum_loss, model.trainable_weights)
            gradients = [g * (1. / n_samples) for g in gradients]
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            epoch_loss_avg.update_state(accum_loss * (1. / n_samples))

        ## Validation
        accuracy_record = np.zeros((len(W_combinations), len(H_combinations)))
        f1_record = np.zeros((len(W_combinations), len(H_combinations)))
        for w, W_comb in enumerate(W_combinations):
            for h, H_comb in enumerate(H_combinations):
                val_accuracy = tf.keras.metrics.Accuracy()
                val_auc = tfa.metrics.F1Score(num_classes=3, threshold=0.5)
                X = X_val.copy()
                X = X[:, :, H_comb, :]
                X = tf.image.resize(X, (W_comb, len(H_comb)))

                logits = model(X, training=False)
                prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
                val_accuracy.update_state(prediction, Y_val)
                one_hot_val = label_encoding(Y_val)
                val_auc(one_hot_val, np.asarray(logits))
                accuracy_record[w, h] = val_accuracy.result()
                f1_record[w, h] = np.mean(val_auc(one_hot_val, np.asarray(logits)))

        current_val_acc = accuracy_record
        curr_f1 = f1_record
        if np.mean(curr_f1) > np.mean(best_val_accuracy):
            best_val_accuracy = curr_f1
            if save_dir:
                model.save_weights(save_dir)
        # if np.mean(current_val_acc) > np.mean(best_val_accuracy):
        # best_val_accuracy = current_val_acc
        # if save_dir:
        # model.save_weights(save_dir)
        if epoch % 5 == 0:
            print(
                "Epoch {} -- Training Loss = {:.4f} -- Validation Mean Accuracy {:.4f}  -- Validation Mean F1 {:.4f}".format(
                    epoch,
                    epoch_loss_avg.result(),
                    np.mean(current_val_acc),
                    np.mean(f1_record)))

    if save_dir:
        model.load_weights(save_dir)
    print("Best Validation Accuracy {}".format(best_val_accuracy.round(4)))
    print("Training Finished! \n------------------\n")
    return model


print("TensorFlow Version: ", tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

with open('all_data.pickle', 'rb') as handle:
    all_data = pickle.load(handle)

training_data = all_data[0]
valid_data = all_data[1]
test_data = all_data[2]
actions = np.load('AdaptiveSamplingV3_lr0_0001_bs32_trial6_action_trajectory.npy')
X_train_new,Y_train = list_to_array(training_data)
X_train = np.expand_dims(X_train_new, 3)

X_valid_new,Y_valid = list_to_array(valid_data)
X_valid = np.expand_dims(X_valid_new, 3)

X_test_new,Y_test = list_to_array(test_data)
X_test = np.expand_dims(X_test_new, 3)

data_class_names = ["N", "S", "V"]
w = X_train.shape[1]
h = X_train.shape[2]

dana_model = define_my_model_dense_2((None, None, 1), len(np.unique(Y_train)))
dana_model.summary()

dana_model.load_weights("saved_models/dana/new_model7_1")
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
predictions = np.zeros((1, ))
for i in range(len(X_test)):
    current_data = X_test[i,:,:,:]
    true_labels = Y_test[i]
    X = current_data.copy()
    X = np.expand_dims(X, 0)
    action = action_to_decimate(actions[i])
    X = tf.image.resize(X, (action, 1))
    logits_np_array = dana_model(X, training=False)
    prediction = np.argmax(logits_np_array, 1)
    predictions = np.concatenate((predictions,prediction))

predictions = np.delete(predictions, 0, axis=0)

fs = compute_AAMI_performance_measures(prediction, Y_test)
write_AAMI_results(fs, str(w) + 'dana_dense2' + '.csv')

strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    for w, W_comb in enumerate(W_combinations):
        for h, H_comb in enumerate(H_combinations):
            logits_np_array = np.zeros((1, 3))
            test_accuracy = tf.keras.metrics.Accuracy()
            test_specifity = tf.keras.metrics.SpecificityAtSensitivity(0.5)

            X = X_test.copy()
            X = X[:, :, H_comb, :]
            X = tf.image.resize(X, (W_comb, len(H_comb)))
            X_splitted = np.array_split(X, 20, axis=0)
            for k in X_splitted:
                logits_np_array = np.concatenate((logits_np_array, dana_model(k, training=False)))
            logits = np.delete(logits_np_array, 0, axis=0)
            prediction = np.argmax(logits, 1)
            fs = compute_AAMI_performance_measures(prediction, Y_test)
            write_AAMI_results(fs, str(w) + 'dana_dense2' + '.csv')

print('exit')