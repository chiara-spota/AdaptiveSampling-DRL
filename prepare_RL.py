import numpy as np
import time
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import tensorflow_addons as tfa
import  pickle
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
    encoded_labels = np.empty((len(labels), 5), dtype=int)
    for i in range(len(labels)):
        if labels[i] == 0:
            encoded_labels[i] = [1, 0, 0, 0, 0]
        elif labels[i] == 1:
            encoded_labels[i] = [0, 1, 0, 0, 0]
        elif labels[i] == 2:
            encoded_labels[i] = [0, 0, 1, 0, 0]
        elif labels[i] == 3:
            encoded_labels[i] = [0, 0, 0, 1, 0]
        elif labels[i] == 4:
            encoded_labels[i] = [0, 0, 0, 0 ,1]
    return np.array(encoded_labels)

def give_me_reward(decimate_rate,true_labels,predict):
    return_array = np.zeros(len(predict,))
    for i in range(len(predict)):
        if predict[i] == true_labels[i]:
            return_array[i,] = decimate_rate
        else:
            return_array[i,] = -10
    return return_array

def dimension_adaptive_training(model, X_train, Y_train, X_val, Y_val,
                                batch_size=128, num_epochs=128, save_dir=None,
                                W_combinations=None, H_combinations=None,
                                n_batch_per_train_setp=1):
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
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
                    ### Dimension Randomization
                    ####### Random Sensor Selection
                    rnd_H = H_combinations[rnd_order_H[j % len(rnd_order_H)]]
                    X = X[:, :, rnd_H, :]
                    ####### Random Sampling Rate Selection
                    rnd_W = W_combinations[rnd_order_W[j % len(rnd_order_W)]]
                    X = tf.image.resize(X, (rnd_W, len(rnd_H)))
                    logits = model(X)
                    accum_loss = accum_loss + loss_fn(Y, logits)
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
                logits_np_array = np.zeros((1, 5))
                val_auc = tfa.metrics.F1Score(num_classes=5, threshold=0.5)
                X = X_val.copy()
                X = X[:, :, H_comb, :]
                X = tf.image.resize(X, (W_comb, len(H_comb)))

                X_splitted = np.array_split(X, 20, axis=0)
                for k in X_splitted:
                    logits_np_array = np.concatenate((logits_np_array, dana_model(k, training=False)))
                logits = np.delete(logits_np_array, 0, axis=0)
                prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
                val_auc(Y_val, np.asarray(logits))
                f1_record[w, h] = np.mean(val_auc(Y_val, np.asarray(logits)))

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
                "Epoch {} -- Training Loss = {:.4f} -- Validation Mean F1 {:.4f}".format(
                    epoch,
                    epoch_loss_avg.result(),
                    np.mean(f1_record)))

    if save_dir:
        model.load_weights(save_dir)
    print("Best Validation Accuracy {}".format(best_val_accuracy.round(4)))
    print("Training Finished! \n------------------\n")
    return model

def add_reward_tables(subject_list, model, W_combinations):
    """
    subject_list: list of subjects, each subject = [X, y]
        X shape: (num_beats, 256)  (or similar)
        y shape: (num_beats,)
    Appends reward_table (num_beats, 4) to each subject => [X, y, reward_table]
    """
    for i in range(len(subject_list)):
        current_patient = subject_list[i]
        X_patient = current_patient[0]   # (num_beats, 256)
        y_patient = current_patient[1]   # (num_beats,)

        patient_array = np.zeros((len(y_patient), 4), dtype=np.int32)
        counter = 0

        # Prepare base shape expected by model: (N, 256, 1, 1)
        current_data = np.expand_dims(X_patient, 2)   # (N, 256, 1)
        current_data = np.expand_dims(current_data, 3)  # (N, 256, 1, 1)

        for W_comb in W_combinations:
            logits_np_array = np.zeros((1, 3))
            X = tf.image.resize(current_data, (W_comb, 1))

            # avoid huge batches
            X_splitted = np.array_split(X, 10, axis=0)
            for k in X_splitted:
                logits_np_array = np.concatenate((logits_np_array, model(k, training=False)))
            logits = np.delete(logits_np_array, 0, axis=0)

            prediction = np.argmax(logits, axis=1)

            # decimate_rate = 256 / W_comb -> {8,4,2,1}
            return_array = give_me_reward(256 / W_comb, y_patient, prediction)
            patient_array[:, counter] = return_array
            counter += 1

        # Append reward table at index 2 (DQL environment expects this!)
        subject_list[i].append(patient_array)

    return subject_list

print("TensorFlow Version: ", tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_class_names = ["N", "S","V"]

with open('all_data.pickle', 'rb') as handle:
    all_data = pickle.load(handle)

training_data = all_data[0]
valid_data = all_data[1]
test_data = all_data[2]

X_train_new, Y_train = list_to_array(training_data)   # (N_train, 256, 1)
X_train = np.expand_dims(X_train_new, 3) 

# dana_model = Ordonez2016DeepWithDAP((None, None, 1), len(np.unique(Y_train)))
dana_model = define_my_model_dense_2((None, None, 1), 3)
dana_model.summary()

### These are a subset of feasible situations in both dimensions
W_combinations = [32,64,128,256]
H_combinations = [[0]]
n_batch_per_train_setp = 5  ## This is B

#dana_model.load_weights("saved_models/dana/checkpoint")
dana_model.load_weights("saved_models/dana/dap_cnn")
"""
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
"""
# Build reward tables for RL training + evaluation
# DQL trainer expects: subject = [states, labels, reward_table]
print("Computing reward tables for TRAIN...")
training_data = add_reward_tables(training_data, dana_model, W_combinations)

print("Computing reward tables for EVAL...")
valid_data = add_reward_tables(valid_data, dana_model, W_combinations)

# Save RL-ready pickles
with open("rl_train.pkl", "wb") as f:
    pickle.dump(training_data, f)

with open("rl_eval.pkl", "wb") as f:
    pickle.dump(valid_data, f)

print("Saved rl_train.pkl and rl_eval.pkl")

"""
for i in range(len(test_data)):
    current_patient = test_data[i]
    patient_array = np.zeros((len(current_patient[1]), 4))
    counter = 0
    for w, W_comb in enumerate(W_combinations):
        logits_np_array = np.zeros((1, 3))
        true_labels = current_patient[1]
        current_data = np.expand_dims(current_patient[0], 2)
        current_data = np.expand_dims(current_data, 3)
        X = current_data.copy()
        X = X[:, :, :, :]
        X = tf.image.resize(X, (W_comb, 1))
        X_splitted = np.array_split(X, 10, axis=0)
        for k in X_splitted:
            logits_np_array = np.concatenate((logits_np_array, dana_model(k, training=False)))
        logits = np.delete(logits_np_array, 0, axis=0)
        prediction = np.argmax(logits, 1)
        return_array = give_me_reward(256/W_comb,true_labels,prediction)
        patient_array[:, counter] = return_array
        counter += 1
    test_data[i].append(patient_array)
"""
"""
with strategy.scope():
    for w, W_comb in enumerate(W_combinations):
        for h, H_comb in enumerate(H_combinations):
            logits_np_array = np.zeros((1, 3))

            X = X_train.copy()
            X = X[:, :, H_comb, :]
            X = tf.image.resize(X, (W_comb, len(H_comb)))
            X_splitted = np.array_split(X, 20, axis=0)
            for k in X_splitted:
                logits_np_array = np.concatenate((logits_np_array, dana_model(k, training=False)))
            logits = np.delete(logits_np_array, 0, axis=0)
            prediction = np.argmax(logits, 1)
            fs = compute_AAMI_performance_measures(prediction, Y_train)
            write_AAMI_results(fs, str(w) + 'dana_best' + '.csv')
"""
print('exit')

print(dana_model(np.zeros((1, 256, 1, 1), dtype=np.float32)).numpy())

import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# --- Load data ---
with open("all_data.pickle", "rb") as f:
    all_data = pickle.load(f)

test_data = all_data[2]  # or use valid_data if you prefer
# convert test_data list -> arrays (like you already do)
def list_to_array(data):
    X = np.zeros((1,256))
    y = np.zeros((1,), dtype=int)
    for seg, lab in data:
        X = np.concatenate((X, seg))
        y = np.concatenate((y, lab))
    X = np.delete(X, 0, 0)
    y = np.delete(y, 0)
    X = np.expand_dims(X, 2)     # (N,256,1)
    X = np.expand_dims(X, 3)     # (N,256,1,1)
    return X, y

X_test, y_test = list_to_array(test_data)

# --- Load CNN model + weights ---
from models import define_my_model_dense_2
dana_model = define_my_model_dense_2((None, None, 1), 3)
dana_model.load_weights("saved_models/dana/dap_cnn")

# --- Choose sampling rates ---
W_combinations = [32, 64, 128, 256]  # Fs/8, Fs/4, Fs/2, Fs
rate_labels = ["Fs/8", "Fs/4", "Fs/2", "Fs"]

# If you want Fs/16 like the paper, add:
# W_combinations = [16, 32, 64, 128, 256]
# rate_labels = ["Fs/16","Fs/8","Fs/4","Fs/2","Fs"]

# --- Compute PPV (precision) per class at each rate ---
ppv = []  # shape (num_rates, num_classes)
for W in W_combinations:
    Xr = tf.image.resize(X_test, (W, 1))  # (N,W,1,1)

    # batch to avoid RAM spikes
    preds = []
    for chunk in np.array_split(Xr, 20, axis=0):
        logits = dana_model(chunk, training=False).numpy()
        preds.append(np.argmax(logits, axis=1))
    y_pred = np.concatenate(preds)

    # precision per class: order is [0,1,2] = [N,S,V] in your setup
    prec = precision_score(y_test, y_pred, labels=[0,1,2], average=None, zero_division=0)
    ppv.append(prec)

ppv = np.array(ppv)  # (num_rates, 3)

# --- Plot like paper (grouped bars per sampling rate) ---
beat_names = ["N", "S", "V"]

x = np.arange(len(rate_labels))
bar_w = 0.25

plt.figure(figsize=(9,5))
for c in range(3):
    plt.bar(x + (c-1)*bar_w, ppv[:, c]*100, width=bar_w, label=beat_names[c])

plt.xticks(x, rate_labels)
plt.ylim(0, 100)
plt.ylabel("Positive Predictive Value (PPV, %)")
plt.xlabel("Sampling rate")
plt.title("PPV per beat type under different sampling rates")
plt.legend()
plt.tight_layout()
plt.savefig("ppv_vs_sampling.png", dpi=200)
plt.close()

print("Saved: ppv_vs_sampling.png")
print("PPV table (%):")
for i, lab in enumerate(rate_labels):
    print(lab, (ppv[i]*100).round(2))

