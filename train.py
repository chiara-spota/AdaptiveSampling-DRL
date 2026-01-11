import numpy as np
import pickle5 as p

import tensorflow as tf
tf.config.run_functions_eagerly(True) 
import tensorflow.keras as keras
import tensorflow_addons as tfa

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

# Reuse helper + metrics
from models import define_my_model_dense_2
from evaluate_me import compute_AAMI_performance_measures, write_AAMI_results

print(">>> RUNNING DAP TRAIN SCRIPT (define_my_model_dense_2, (None, None, 1)) <<<")

print("TensorFlow Version: ", tf.__version__)

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

# -------------------------------------------------------------------
# 1. Load data (same logic as dana_MIT_constant)
# -------------------------------------------------------------------
with open('all_data.pickle', 'rb') as handle:
    all_data = p.load(handle)

training_data = all_data[0]
valid_data = all_data[1]
test_data = all_data[2]

X_train_new, Y_train = list_to_array(training_data)   # (N_train, 256, 1)
X_valid_new, Y_valid = list_to_array(valid_data)      # (N_valid, 256, 1)
X_test_new,  Y_test  = list_to_array(test_data)       # (N_test,  256, 1)

# Add the last dimension for channels: (N, 256, 1, 1)
X_train = np.expand_dims(X_train_new, 3)
X_valid = np.expand_dims(X_valid_new, 3)
X_test  = np.expand_dims(X_test_new, 3)

data_class_names = ["N", "S", "V"]

w = X_train.shape[1]
h = X_train.shape[2]

# -------------------------------------------------------------------
# 2. Compute class weights (reuse same idea as dana_MIT_constant)
# -------------------------------------------------------------------
data_class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(data_class_names)),
    y=Y_train
)
data_class_weights = dict(zip(range(len(data_class_weights)),
                              data_class_weights.round(2)))

print("- Data Shape:\n -- Training:  Data {} Labels {} \n -- Validation: Data {} Labels {} \n -- Testing: Data {} Labels {}"
      .format(X_train.shape, Y_train.shape,
              X_valid.shape, Y_valid.shape,
              X_test.shape,  Y_test.shape))
print("- Activity Weights", dict(zip(data_class_names, data_class_weights.values())))

# One-hot labels for CategoricalCrossentropy + F1
ytrain = to_categorical(Y_train, num_classes=3, dtype='int')
yval   = to_categorical(Y_valid, num_classes=3, dtype='int')

# -------------------------------------------------------------------
# 3. Build + compile DAP CNN (define_my_model_dense_2 from models.py)
# -------------------------------------------------------------------
# Input shape uses variable length and 1 “sensor” dimension
dana_model = define_my_model_dense_2(inp_shape=(None, None, 1), out_shape=3)

opt = Adam(learning_rate=0.001)
cce = tf.keras.losses.CategoricalCrossentropy()

dana_model.compile(
    optimizer=opt,
    loss=cce,
    metrics=[
        'accuracy',
        tfa.metrics.F1Score(num_classes=3, threshold=0.5, average='macro', name='f1_score')
    ]
)

dana_model.summary()

# -------------------------------------------------------------------
# 4. Callbacks: checkpoint + early stopping (similar to constant CNN)
# -------------------------------------------------------------------
checkpoint_filepath = 'saved_models/dana/dap_cnn'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,  
    monitor='val_f1_score',
    mode='max',
    save_best_only=True,
    verbose=1
)

stop_me = tf.keras.callbacks.EarlyStopping(
    monitor='val_f1_score',
    min_delta=0,
    patience=100,
    verbose=1,
    mode='max',
    baseline=None,
    restore_best_weights=True
)

# -------------------------------------------------------------------
# 5. Train DAP CNN at full length (256)
#    (DAP layer still allows variable W at inference)
# -------------------------------------------------------------------
history = dana_model.fit(
    X_train,
    ytrain,
    epochs=500,
    batch_size=80,
    verbose=1,
    validation_data=(X_valid, yval),
    callbacks=[stop_me, model_checkpoint_callback],
    class_weight=data_class_weights
)

# -------------------------------------------------------------------
# 6. Load best weights and evaluate on test set
# -------------------------------------------------------------------
dana_model.load_weights(checkpoint_filepath)

y_pred = dana_model.predict(X_test)
prediction = np.argmax(y_pred, axis=1)

fs = compute_AAMI_performance_measures(prediction, Y_test)
write_AAMI_results(fs, 'dap_cnn_256.csv')

print("Finished training DAP CNN. Best weights saved to:", checkpoint_filepath)
print("Test performance written to: dap_cnn_256.csv")
