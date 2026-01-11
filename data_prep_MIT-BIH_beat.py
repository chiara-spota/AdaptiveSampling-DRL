import glob
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import math
import pickle
from scipy import signal
import sklearn.preprocessing

def normalize_me(data):
    for k in range(len(data)):
        data[k, :] = sklearn.preprocessing.normalize(data[k, :].reshape(1, -1), norm='max', axis=1, copy=True, return_norm=False)
    return data

def give_annots(current_annot, data, limits):
    symbol = current_annot.symbol
    samples  = current_annot.sample
    # 110 points before 146 points after
    return_array = np.zeros((1,258))
    for i in range(0,len(symbol)):
        if samples[i] < 110 or samples[i] + 146 > len(data) or samples[i] > limits[-1] or samples[i] < limits[0]:
            continue
        else:
            if symbol[i] == 'N' or symbol[i] == 'L' or symbol[i] == 'R' or symbol[i] == 'e' or symbol[i] == 'j':
                new_beat = data[samples[i]-110:samples[i]+146]
                new_beat = np.append(new_beat,samples[i])
                new_beat = np.append(new_beat, 0)
                new_beat_reshape = np.reshape(new_beat, (1, 258))
                return_array = np.concatenate((return_array,new_beat_reshape))
            elif symbol[i] == 'A' or symbol[i] == 'a' or symbol[i] == 'J' or symbol[i] == 'S':
                new_beat = data[samples[i]-110:samples[i]+146]
                new_beat = np.append(new_beat,samples[i])
                new_beat = np.append(new_beat, 1)
                new_beat_reshape = np.reshape(new_beat, (1, 258))
                return_array = np.concatenate((return_array,new_beat_reshape))
            elif symbol[i] == 'V' or symbol[i] == 'E':
                new_beat = data[samples[i]-110:samples[i]+146]
                new_beat = np.append(new_beat,samples[i])
                new_beat = np.append(new_beat, 2)
                new_beat_reshape = np.reshape(new_beat, (1, 258))
                return_array = np.concatenate((return_array,new_beat_reshape))
            else:
                continue
    return_array = np.delete(return_array, 0, 0)
    return return_array

file1 = open('mitbih/RECORDS.txt', 'r')
count = 0

# Using for loop
records = []
print("Using for loop")
for line in file1:
    count += 1
    records.append(line.strip())

file1.close()
training_dataset = ['100','101','103','105','106','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124']
patients_train_cnn = []
patients_valid_cnn = []
patients_test_cnn = []
limit = 360 * 60 * 5
positive_infinity = float('inf')
patients = []
patients_eval = []
cnn_train = []
for i in records:
    current_data = wfdb.rdsamp(f"mitbih/{i}")
    current_annot = wfdb.rdann(f"mitbih/{i}",extension='atr')
    if i in training_dataset:
        data = give_annots(current_annot, current_data[0][:, 0], [0, positive_infinity])
        data_train = normalize_me(data[len(data)//5:,0:256])
        data_train_label = data[len(data)//5:,-1].astype(int)
        data_valid = normalize_me(data[:len(data)//5,0:256])
        data_valid_label = data[:len(data)//5,-1].astype(int)
        patients_train_cnn.append([data_train, data_train_label])
        patients_valid_cnn.append([data_valid, data_valid_label])
    else:
        data_test = give_annots(current_annot, current_data[0][:, 0], [limit, positive_infinity])
        data_train = give_annots(current_annot, current_data[0][:, 0], [0,limit])

        data_train_2 = normalize_me(data_train[len(data_train)//5:,0:256])
        data_train_label = data_train[len(data_train)//5:,-1].astype(int)
        data_valid = normalize_me(data_train[:len(data_train)//5,0:256])
        data_valid_label = data_train[:len(data_train) // 5, -1].astype(int)
        patients_train_cnn.append([data_train_2, data_train_label])
        patients_valid_cnn.append([data_valid, data_valid_label])
        data_test_2 = normalize_me(data_test[:,0:256])
        data_test_label = data_test[:,-1].astype(int)
        patients_test_cnn.append([data_test_2, data_test_label])

all_data = [patients_train_cnn, patients_valid_cnn, patients_test_cnn]

with open('all_data.pickle', 'wb') as handle:
    pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('exit')
