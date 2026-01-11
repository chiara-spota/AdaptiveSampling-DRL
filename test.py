import pickle

with open('all_data.pickle', 'rb') as handle:
    all_data = pickle.load(handle)

print(type(all_data))
print(len(all_data))

patients_train_cnn, patients_valid_cnn, patients_test_cnn = all_data

print("Number of patients in CNN train:", len(patients_train_cnn))
print("Number of patients in CNN valid:", len(patients_valid_cnn))
print("Number of patients in CNN test:", len(patients_test_cnn))

example_data, example_labels = patients_train_cnn[0]
print(example_data.shape)
print(example_labels.shape)
print("Label values:", set(example_labels))
