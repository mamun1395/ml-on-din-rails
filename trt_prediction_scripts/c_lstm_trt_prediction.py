import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.python.saved_model import tag_constants


DATASET_PATH = '../dataset/processed_dataset/'
SAVED_MODEL_PATH = '../trt_saved_models/c_lstm_saved_model_TFTRT_FP32'

datasets = pd.concat([pd.read_csv(file) for file in glob.glob(DATASET_PATH + '*')])

X = datasets.iloc[:, 0:1010]
y = datasets['class']

X_train, X_test, _, y_test = train_test_split(X, y, stratify=y, random_state=42, train_size=0.70)

X_test = X_test.to_numpy()
y_test = list(y_test)

NUMBER_OF_CLASSES = 10

class_mapping = {'nc1_ncn5121_no4': 0, 'nc1_ncn5121_no5': 1, 'nc1_switch_sl2': 2, 'nc1_switch_dem': 3, 'nc1_nodevice': 4,
                 'nc2_ncn5121_no4': 5, 'nc2_ncn5121_no5': 6, 'nc2_switch_sl2': 7, 'nc2_switch_dem': 8, 'nc2_nodevice': 9,}
                 
classes = ['nc1_ncn5121_no4', 'nc1_ncn5121_no5', 'nc1_switch_sl2', 'nc1_switch_dem', 'nc1_nodevice', 'nc2_ncn5121_no4', 'nc2_ncn5121_no5', 'nc2_switch_sl2', 'nc2_switch_dem', 'nc2_nodevice']

y_test = np.array([class_mapping[label] for label in y_test])

y_test = to_categorical(y_test, NUMBER_OF_CLASSES).astype('int32')                 

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_float32 = np.array(X_test, dtype=np.float32)

X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test_reshaped_float32 = np.array(X_test_reshaped, dtype=np.float32)

print('Loading model ...')
saved_model_loaded = tf.saved_model.load(SAVED_MODEL_PATH, tags=[tag_constants.SERVING])
print('Loading finished')

infer = saved_model_loaded.signatures['serving_default']

print('Tensor creating ...')
tensor_dataset = tf.constant(X_test_reshaped_float32)
print('Tensor finished ...')

batch_size = 2
prediction_classes = []
for i in range(0, X_test.shape[0], batch_size):
    print(f'Iteration {i}:') 
    inferred_result = infer(tensor_dataset[i : i+batch_size])
    key = next(iter(inferred_result))
    prediction_classes = prediction_classes + list(np.argmax(inferred_result[key], axis=1))

acc = accuracy_score(prediction_classes, np.argmax(y_test, axis=1))
print(f'\nTest accuracy: {acc}')

print('\nClassification report:\n')
print(classification_report(prediction_classes, np.argmax(y_test, axis=1)))

cm = confusion_matrix(np.argmax(y_test, axis=1), prediction_classes, labels=range(9))
print('\nConfusion matrix:\n')
print(cm)
