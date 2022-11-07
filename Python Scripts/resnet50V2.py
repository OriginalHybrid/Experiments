import tensorflow as tf

print(tf.__version__)

# other imports
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
import tensorflow_hub as hub
from tensorflow.keras import layers

# Scikit Imports
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

import os
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
# os.environ["KMP_BLOCKTIME"] = "0"
# os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

from tensorflow.keras import mixed_precision
# Setting precision policy
policy = mixed_precision.Policy('bfloat16')
mixed_precision.set_global_policy(policy)


result = {}
acc = []
time_taken = []
IMAGE_SHAPE = (32, 32)

import sys

k = int(sys.argv[1])
e = int(sys.argv[2])

def create_model(num_classes=10):

    #  1. Create base model with tf.keras.applications
    feature_extractor_layer = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')

    # 2. Freeze the base model (so the pre-learned patterns remain)
    feature_extractor_layer.trainable = False

    print(feature_extractor_layer)

    # 3. Create inputs into the base model
    inputs = tf.keras.layers.Input(shape=(32, 32, 3), name="input_layer")
    x = feature_extractor_layer(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax', name='output_layer')(x)
  
    model = tf.keras.Model(inputs, outputs)

    return model

def build_model(no_of_epochs):
    
    import time
    tf.keras.backend.clear_session()
    # Load in the data
    cifar10 = tf.keras.datasets.cifar10

    # Distribute it to train and test set
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x_train, x_val, y_train, y_val = train_test_split(x_train, 
                                                  y_train, 
                                                  test_size=0.1, 
                                                  stratify=np.array(y_train), 
                                                  random_state=42)

    NUM_CLASSES = 10
    # Reduce pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # convert the label values
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
    y_val = np_utils.to_categorical(y_val, NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

    IMAGE_SHAPE = (32, 32)
    # number of classes
    K = 10

    # calculate total number of classes
    # for output layer
    print("number of classes:", K)

    # Create model
    tf.keras.backend.clear_session()
    resnet_model = create_model(num_classes=K)

    # Compile
    resnet_model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
                        metrics=['accuracy'])
                        
    # model description
    print(resnet_model.summary())

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)

    # x_train = tf.expand_dims(x_train, axis = 1)

    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)

    # Fit the model
    start_time = time.time()
    resnet_history = resnet_model.fit(x_train, y_train, validation_data=(x_test, y_test),
                                    epochs=no_of_epochs, batch_size = 64)


    end_time = time.time()
    total_training_time = end_time - start_time
    total_training_time = total_training_time/3600
    val_accuracy = resnet_history.history['val_accuracy'][-1]

    print("Training Time(hrs)", total_training_time)
    print("Validation Accuracy",val_accuracy)
    #resnet_model.save('resnet50V2.h5')

    time_taken.append(total_training_time)
    acc.append(val_accuracy)


def run_script(n):


    for i in range(0,n):
        print("Round: ", i + 1)
        build_model(e)

    result['time_taken'] = time_taken
    result['accuracy'] = acc

    print(result)
    return  result

run_script(k)
