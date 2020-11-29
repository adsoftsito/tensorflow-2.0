import warnings
import logging, os

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras

# create a Tensor
hello = tf.constant("hello keras world...")

# to acces a Tensor value, call numpy()
print(hello.numpy())

dataset = keras.preprocessing.text_dataset_from_directory(
    '/home/adsoft/tensorflow-2.0/keras/text/texts', batch_size=64)

for data, labels in dataset:
    print(data.shape)
    print(data.dtype)
    print(labels.shape)
    print(labels.dtype)
