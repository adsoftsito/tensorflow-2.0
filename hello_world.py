import warnings
import logging, os

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# create a Tensor
hello = tf.constant("hello world")
print(hello)

# to acces a Tensor value, call numpy()
print(hello.numpy())
