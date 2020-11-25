from __future__ import print_function

import warnings
import logging, os

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

# Graph
a = tf.Variable(2, name='a')
b = tf.Variable(3, name='b')


@tf.function  # tf.function allows us to take a graph from a function
def graph_to_visualize(a, b):
    c = tf.add(a, b, name='Add')

# Visualize
writer = tf.summary.create_file_writer('./graphs')

with writer.as_default():
    graph = graph_to_visualize.get_concrete_function(a, b).graph # get graph from function
    summary_ops_v2.graph(graph.as_graph_def()) # visualize

writer.close()

