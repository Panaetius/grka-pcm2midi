"""Evaluation for grka.

Accuracy:
grka_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by grka_eval.py.

Speed:
On a single Tesla K40, grka_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the grka
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import sys

import numpy as np
import tensorflow as tf

import grka

import atexit

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '../../models/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_float('dropout_keep_probability', 1.0,
                          "How many nodes to keep during dropout")


class MLServer(object):
    def inference(self, data):
        images = np.reshape(data, (1, 4410))
        predictions = self.logits.eval(feed_dict={self.images: images},
                                       session=self.sess)
        result = np.argmax(predictions).item()
        if result < 128:
            print(np.array2string(self.softmax(predictions)) + " " + str(
                result),
                  file=sys.stderr)

        return result

    def setup(self):
        self.g = tf.Graph().as_default()
        # Get images and labels for grka.
        self.images = tf.placeholder(tf.float32, shape=(1, 4410),
                                     name="input_images")

        # Build a Graph that computes the logits predictions from the
        # inference model.
        self.logits = grka.inference(self.images, False)

        # Calculate predictions.
        # top_k_op = tf.nn.in_top_k(logits, labels, 1)
        # top_k_op2 = tf.nn.in_top_k(logits, labels, 3)
        # conf_matrix_op = tf.contrib.metrics.confusion_matrix(
        #     tf.argmax(logits, 1), labels,
        #     num_classes=grka.NUM_CLASSES)

        # Restore the moving average version of the learned variables for eval.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     grka.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(
                               write_version=tf.train.SaverDef.V2)

        self.sess = tf.Session()

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            self.sess.run(tf.initialize_all_variables())
            saver.restore(self.sess, FLAGS.checkpoint_dir +
            os.path.basename(
                ckpt.model_checkpoint_path))
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/grka_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1] \
                .split('-')[-1]
        else:
            print('No checkpoint file found')
            return

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def main(argv=None):
    np.set_printoptions(edgeitems=128)
    serv = MLServer()

    serv.setup()

    atexit.register(teardown, serv.sess, serv.g)

    print("ready")

    print("sent ready: ", file=sys.stderr)

    while(True):
        entry = input()
        values = np.fromstring(entry, dtype=np.float32, sep=',')

        # print("got value: " + np.array_str(values), file=sys.stderr)

        result = serv.inference(values)

        # print("calculated: " + str(result), file=sys.stderr)

        print(result)

        # print("Wrote Result: ", file=sys.stderr)

def teardown(session, graph):
    session.close()
    graph.close()


if __name__ == '__main__':
    tf.app.run()
