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

import numpy as np
import tensorflow as tf

import grka

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/grka_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'validation',
                           """Either 'test' or 'validation'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/grka_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 600,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 8000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

tf.app.flags.DEFINE_float('dropout_keep_probability', 1.0,
                          "How many nodes to keep during dropout")


def eval_once(saver, summary_writer, logits, labels, num_classes, summary_op):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/grka_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1] \
                .split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_positives = np.zeros(shape=(num_classes))  # Counts the number of correct predictions.
            positives = np.zeros(shape=(num_classes))
            actual_positives = np.zeros(shape=(num_classes))
            true_count = 0
            true_count2 = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            precisions = np.zeros(shape=(num_classes))
            recalls = np.zeros(shape=(num_classes))
            tp = np.zeros(shape=(num_classes))

            while step < num_iter and not coord.should_stop():
                predictions, actual = sess.run([logits, labels])
                adj_predictions = np.round(1 / (1 + np.exp(
                    predictions)))
                true_positives += np.sum(np.bitwise_and(np.equal(adj_predictions,
                                                         actual)
                                        .astype(int), actual), axis=1)
                positives += np.sum(adj_predictions.astype(int), axis=1)
                actual_positives += np.sum(actual, axis=1)

                step += 1

            precisions += np.divide(true_positives, positives)
            recalls += np.divide(true_positives, actual_positives)
            tp = true_positives
            true_count = np.sum(true_positives)

            # Compute precision @ 1.
            precision = true_count / (total_sample_count * 128)
            print('%s: precision @ 1 = %.3f' % (datetime.now(),
                                                            precision))


            f1_scores = np.multiply(2.0, np.divide(np.multiply(precisions, recalls),
                                                   np.add(precisions, recalls)))

            print('precisions: ' + str(precisions))
            print('recalls: ' + str(recalls))
            print('f1: ' + str(f1_scores))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval grka for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for grka.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = grka.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = grka.inference(images)

        # Calculate predictions.
        #top_k_op = tf.nn.in_top_k(logits, labels, 1)
        #top_k_op2 = tf.nn.in_top_k(logits, labels, 3)
        # conf_matrix_op = tf.contrib.metrics.confusion_matrix(
        #     tf.argmax(logits, 1), labels,
        #     num_classes=grka.NUM_CLASSES)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            grka.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore,
                               write_version=tf.train.SaverDef.V2)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, logits, labels,
                      grka.NUM_CLASSES, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    # grka.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
