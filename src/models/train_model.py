"""A binary to train grka using a single GPU.

Accuracy:
grka_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by grka_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the grka
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# from tensorflow.models.image.grka import grka
import grka

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/grka_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('dropout_keep_probability', 0.5,
                          """How many nodes to keep during dropout""")


def train():
    """Train grka for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        conf_matrix = tf.Variable(tf.zeros([grka.NUM_CLASSES,
                                            grka.NUM_CLASSES],
                                           tf.float32),
                                  name='conf_matrix',
                                  trainable=False)

        # Get images and labels for grka.
        images, labels = grka.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = grka.inference(images, True)

        # Calculate loss.
        loss = grka.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = grka.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(),
                               write_version=tf.train.SaverDef.V2)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        # Start running operations on the Graph.
        sess = tf.Session(config=config)
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                prediction = tf.round(tf.nn.sigmoid(logits))
                correct_prediction = tf.equal(prediction,
                                              tf.cast(labels, tf.float32))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                  tf.float32))

                lab = tf.cast(labels, tf.float32)
                tp = tf.reduce_sum(tf.mul(prediction, lab))
                p = tf.reduce_sum(prediction)
                l = tf.reduce_sum(lab)
                precision = tf.div(tp,p)
                recall = tf.div(tp,l)
                f1 = tf.mul(2.0, tf.div(tf.mul(precision, recall), tf.add(
                    precision, recall)))

                train_pred = sess.run(prediction)
                train_corrpred = sess.run(correct_prediction)
                train_lab = sess.run(lab)
                train_tp = sess.run(tp)
                train_p = sess.run(p)
                train_l = sess.run(l)
                train_acc = sess.run(accuracy)
                tf.scalar_summary('accuracy', accuracy)

                train_prec = sess.run(precision)
                tf.summary.scalar('precision', precision)
                train_rec = sess.run(recall)
                tf.summary.scalar('recall', recall)
                train_f1 = sess.run(f1)
                tf.summary.scalar('f1', f1)

                format_str = ('%s: step %d, loss = %.2f, accuracy = %.3f '
                              'precision = %.3f, recall = %.3f, f1 = %.3f '
                              '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, train_acc,
                                    train_prec, train_rec, train_f1,
                                    examples_per_sec, sec_per_batch))

            if step % 25 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
