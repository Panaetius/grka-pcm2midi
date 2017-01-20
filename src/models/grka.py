"""Builds the grka network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf
import numpy as np
from scipy.sparse import hstack, vstack, coo_matrix
import math as m

import grka_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', os.path.join(os.path.dirname(__file__),
                                                    os.pardir, os.pardir,
                                                    'data', 'processed'),
                           """Path to the grka data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the grka data set.
IMAGE_SIZE = grka_input.IMAGE_SIZE
NUM_CLASSES = grka_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = grka_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = grka_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.003  # Initial learning rate. 0.00007
WEIGHT_DECAY = 0.0015
ADAM_EPSILON = 0.0001

hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(4410) / 4410)
hamming2 = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(2205) / 2205)

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/grka-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, connections, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.xavier_initializer(dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """Construct distorted input for grka training using the Reader ops.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    images, labels = grka_input.distorted_inputs(data_dir=data_dir,
                                                 batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for grka evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    images, labels = grka_input.inputs(eval_data=eval_data,
                                       data_dir=data_dir,
                                       batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images, isTraining=False):
    """Build the grka model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU
    # training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().

    spect = tf.complex_abs(tf.fft(tf.complex(images * hamming, tf.zeros_like(
        images))))

    window_width = 2205
    shift = 735

    data =                      [tf.reshape(tf.complex(images[:,
                                                       i:i + window_width] *
                                                       hamming2,
                                                       tf.zeros_like(images[:,
                                                       i:i + window_width])),
                                 [FLAGS.batch_size, window_width])
                      for i in range(0, IMAGE_SIZE - window_width + 1, shift)]

    data = tf.reshape(tf.transpose(tf.complex_abs(tf.fft(data)), [1, 0, 2]),
                      [FLAGS.batch_size, 2205 * 4])

    spect = tf.concat(1, [spect, data])

    tf.summary.image('images',
                     tf.reshape(spect, [FLAGS.batch_size, 1, 13230, 1]),
                     max_outputs=16)

    # local3
    with tf.variable_scope('local1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # reshape = tf.reshape(spect, [FLAGS.batch_size, -1])
        dim = spect.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 13230],
                                              connections=13230 + 13230,
                                              wd=WEIGHT_DECAY)
        bn1 = batch_norm_wrapper(tf.matmul(spect, weights),
                                 is_training=isTraining)
        local1 = tf.nn.elu(bn1, name=scope.name)
        local1 = tf.nn.dropout(local1, FLAGS.dropout_keep_probability)
        _activation_summary(local1)

    # local3
    with tf.variable_scope('local2') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # reshape = tf.reshape(spect, [FLAGS.batch_size, -1])
        weights = _variable_with_weight_decay('weights', shape=[13230, 8820],
                                              connections=13230 + 8820,
                                              wd=WEIGHT_DECAY)
        bn2 = batch_norm_wrapper(tf.matmul(local1, weights),
                                 is_training=isTraining)
        local2 = tf.nn.elu(bn2, name=scope.name)
        local2 = tf.nn.dropout(local2, FLAGS.dropout_keep_probability)
        _activation_summary(local2)

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # reshape = tf.reshape(spect, [FLAGS.batch_size, -1])
        dim = spect.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[8820, 4410],
                                              connections=8820 + 4410,
                                              wd=WEIGHT_DECAY)
        bn3 = batch_norm_wrapper(tf.matmul(local2, weights),
                                 is_training=isTraining)
        local3 = tf.nn.elu(bn3, name=scope.name)
        local3 = tf.nn.dropout(local3, FLAGS.dropout_keep_probability)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4410, 4410],
                                              connections=4410 + 4410,
                                              wd=WEIGHT_DECAY)
        bn4 = batch_norm_wrapper(tf.matmul(local3, weights),
                                 is_training=isTraining)
        local4 = tf.nn.elu(bn4, name=scope.name)
        local4 = tf.nn.dropout(local4, FLAGS.dropout_keep_probability)
        _activation_summary(local4)

        # local5
    with tf.variable_scope('local5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4410, 128],
                                              connections=4410 + 128,
                                              wd=WEIGHT_DECAY)
        bn5 = batch_norm_wrapper(tf.matmul(local4, weights),
                                 is_training=isTraining)
        local5 = tf.nn.elu(bn5, name=scope.name)
        local5 = tf.nn.dropout(local5, FLAGS.dropout_keep_probability)
        _activation_summary(local5)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [128, NUM_CLASSES],
                                              connections=128 + NUM_CLASSES,
                                              wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local5, weights), biases,
                                name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    sig_logits = tf.nn.sigmoid(logits)
    sigmoid_logits = tf.round(sig_logits)
    # sigmoid_logits = tf.Print(sigmoid_logits, [sigmoid_logits],
    #                           message="sigmoid_logits: ", summarize=128)
    # labels = tf.Print(labels, [labels], message="labels: ", summarize=128)
    labels = tf.cast(labels, tf.float32)

    correct_prediction = tf.equal(sigmoid_logits, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection('accuracies', accuracy)

    # curr_conf_matrix = tf.cast(
    #     tf.contrib.metrics.confusion_matrix(sigmoid_logits, labels,
    #                                         num_classes=NUM_CLASSES),
    #     tf.float32)
    # conf_matrix = tf.get_variable('conf_matrix', dtype=tf.float32,
    #                               initializer=tf.zeros(
    #                                   [NUM_CLASSES, NUM_CLASSES],
    #                                   tf.float32),
    #                               trainable=False)
    #
    # # make old values decay so early errors don't distort the confusion matrix
    # conf_matrix.assign(tf.mul(conf_matrix, 0.97))
    #
    # conf_matrix = conf_matrix.assign_add(curr_conf_matrix)
    #
    # tf.summary.image('Confusion Matrix',
    #                  tf.reshape(tf.clip_by_norm(conf_matrix, 1, axes=[0]),
    #                             [1, NUM_CLASSES, NUM_CLASSES, 1]))

    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    #    logits, labels, name='cross_entropy_per_example')
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
        logits, labels, 128.0, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    count = tf.tile(
        tf.reshape((128.0 / tf.clip_by_value(tf.reduce_sum(labels, 1), 1,
                                             128) - 1.0),
                   [FLAGS.batch_size, 1]),
        tf.pack([1, 128]))
    weight = tf.add(tf.mul(labels, count), 1)
    # weight = tf.Print(weight,[weight],"Weights:",
    # summarize=128*FLAGS.batch_size)

    # cross_entropy_mean = tf.contrib.losses.log_loss(sig_logits,
    #                                                 labels, weight=weight)

    tf.add_to_collection('cross_entropy', cross_entropy_mean)

    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def batch_norm_wrapper(inputs, is_training, decay=0.999):
    epsilon = 1e-3
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale,
                                             epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale,
                                         epsilon)


def _add_loss_summaries(total_loss):
    """Add summaries for losses in grka model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('cross_entropy')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    accuracies = tf.get_collection('accuracies')
    for a in accuracies:
        tf.scalar_summary('accuracy', a)

        # Attach a scalar summary to all individual losses and the total loss;
        # do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the
        # loss as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    # for l in losses:
    #     tf.scalar_summary('losses (raw)', l)

    return  # loss_averages_op


def train(total_loss, global_step):
    """Train grka model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    # loss_averages_op =
    _add_loss_summaries(total_loss)

    # Compute gradients.
    # with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(lr, epsilon=ADAM_EPSILON)
    grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def put_kernels_on_grid(kernel, grid, pad=1):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):shape of the grid. Require: NumKernels == grid_Y * grid_X
                        User is responsible of how to break into two multiples.
      pad:              number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    """
    grid_Y, grid_X = grid
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 1]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 1]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    # x_min = tf.reduce_min(x7)
    # x_max = tf.reduce_max(x7)
    # x8 = (x7 - x_min) / (x_max - x_min)

    return x7


def put_activations_on_grid(activations, grid, pad=1):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):shape of the grid. Require: NumKernels == grid_Y * grid_X
                        User is responsible of how to break into two multiples.
      pad:              number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    """
    grid_Y, grid_X = grid
    # get first image in batch to make things simpler
    activ = activations[1, :]
    # greyscale
    activ = tf.expand_dims(activ, 2)
    # pad X and Y
    x1 = tf.pad(activ, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

    # X and Y dimensions, w.r.t. padding
    Y = activ.get_shape()[0] + pad
    X = activ.get_shape()[1] + pad

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 1]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 1]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    # x_min = tf.reduce_min(x7)
    # x_max = tf.reduce_max(x7)
    # x8 = (x7 - x_min) / (x_max - x_min)

    # return x8
    return x7
