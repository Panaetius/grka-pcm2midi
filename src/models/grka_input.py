"""Routine for decoding the grka binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np

# Process images of this size. Note that this differs from the original grka
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 2205

# Global constants describing the grka data set.
NUM_CLASSES = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 16000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_grka(filename_queue, data_dir):
    """Reads and parses examples from grka data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class grkaRecord(object):
        pass

    result = grkaRecord()

    # reader = tf.TextLineReader()
    # result.key, value = reader.read(filename_queue)
    # record_defaults = [[''], ['']]
    # values, keys = tf.decode_csv(value, field_delim=' ',
    #                                 record_defaults=record_defaults)
    #
    # record_defaults = np.zeros([IMAGE_SIZE, 1]).tolist()
    # image = tf.decode_csv(values, field_delim=',', record_defaults=record_defaults)
    #
    # record_defaults = np.zeros([NUM_CLASSES, 1]).tolist()
    # result.label = tf.decode_csv(keys, field_delim=',',
    #                       record_defaults=record_defaults)
    # #image = tf.image.decode_png(file_contents, channels=3)
    #
    # # The first bytes represent the label, which we convert from uint8->int32.
    # #result.label = tf.cast(label, tf.int32)
    #
    # # Convert from [depth, height, width] to [height, width, depth].
    # result.uint8image = tf.reshape(image, [IMAGE_SIZE, 1])
    reader = tf.TFRecordReader()
    result.key, value = reader.read(filename_queue)

    features = tf.parse_single_example(
        value,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([NUM_CLASSES], tf.int64),
            'data': tf.FixedLenFeature([IMAGE_SIZE], tf.float32)
        })

    result.label = features['label']
    fft1 = tf.fft(tf.complex(features['data'], 0.0))
    fft2 = tf.fft(tf.complex(tf.reshape(features['data'], [3, 735]),
                             0.0))
    fft3 = tf.fft(tf.complex(tf.reshape(features['data'], [5,441]),
                            0.0))
    out1 = tf.concat(0, [tf.complex_abs(fft1), atan2(fft1)])
    out2 = tf.reshape(tf.concat(1, [tf.complex_abs(fft2), atan2(fft2)]), [-1])
    out3 = tf.reshape(tf.concat(1, [tf.complex_abs(fft3), atan2(fft3)]), [-1])
    result.uint8image = tf.concat(0, [out1, out2, out3])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.audio('audio', images, 44100)
    tf.summary.image('images', tf.expand_dims(tf.reshape(images, [batch_size,
                                                                  13230,
                                                                  1]), 1),
                     max_outputs=16)

    return images, tf.reshape(label_batch, [batch_size, NUM_CLASSES])


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for grka training using the Reader ops.

    Args:
      data_dir: Path to the grka data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    #filenames = [os.path.join(data_dir, 'data.csv')]
    filenames = [os.path.join(data_dir, 'data.tfrecords')]
    # for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_grka(filename_queue, data_dir)
    distorted_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    #distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    #distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    #distorted_image = tf.image.random_brightness(distorted_image,
    #                                             max_delta=75)
    #distorted_image = tf.image.random_contrast(distorted_image,
    #                                           lower=0.2, upper=1.8)
    #distorted_image = tf.image.random_hue(distorted_image, max_delta=0.05)

    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d grka images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(distorted_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for grka evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the grka data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = [os.path.join(data_dir,
                                  'test.tfrecords')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir,
                                  'test.tfrecords')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    data_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_grka(data_queue, data_dir)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)


    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_whitening(reshaped_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(reshaped_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)

def atan2(c):
    x = tf.real(c)
    y = tf.imag(c)
    angle = tf.select(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
    angle = tf.select(tf.greater(y, 0.0), 0.5 * np.pi - tf.atan(x / y), angle)
    angle = tf.select(tf.less(y, 0.0), -0.5 * np.pi - tf.atan(x / y), angle)
    angle = tf.select(tf.less(x, 0.0), tf.atan(y / x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)),
                      np.nan * tf.zeros_like(x), angle)

    indices = tf.where(tf.less(angle, 0.0))
    updated_values = tf.gather_nd(angle, indices) + (2 * np.pi)
    update = tf.SparseTensor(indices, updated_values, angle.get_shape())
    update_dense = tf.sparse_tensor_to_dense(update)

    result = angle + update_dense

    return tf.select(tf.is_nan(result), tf.zeros_like(result), result)