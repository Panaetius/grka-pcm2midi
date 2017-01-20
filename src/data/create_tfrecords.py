# load up some dataset. Could be anything but skdata is convenient.
import numpy as np
import tensorflow as tf
import random

writer = tf.python_io.TFRecordWriter("../../data/processed/data.tfrecords")
testwriter = tf.python_io.TFRecordWriter("../../data/processed/test.tfrecords")

data = open("../../data/raw/data.csv", "r")

lines = data.readlines()
random.shuffle(lines)

current_inactive_count = 0

for line in lines:
    values, labels = line.split(' ')
    values = list(map(float, values.split(',')))
    labels = list(map(int, labels.split(',')))

    if np.sum(labels) == 0:
        current_inactive_count += 1
        num = 1
    else:
        num = current_inactive_count
        current_inactive_count = 0

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(
            value=labels)),
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=values))}))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()


    # write the serialized object to disk, oversampling positives
    if random.random() < 0.15:
        testwriter.write(serialized)
    else:
        for i in range(0, num): #oversample
            writer.write(serialized)

writer.close()
testwriter.close()