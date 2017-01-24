# load up some dataset. Could be anything but skdata is convenient.
import numpy as np
import tensorflow as tf
import random

writer = tf.python_io.TFRecordWriter("../../data/processed/data.tfrecords")
testwriter = tf.python_io.TFRecordWriter("../../data/processed/test.tfrecords")

data = open("../../data/raw/data.csv", "r")

lines = data.readlines()
random.shuffle(lines)

bad_classes = [24, 25, 26, 27, 29, 30, 31, 34, 39, 44, 46, 82, 83, 88, 90, 92,
               93, 95, 96, 97, 100]

current_inactive_count = 0

train = []
test = []

for line in lines:
    values, labels = line.split(' ')
    values = list(map(float, values.split(',')))
    labels = list(map(int, labels.split(',')))

    if np.sum(labels) == 0:
        current_inactive_count += 1
        num = 1
    else:
        num = 4 * current_inactive_count
        current_inactive_count = 0

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(
            value=labels)),
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=values))}))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()

    if np.argmax(labels).item() in bad_classes:
        num = num * 2 #oversample labels with high error rate more

    # # write the serialized object to disk, oversampling positives
    # if random.random() < 0.15:
    #     test.append(serialized)
    # else:
    for i in range(0, num): #oversample
        train.append(serialized)
random.shuffle(train)

for example in train:
    writer.write(example)

writer.close()

# random.shuffle(test)
#
# for example in test:
#     testwriter.write(example)
#
# testwriter.close()