#!/usr/local/bin/python
#!/usr/bin/python

import argparse
import os.path
import time
import sys
import numpy as np
import tensorflow as tf

FLAGS = None
TRAIN_FILE = ''#glob
VALIDATION_FILE = ''
'''
label='n03'
labels = np.array(['n02', 'n03', 'n04'])
y = [label == i for i in labels]
y=np.array(y)
y=y.astype(int)
print(y)
'''
labels = ['n02666624', 'n02860415', 'n02880940', 
          'n02883344', 'n02908217', 'n02960690',
          'n03003091', 'n03147509', 'n03261776',
          'n03294833', 'n03438863', 'n03665924',
          'n03690938', 'n03793489', 'n03797390',
          'n03805725', 'n03848348', 'n03874599',
          'n03904909', 'n04148054', 'n04154938',
          'n04284002', 'n04303497', 'n04356056',
          'n04450749', 'n07607605']
def read_and_decode(filename_queue):
    #work out the labels
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    label = key[31:40]
    image_raw = tf.image.decode_jpeg(value)
    image = tf.image.resize_images(image_raw, [224, 224])    
    return image, label
    
def inputs(train, batch_size, num_epochs):
    if not num_epochs: num_epochs = None
    filename = os.path.join(FLAGS.train_dir,
                            TRAIN_FILE if train else VALIDATION_FILE)
                            
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)
        image, label = read_and_decode(filename_queue)
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000+3*batch_size,
            min_after_dequeue=1000)
        return images, sparse_labels

def run_training():
    with tf.Graph().as_default():
        images, labels = inputs(train=True,
                                batch_size=FLAGS.batch_size,
                                num_epochs=FLAGS.num_epochs)
        logits = inference
def main(_):
    run_taining()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/dataset/imagenet_synsets',
        help='Directory with training data.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.'
    )
    FLAGS, unparsed = parser.arse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)