#!/usr/local/bin/python
#!/usr/bin/python

import argparse
import glob
import os.path
import time
import sys
import numpy as np
import tensorflow as tf

FLAGS = None
#TRAIN_FILE = glob.glob("/root/imagenet-data/train*")
TRAIN_FILE = 'train.tfrecords'
#glob.glob("/home/iki/master_project/cgd/0?/pcd*r.png")
VALIDATION_FILE = 'validation*'
'''
label='n03'
labels = np.array(['n02', 'n03', 'n04'])
y = [label == i for i in labels]
y=np.array(y)
y=y.astype(int)
print(y)

labels = ['n02666624', 'n02860415', 'n02880940', 
          'n02883344', 'n02908217', 'n02960690',
          'n03003091', 'n03147509', 'n03261776',
          'n03294833', 'n03438863', 'n03665924',
          'n03690938', 'n03793489', 'n03797390',
          'n03805725', 'n03848348', 'n03874599',
          'n03904909', 'n04148054', 'n04154938',
          'n04284002', 'n04303497', 'n04356056',
          'n04450749', 'n07607605']
'''
def read_and_decode(filename_queue):
    print('ok')
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    print('ok'*2)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    print('ok'*3)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5    
    image.set_shape([None, None, 3])
    label = tf.cast(features['label'], tf.int32)
    '''
    #work out the labels
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    label = key[31:40]
    image_raw = tf.image.decode_jpeg(value)
    image = tf.image.resize_images(image_raw, [224, 224])    
    return image, label
    '''
    print('ok'*10)
    return image, label

def inputs(train, batch_size, num_epochs):
    if not num_epochs: num_epochs = None
    filename = os.path.join(FLAGS.train_dir, TRAIN_FILE if train else VALIDATION_FILE)
    print(len(filename))
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)
        print('ok'*6)
        image, label = read_and_decode(filename_queue)
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000+3*batch_size,
            min_after_dequeue=1000)
        print('ok'*5)
        return images, sparse_labels

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op
    
def inference(images):
    return logits

def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return tf.reduce_mean(cross_entropy)

def run_training():
    with tf.Graph().as_default():
        images, labels = inputs(train=True,
                                batch_size=FLAGS.batch_size,
                                num_epochs=FLAGS.num_epochs)
        
        logits = inference(images)
        #loss = loss(logits, labels)
        loss = tf.reduce_mean(tf.nn.softmax_entropy_with_logits(labels, logits))
        train_op = training(loss, FLAGS.learning_rate)
        init_op = tf.group(tf.global_variables_initializer(),
			tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
			start_time = time.time()
			_, loss_value = sess.run([train_op, loss])

			duration = time.time() - start_time

			if step % 100 == 0:
				print('Step %d: loss = %.2f (%.3f sec)') % (step, loss_value, 
										duration)
			step +=1
	except tf.errors.OutOfRangeError:
		print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
	finally:
		coord.request_stop()

	coord.join(threads)
	sess.close()

def main(_):
    run_training()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/root/imagenet-data/',
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
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
