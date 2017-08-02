#!/usr/local/bin/python
#!/usr/bin/python

import argparse
import glob
import os.path
import time
import sys
import numpy as np
import inference
import inference_redmon
import tensorflow as tf
FLAGS = None
TRAIN_FILE = '/root/imagenet-data/train-00004-of-01024'
VALIDATION_FILE = '/root/imagenet-data/validation-00127-of-00128'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def data_files():
    tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % 'train')
    data_files = tf.gfile.Glob(tf_record_pattern)
    return data_files

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], tf.string, default_value='')
    }
    features = tf.parse_single_example( serialized_example, feature_map)
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [224,224])
    label = tf.cast(features['image/class/label'], tf.int32)
    return image, label

def inputs(train, batch_size, num_epochs):
    if not num_epochs: num_epochs = None
    data_files_ = data_files()
    filename_queue = tf.train.string_input_producer(
        data_files_, num_epochs=num_epochs)
    image, label = read_and_decode(filename_queue)
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=4,
        capacity=1000+3*batch_size,
        min_after_dequeue=1000)
    # -- 1 --
    # images = image_processing.distorted_inputs(images)
    images = image_processing.distorted_inputs(images)
    return images, tf.reshape(sparse_labels, [batch_size])

def run_training():
    images, labels = inputs(train=True,
                            batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)
    # -- 2 --        
    #images = distorted_inputs(images)
    labels_one = tf.one_hot(labels, 1000)
    logits = inference_redmon.inference(images)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels_one))        
    tf.summary.scalar('loss', loss)
    correct_pred = tf.equal( tf.argmax(logits,1), tf.argmax(labels_one,1))
    accuracy = tf.reduce_mean( tf.cast( correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    global_step=tf.Variable(0,trainable=False)
    lr=tf.train.exponential_decay(FLAGS.learning_rate, 
                                  global_step=global_step, 
                                  decay_steps=800000,
                                  decay_rate=0.16,
                                  staircase=True)
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #restore model        
    #saver.restore(sess, FLAGS.model_path)        
    try:
        step = 0
        start_time = time.time()
        while not coord.should_stop():
            start_batch = time.time()
            #train                
            _, loss_value, pred, acc, summary = sess.run(
            [train_op, loss, correct_pred, accuracy,  merged_summary_op])
            #evaluate
            #loss_value, pred, acc, summary = sess.run(
            #    [loss, correct_pred, accuracy,  merged_summary_op])
            summary_writer.add_summary(summary, step*FLAGS.batch_size)
            duration = time.time() - start_batch
            if step % 10 == 0:
                print('Step %d | loss = %.2f | accuracy = %.2f (%.3f sec/batch)')%(
                step, loss_value, acc, duration)
            step +=1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps, %.1f min.' % (FLAGS.num_epochs, step, (time.time()-start_time)/60))
    finally:
        coord.request_stop()

    saver.save(sess, FLAGS.model_path)
    print('Model saved in: %s' % FLAGS.model_path)
    #print('Evaluated from restored variables from: %s' % FLAGS.model_path)
    coord.join(threads)
    sess.close()

def main(_):
    run_training()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--data_dir',
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
        default=32,
        help='Batch size.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tf/exp',
        help='Tensorboard log_dir.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='/tmp/model/model.ckpt',
        help='Variables for the model.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
