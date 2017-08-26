#!/usr/local/bin/python
''' Training a network on Imagenet.
'''
import sys
import argparse
import os.path
import glob
import tensorflow as tf
import image_processing
from inference import inference
import time

TRAIN_FILE = '/root/imagenet-data/train-00001-of-01024'
VALIDATION_FILE = '/root/imagenet-data/validation-00004-of-00128'

def data_files():
    tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % FLAGS.train)
    data_files = tf.gfile.Glob(tf_record_pattern)
    return data_files

def run_training():
    tf.reset_default_graph()
    data_files_ = TRAIN_FILE
    #data_files_ = VALIDATION_FILE
    #data_files_ = data_files()
    images, labels = image_processing.distorted_inputs(
        [data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size)
    labels = tf.one_hot(labels, 1000)   
    logits = inference(images)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    tf.summary.scalar('loss', loss)
    correct_pred = tf.equal(tf.arg_max(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    global_step = tf.Variable(0, trainable=False)
    lr=tf.train.exponential_decay(FLAGS.learning_rate, 
                                  global_step=global_step, 
                                  decay_steps=1200000,
                                  decay_rate=0.8,
                                  staircase=True)
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #save/restore model
    #saver = tf.train.Saver({'w1': w1, 'b1':b1, 'w2':w2, 'b2':b2, 'w3':w3, 'b3':b3, 'w4':w4, 'b4':b4, 'w5':w5, 'b5':b5, 'w_fc1':w_fc1, 'b_fc1':b_fc1, 'w_fc2':w_fc2, 'b_fc2':b_fc2})
    saver = tf.train.Saver({'w1': w1})
    #saver = tf.train.Saver()
    #saver.restore(sess, FLAGS.model_path)
    try:
        step = 0
        start_time = time.time()
        while not coord.should_stop():
            start_batch = time.time()
            #train                
            _, loss_value, pred, acc = sess.run(
                [train_op, loss, correct_pred, accuracy])
            duration = time.time() - start_batch
            if step % 10 == 0:             
                print('Step %d | loss = %.2f | accuracy = %.2f (%.3f sec/batch)')%(
                step, loss_value, acc, duration)
            if step % 500 == 0:
                summary = sess.run(merged_summary_op)
                summary_writer.add_summary(summary, step*FLAGS.batch_size)
            if step % 5000 == 0:
                saver.save(sess, FLAGS.model_path)
                                
            step +=1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps, %.1f min.' % (FLAGS.num_epochs, step, (time.time()-start_time)/60))
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
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/root/imagenet-data',
        help='Directory with training data.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
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
        default='/tmp/tf',
        help='Tensorboard log_dir.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='/tmp/tf/model.ckpt',
        help='Variables for the model.'
    )
    parser.add_argument(
        '--train',
        type=str,
        default='train',
        help='Train or evaluate the dataset'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
