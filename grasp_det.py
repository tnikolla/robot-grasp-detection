#!/usr/local/bin/python
''' 
Training a network on cornell grasping dataset for detecting grasping positions.
'''
import sys
import argparse
import os.path
import glob
import tensorflow as tf
import numpy as np
import grasp_img_proc
from grasp_inf import inference
import time

TRAIN_FILE = '/root/dataset/cornell_grasping_dataset/train-cgd'
#VALIDATION_FILE = '/root/imagenet-data/validation-00004-of-00128'

def data_files():
    tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % FLAGS.train)
    data_files = tf.gfile.Glob(tf_record_pattern)
    return data_files

def intersection_over_union(bbox_pred, bbox):
    # x are the coo. of the parallel rotated predicted bbox
    x1 = [bbox_pred[0]-0.5*bbox_pred[5], bbox_pred[1]-0.5*bbox_pred[4]]
    x2 = [bbox_pred[0]+0.5*bbox_pred[5], bbox_pred[1]-0.5*bbox_pred[4]]
    x3 = [bbox_pred[0]+0.5*bbox_pred[5], bbox_pred[1]+0.5*bbox_pred[4]]
    x4 = [bbox_pred[0]-0.5*bbox_pred[5], bbox_pred[1]+0.5*bbox_pred[4]]
    # y are the coo. of the parallel rotated GT bbox
    origin = [(bbox[5]-bbox[1])/2, (bbox[4]-bbox[0])/2]
    w = tf.sqrt(tf.pow(bbox[2]-bbox[0],2) + tf.pow(bbox[3]-bbox[1],2))
    h = tf.sqrt(tf.pow(bbox[6]-bbox[0],2) + tf.pow(bbox[7]-bbox[1],2))
    y1 = [origin[0]-w/2, origin[1]-h/2]
    y2 = [origin[0]+w/2, origin[1]-h/2]
    y3 = [origin[0]-w/2, origin[1]+h/2]
    y4 = [origin[0]+w/2, origin[1]+h/2]
    intersection = tf.maximum(0., tf.minimum(x2[0], y2[0]) - tf.maximum(x1[0], y1[0])) * \
                   tf.maximum(0., tf.minimum(x4[1], y4[1]) - tf.maximum(x4[1], y4[1]))
    iou = intersection / (bbox_pred[5]*bbox_pred[4] + w*h - intersection)
    return iou

def elem(tensor,element):
    bs = FLAGS.batch_size
    return tf.slice(tensor, [0,element], [bs,1])

'''def bboxes_to_grasps(bboxes):
    x = elem(bboxes, 5) / tf.constant(2.) # check this
    y = elem(bboxes, 4) / tf.constant(2.)
    tan = (elem(bboxes, 3) -elem(bboxes, 1)) / (elem(bboxes, 2) -elem(bboxes, 0))
    h = tf.sqrt(tf.pow(elem(bboxes,6) -elem(bboxes,0), 2) +tf.pow(elem(bboxes, 7) -elem(bboxes, 1),2))
    w = tf.sqrt(tf.pow(elem(bboxes,2) -elem(bboxes,0), 2) +tf.pow(elem(bboxes, 3) -elem(bboxes, 1),2))
    return x, y, tan, h, w'''

def bboxes_to_grasps(bboxes):
    # converting and scaling
    box = tf.unstack(bboxes, axis=1)
    x = (box[0] + (box[4] - box[0])/2) * 0.35
    y = (box[1] + (box[5] - box[1])/2) * 0.47
    tan = (box[3] -box[1]) / (box[2] -box[0]) *0.47/0.35
    h = tf.sqrt(tf.pow((box[2] -box[0])*0.35, 2) + tf.pow((box[3] -box[1])*0.47, 2))
    w = tf.sqrt(tf.pow((box[6] -box[0])*0.35, 2) + tf.pow((box[7] -box[1])*0.47, 2))
    return x, y, tan, h, w 
    
def run_training():
    #tf.reset_default_graph()
    data_files_ = TRAIN_FILE
    #data_files_ = VALIDATION_FILE
    #data_files_ = data_files()
    images, bboxes = grasp_img_proc.distorted_inputs(
        [data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size)
    print('images: %s' %images.get_shape())
    print('bboxes: %s' %bboxes.get_shape())
    # grasp is in the form: g = {x, y, tan(theta), h, w}    
    x, y, tan, h, w = bboxes_to_grasps(bboxes) # list
    x_hat, y_hat, tan_hat, h_hat, w_hat = tf.unstack(inference(images), axis=1) # list
    # tangent of 85 degree is 11 
    tan_hat_confined = tf.minimum(11., tf.maximum(-11., tan_hat))
    tan_confined = tf.minimum(11., tf.maximum(-11., tan))
    # Loss function
    gamma = tf.constant(100.)
    loss = tf.reduce_sum(tf.pow(x_hat -x, 2) +tf.pow(y_hat -y, 2) + gamma*tf.pow(tan_hat_confined - tan_confined, 2) +tf.pow(h_hat -h, 2) +tf.pow(w_hat -w, 2))
    tf.summary.scalar('loss', loss)
    #accuracy = theta# iou #tf.reduce_mean(iou)
    #tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    train_op = tf.train.AdamOptimizer(epsilon=0.1).minimize(loss)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #save/restore model
    #d={}
    #l = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2']
    #for i in l:
    #    d[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]
    
    dg={}
    lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
    for i in lg:
        dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]

    saver = tf.train.Saver(dg)
    #saver_g = tf.train.Saver(dg)
    saver.restore(sess, FLAGS.model_path)
    try:
        step = 0
        start_time = time.time()
        while not coord.should_stop():
            start_batch = time.time()
            #train             
            _, loss_value, x_value, x_model, tan_value, tan_model, h_value, h_model, w_value, w_model = sess.run(
                [train_op, loss, x, x_hat, tan, tan_hat_confined, h, h_hat, w, w_hat])
            duration = time.time() - start_batch
            if step % 100  == 0:             
                print('Step %d | loss = %s\n | x = %s\n | x_hat = %s\n | tan = %s\n | tan_hat = %s\n | h = %s\n | h_hat = %s\n | w = %s\n | w_hat = %s\n | (%.3f sec/batch\n')%(
                        step, loss_value, x_value[:3], x_model[:3], tan_value[:3], tan_model[:3], h_value[:3], h_model[:3], w_value[:3], w_model[:3], duration)
            #if step % 100 == 0:
            #    summary = sess.run(merged_summary_op)
            #    summary_writer.add_summary(summary, step*FLAGS.batch_size)
            if step % 1000 == 0:
                saver.save(sess, FLAGS.model_path)
                                
            step +=1
            #time.sleep(0.5)
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
        default=64,
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
