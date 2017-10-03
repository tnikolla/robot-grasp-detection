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

def convert_to_bbox(grasp):
    x=grasp[0]
    y=grasp[1]
    h=grasp[4]
    w=grasp[5]
    theta = 0.5*np.arcsin(grasp[2])
    w2c=0.5*w*np.cos(theta)
    w2s=0.5*w*np.sin(theta)
    h2c=0.5*h*np.cos(theta)
    h2s=0.5*h*np.sin(theta)
    x3=x +w2c -h2s    
    x1=x -w2c +h2s
    x2=x +w2c +h2s    
    x4=x -w2c -h2s
    y3=y +w2s +h2c   
    y1=y -w2s -h2c   
    y2=y +w2s -h2c   
    y4=y -w2s +h2c   
    return [x1,y1,x2,y2,x3,y3,x4,y4]

def clip(subjectPolygon, clipPolygon):
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
   return(outputList)

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def intersection_over_union(bbox_hat, bbox):
    intersection_polygon = clip(bbox_hat, bbox)
    return PolygonArea(intersection_polygon)
    
def run_training():
    #tf.reset_default_graph()
    data_files_ = TRAIN_FILE
    #data_files_ = VALIDATION_FILE
    #data_files_ = data_files()
    #images, labels = image_processing.distorted_inputs(
    #    [data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size)   
    images, bbox = grasp_img_proc.distorted_inputs(
        [data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size)
    grasp = inference(images)
    # tf.py_func
    bbox_hat = tf.py_func(convert_to_bbox, [grasp], [tf.float32])
    iou = tf.py_func(intersection_over_union, [bbox_hat, bbox], tf.float32)[0]
    loss = tf.negative(tf.log(iou)) #check this
    tf.summary.scalar('loss', loss)
    accuracy = tf.reduce_mean(iou)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    train_op = tf.train.AdamOptimizer(epsilon=0.1).minimize(loss)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #save/restore model
    d={}
    l = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
    for i in l:
        d[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]
    saver = tf.train.Saver(d)
    saver.restore(sess, FLAGS.model_path)
    try:
        step = 0
        start_time = time.time()
        while not coord.should_stop():
            start_batch = time.time()
            #train             
            _, loss_value, acc = sess.run(
                [train_op, loss, accuracy])
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
