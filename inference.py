'''
Inference model for grasping
'''
import tensorflow as tf

def conv2d_s2(x, W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

def conv2d_s1(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def inference(images):
    w1 = tf.get_variable('w1', shape=[5,5,3,64])
    b1 = tf.get_variable('b1', initializer=tf.constant(0.1, shape=[64]))
    h1 = tf.nn.relu(conv2d_s2(images, w1)+b1)
    h1_pool = max_pool_2x2(h1)
    
    w2 = tf.get_variable('w2', [3,3,64,128])
    b2 = tf.get_variable('b2', initializer=tf.constant(0.1, shape=[128]))
    h2 = tf.nn.relu(conv2d_s2(h1_pool,w2)+b2)
    h2_pool = max_pool_2x2(h2)

    w3 = tf.get_variable('w3', [3,3,128,128])
    b3 = tf.get_variable('b3', initializer=tf.constant(0.1, shape=[128]))
    h3 = tf.nn.relu(conv2d_s1(h2_pool,w3)+b3)
    
    w4 = tf.get_variable('w4', [3,3,128,128])
    b4 = tf.get_variable('b4', initializer=tf.constant(0.1, shape=[128]))
    h4 = tf.nn.relu(conv2d_s1(h3,w4)+b4)

    w5 = tf.get_variable('w5', [3,3,128,256])
    b5 = tf.get_variable('b5', initializer=tf.constant(0.1, shape=[256]))
    h5 = tf.nn.relu(conv2d_s1(h4,w5)+b5)
    h5_pool = max_pool_2x2(h5)
    
    w_fc1 = tf.get_variable('w_fc1', [7*7*256,512])
    b_fc1 = tf.get_variable('b_fc1', initializer=tf.constant(0.1, shape=[512]))
    h5_flat = tf.reshape(h5_pool, [-1, 7*7*256])
    h_fc1 = tf.nn.relu(tf.matmul(h5_flat,w_fc1)+b_fc1)
    
    w_fc2 = tf.get_variable('w_fc2', [512,512])
    b_fc2 = tf.get_variable('b_fc2', initializer=tf.constant(0.1, shape=[512]))
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)
    
    w_output = tf.get_variable('w_output', [512, 1000])
    b_output = tf.get_variable('b_output', [1000])
    output = tf.matmul(h_fc2,w_output)+b_output
    
    return output
