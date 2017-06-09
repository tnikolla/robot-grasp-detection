'''
Inference model for grasping
'''
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d_s2(x, W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

def conv2d_s1(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
def inference(images):
    
    w1 = weight_variable([5,5,3,64])
    b1 = bias_variable([64])
    h1 = tf.nn.relu(conv2d_s2(images, w1)+b1)
    h1_pool = max_pool_2x2(h1)
    print("h1_pool: {}".format(h1_pool.get_shape()))
    
    w2 = weight_variable([3,3,64,128])
    b2 = bias_variable([128])
    h2 = tf.nn.relu(conv2d_s2(h1_pool,w2)+b2)
    h2_pool = max_pool_2x2(h2)
    #print("h2_pool: {}".format(h2_pool.get_shape()))

    w3 = weight_variable([3,3,128,128])
    b3 = bias_variable([128])
    h3 = tf.nn.relu(conv2d_s1(h2_pool,w3)+b3)
    #print("h3: {}".format(h3.get_shape()))
    
    #w4 = weight_variable([3,3,128,128])
    #b4 = bias_variable([128])
    #h4 = tf.nn.relu(conv2d_s1(h3,w4)+b4)
    #print("h4: {}".format(h4.get_shape()))
    
    w5 = weight_variable([3,3,128,256])
    b5 = bias_variable([256])
    h5 = tf.nn.relu(conv2d_s1(h3,w5)+b5)
    h5_pool = max_pool_2x2(h5)
    #print("h5_pool: {}".format(h5_pool.get_shape()))
    
    w_fc1 = weight_variable([7*7*256,512])
    b_fc1 = bias_variable([512])
    h5_flat = tf.reshape(h5_pool, [-1, 7*7*256])
    h_fc1 = tf.nn.relu(tf.matmul(h5_flat,w_fc1)+b_fc1)
    #print("h_fc1: {}".format(h_fc1.get_shape()))
    
    w_fc2 = weight_variable([512,512])
    b_fc2 = bias_variable([512])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)
    #print("h_fc2: {}".format(h_fc2.get_shape()))    
    
    w_output = weight_variable([512,1000])
    b_output = bias_variable([1000])
    output = tf.nn.relu(tf.matmul(h_fc2,w_output)+b_output)
    #print("output: {}".format(output.get_shape()))
    
    return output