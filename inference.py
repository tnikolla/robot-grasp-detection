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
    # conv1 pool1 layer
    W_conv1 = weight_variable([5,5,3,64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d_s2(images, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #print(h_pool1.get_shape())
    
    # conv2 pool layer
    W_conv2 = weight_variable([3,3,64,128])
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d_s2(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    #print(h_pool2.get_shape())
    
    # conv3 pool layer
    W_conv3 = weight_variable([3,3,128,256])
    b_conv3 = bias_variable([256])
    h_conv3 = tf.nn.relu(conv2d_s1(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    #print(h_pool3.get_shape())
    # fc1 layer
    W_fc1 = weight_variable([7*7*256, 512])
    b_fc1 = bias_variable([512])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 7*7*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    #print("h_fc1: {}".format(h_fc1.get_shape()))

    # fc1 layer
    W_fc2 = weight_variable([512, 512])
    b_fc2 = bias_variable([512])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    #print("h_fc2: {}".format(h_fc2.get_shape()))

    # fc1 layer
    W_fc3 = weight_variable([512, 1000])
    b_fc3 = bias_variable([1000])
    output = tf.matmul(h_fc2, W_fc3) + b_fc3
    #print("output: {}".format(output.get_shape()))
    
    return output
