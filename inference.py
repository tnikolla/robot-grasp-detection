'''
Inference model for grasping
'''

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def inference(images):
    # conv1 pool1 layer
    W_conv1 = weight_variable([5,5,3,64])
    b_conv1 = bias_variable([64])
    #x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # conv2 pool layer
    W_conv2 = weight_variable([3,3,64,128])
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # conv3 pool layer
    W_conv3 = weight_variable([3,3,128,256])
    b_conv3 = bias_variable([256])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    
    # fc1 layer
    W_fc1 = weight_variable([7*7*256, 512])
    b_fc1 = bias_variable([512])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 7*7*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)