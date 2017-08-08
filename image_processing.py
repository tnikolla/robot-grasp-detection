import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_integer('batch_size', 32,
#                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 224,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 8,
                            """Number of parallel readers during train.""")

def parse_example_proto(examples_serialized):
    feature_map={
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value='')
        }
    features=tf.parse_single_example(examples_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    return features['image/encoded'], label

def distort_color(image, thread_id):
    color_ordering = thread_id % 2
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def distort_image(image, height, width, thread_id):
    #begin = [height,width]-np.random.randint(10, 2)
    #begin = np.append(begin, 0)
    #size = [height-10,width-10,3]
    #distorted_image = tf.slice(image, begin, size)
    #distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image, thread_id)
    return distorted_image

def image_preprocessing(image_buffer, train, thread_id=0):
    height = FLAGS.image_size
    width = FLAGS.image_size
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [height,width])
    if train:
        image = distort_image(image, height, width, thread_id)
    else:
        image = eval_image(image, height, width)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

#num_readers issue
def batch_inputs(data_files, num_epochs, batch_size=None, train=True, num_preprocess_threads=None,
                 num_readers=1):
    if train:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=True,
                                                        capacity=16)
    else:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        shuffle=False,
                                                        capacity=1)
    
    examples_per_shard = 1024
    #input_queue_memory_factor = 16 works well (inception people)
    min_queue_examples = examples_per_shard * 16
    #num_readers issue
    if train:
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples+3*batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string])
    else:
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard + 3 * batch_size,
            dtypes=[tf.string])
    reader = tf.TFRecordReader()
    _, examples_serialized = reader.read(filename_queue)
    
    #check how this works
    images_and_labels=[]
    for thread_id in range(num_preprocess_threads):
        image_buffer, label_index = parse_example_proto(examples_serialized)
        image = image_preprocessing(image_buffer, train, thread_id)
        images_and_labels.append([image, label_index])
    
    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=batch_size,
        capacity=2*num_preprocess_threads*batch_size)
    
    height = FLAGS.image_size
    width = FLAGS.image_size
    depth = 3
    
    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, height, width, depth])
    
    return images, tf.reshape(label_index_batch, [batch_size])

def distorted_inputs(data_files, num_epochs, batch_size=None, num_preprocess_threads=FLAGS.num_preprocess_threads):
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            data_files, num_epochs, batch_size, train=True,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=FLAGS.num_readers)
    return images, labels
