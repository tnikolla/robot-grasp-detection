import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 224,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 12, 
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 12,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 12,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")
def parse_example_proto(examples_serialized):
    feature_map={
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'bboxes': tf.VarLenFeature(dtype=tf.float32)
        }
    features=tf.parse_single_example(examples_serialized, feature_map)
    bboxes = tf.sparse_tensor_to_dense(features['bboxes'])
    r = 8*tf.random_uniform((1,), minval=0, maxval=tf.size(bboxes, out_type=tf.int32)/8, dtype=tf.int32)
    bbox = tf.gather_nd(bboxes, [r,r+1,r+2,r+3,r+4,r+5,r+6,r+7])
    
    return features['image/encoded'], bbox


def eval_image(image, height, width):
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
 
    return image


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
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image, thread_id)
    return distorted_image


def image_preprocessing(image_buffer, train, thread_id=0):
    height = FLAGS.image_size
    width = FLAGS.image_size
    image = tf.image.decode_png(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [height,width])
    if train:
        image = distort_image(image, height, width, thread_id)
    #else:
    #    image = eval_image(image, height, width)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def batch_inputs(data_files, train, num_epochs, batch_size,
                 num_preprocess_threads, num_readers):
    print(train)
    if train:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=True,
                                                        capacity=16)
    else:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=False,
                                                        capacity=1)
    
    examples_per_shard = 1024
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    if train:
        print('pass')
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples+3*batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string])
    else:
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard + 3 * batch_size,
            dtypes=[tf.string])

    if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue,enqueue_ops))
        examples_serialized = examples_queue.dequeue()
    else:
        reader = tf.TFRecordReader()
        _, examples_serialized = reader.read(filename_queue)
    
    images_and_bboxes=[]
    for thread_id in range(num_preprocess_threads):
        image_buffer, bbox = parse_example_proto(examples_serialized)
        image = image_preprocessing(image_buffer, train, thread_id)
        images_and_bboxes.append([image, bbox])
    
    images, bboxes = tf.train.batch_join(
        images_and_bboxes,
        batch_size=batch_size,
        capacity=2*num_preprocess_threads*batch_size)
    
    height = FLAGS.image_size
    width = FLAGS.image_size
    depth = 3
    
    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, height, width, depth])

    return images, bboxes 


def distorted_inputs(data_files, num_epochs, train=True, batch_size=None):
    with tf.device('/cpu:0'):
        print(train)
        images, bboxes = batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=FLAGS.num_readers)
  
    return images, bboxes


def inputs(data_files, num_epochs=1, train=False, batch_size=1):
    with tf.device('/cpu:0'):
        print(train)
        images, bboxes = batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=1)
    
    return images, bboxes
