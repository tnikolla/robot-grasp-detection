#!/usr/local/bin/python
''' Training a network on Imagenet.
'''
import os.path
import tensorflow as tf


tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % 'train')
data_files = tf.gfile.Glob(tf_record_pattern)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/root/imagenet-data',
        help='Imagenet dataset.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/root/imagenet-data/',
        help='Directory with training data.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size.'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='/tmp/tf/exp',
        help='Tensorboard logdir.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)