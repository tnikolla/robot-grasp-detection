import tensorflow as tf
from itertools import islice
import os
import glob
import numpy as np

#dataset = '/root/dataset/cornell_grasping_dataset'
#49
dataset = '/home/tomi/py/dataset'
#32
# If the directory of the current script lies where the out_dir will be
#output_directory = os.path.dirname(os.path.abspath(__file__))
output_filename = 'train-cgd'

class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)
    def decode_png(self, image_data):
        return self._sess.run(self._decode_png,
                              feed_dict={self._decode_png_data: image_data})

def _process_image(filename, coder):
    with open(filename) as f:
        image_data = f.read()
    image = coder.decode_png(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width

def _process_bboxes(name):
    with open(name, 'r') as f:
        bboxes = list(map(
              lambda coordinate: float(coordinate), f.read().strip().split()))
    #print(bboxes)
    return bboxes

def _int64_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def _floats_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

def _bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def _convert_to_example(filename, bboxes, image_buffer, height, width):
    example = tf.train.Example(features=tf.train.Features(feature={
          'image/filename': _bytes_feature(filename),
          'image/encoded': _bytes_feature(image_buffer),
          'image/height': _int64_feature(height),
          'image/width': _int64_feature(width),
          'bboxes': _floats_feature(bboxes)}))
    return example
    
def main():
    
    output_file = os.path.join(dataset, output_filename)
    print(output_file)
    writer = tf.python_io.TFRecordWriter(output_file)
    
    folders = range(1,11)
    folders = ['0'+str(i) if i<10 else '10' for i in folders]
    filenames = []
    
    for i in folders:
        for name in glob.glob(os.path.join(dataset, i, 'pcd'+i+'*r.png')):
            filenames.append(name)
    
    coder = ImageCoder()
    for filename in filenames:
        bbox = filename[:32]+'cpos.txt'
        bboxes = _process_bboxes(bbox)
        image_buffer, height, width = _process_image(filename, coder)
        example = _convert_to_example(filename, bboxes, image_buffer, height, width)
        writer.write(example.SerializeToString())
    
    writer.close()

if __name__ == '__main__':
    main()
