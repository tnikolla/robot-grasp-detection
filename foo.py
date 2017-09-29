import tensorflow as tf

filename = '/home/tomi/py/dataset/train-cgd'
#filename = '/home/tomi/grasp-detection/record'
#filename = '/root/dataset/cornell_grasping_dataset/train-cgd'
record_iter = tf.python_io.tf_record_iterator(path=filename)
example = tf.train.Example()
l = []
for record in record_iter:
    example.ParseFromString(record)
    bboxes = example.features.feature['bboxes'].float_list.value[:]
    height = example.features.feature['image/height'].int64_list.value[:]
    width = example.features.feature['image/width'].int64_list.value[:]
    l.append([bboxes,(height,width)])

print(l)