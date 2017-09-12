import tensorflow as tf

filename = '/home/tomi/py/dataset/train-cgd'
#filename = '/home/tomi/grasp-detection/record'

record_iter = tf.python_io.tf_record_iterator(path=filename)
l = []
for string_record in record_iter:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    name = example.features.feature['bboxes'].value.float_list[0]
    print(name)    
    break
    l.append(name)

print(l)