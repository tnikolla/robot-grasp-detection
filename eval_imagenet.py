#!/usr/local/bin/python

''' Evaluating ImageNet.
'''

import sys
import argparse
import os.path
import tensorflow as tf
import image_processing
import inference_redmon
import time

def evaluate():
    images, labels = image_processing.inputs(data_files_, batch_size=32)
    
    
