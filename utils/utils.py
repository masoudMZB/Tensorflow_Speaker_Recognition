import os
import sys
import math
from typing import Union, List
import numpy as np
import tensorflow as tf
from pprint import pprint

# here we will have useful functions : 

'''
tf.where([True, False, False, True], [1,2,3,4], [100,200,300,400])

>>> [1, 200, 300, 4]
'''

def nan_to_zero(input_tensor):
    return tf.where(tf.math.is_nan(input_tensor), tf.zeros_like(input_tensor), input_tensor)


def preprocess_paths(paths: Union[List, str]):
    ''' abspath("../data") give absolute path of where I locate '''
    ''' On Unix and Windows, return the argument with an initial component of ~ or ~user replaced by that user’s home directory. '''
    if isinstance(paths, list):
        return [path if path.startswith('gs://') else os.path.abspath(os.path.expanduser(path)) for path in paths]
    elif isinstance(paths, str):
        return paths if paths.startswith('gs://') else os.path.abspath(os.path.expanduser(paths))
    else:
        return None   

def bytes_to_string(array: np.ndarray, encoding: str = "utf-8"):
    if array is None: return None
    return [transcript.decode(encoding) for transcript in array]

def has_gpu_or_tpu():
    gpus = tf.config.list_logical_devices("GPU")
    tpus = tf.config.list_logical_devices("TPU")
    if len(gpus) == 0 and len(tpus) == 0: return False
    return True

def print_all_properties(obj):
    pprint(vars(obj))

def print_one_line(*args):
    tf.print("\033[K", end="")
    tf.print("\r", *args, sep="", end=" ", output_stream=sys.stdout)