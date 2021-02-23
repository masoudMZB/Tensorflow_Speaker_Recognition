# to read more about abstract base class read  : https://www.geeksforgeeks.org/abstract-base-class-abc-in-python/

import abc
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, name, **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def call(self, inputs, training=False, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def recognize(self, features, input_lengths, **kwargs):
        pass

    @abc.abstractmethod
    def recognize_beam(self, features, input_lengths, **kwargs):
        pass