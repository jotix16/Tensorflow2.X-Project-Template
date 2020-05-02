#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sam May  2 15:25:34 2020

@author: mikelzhobro
"""

import sys
sys.dont_write_bytecode = True

import tensorflow as tf

def get_optimizer():
    """
    """
    optimizer = tf.keras.optimizers.SGD()
    return optimizer


        