# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:18:52 2019

@author: shrisha
"""

import tensorflow as tf
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('flower1.jpg')
image=tf.image.rgb_to_grayscale(image)

sess = tf.Session()

rg=sess.run(image)
print(rg.shape)
rg=np.reshape(rg,[5528,3685])
plt.imshow(rg)
print(rg.shape)
sess.close()