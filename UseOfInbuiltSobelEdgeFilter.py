# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:23:09 2019

@author: shree
"""

import tensorflow as tf
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()
#Read Image
image = skimage.io.imread('vaderHD.jpg')
#Convert image to np array , cast it to float32 and expand dimension batchsize to 1 for 
#single image so as to become ompatible input for tf.image.sobel_edges  
image=np.array(image)
image = tf.cast(image, tf.float32)
image=tf.compat.v1.expand_dims(image, 0)

# call tf.image.sobel_edges and run the tensor
sobel= tf.image.sobel_edges(image)
SobelImage=sess.run(sobel)
# Convert the last  dimension to individual x and y coordinates 
sobel_x = np.asarray(SobelImage[0, :, :, :, 0])
sobel_y = np.asarray(SobelImage[0, :, :, :, 1])

# plot images

plt.imshow(sobel_x)
plt.imshow(sobel_y)
sess.close()