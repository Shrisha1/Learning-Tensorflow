# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:32:33 2019
This program implement XOR gate. using NAND, OR and AND gate.
@author: shree
"""

import tensorflow as tf

a=tf.constant([0,0,0,1,1,0,1,1],shape=[4,2],dtype=tf.float32)
b=tf.constant([0.5,0.5],shape=[2,1],dtype=tf.float32)

bvalue=tf.constant([-0.8],dtype=tf.float32)

d=tf.matmul(a,b)+bvalue
j=tf.round(tf.nn.sigmoid(d))
h=tf.cast(tf.logical_not(tf.cast(tf.round(tf.nn.sigmoid(d)),tf.bool)),tf.float32)

d=tf.matmul(a,b)+(-0.3)
o=tf.round(tf.nn.sigmoid(d))

conc=tf.concat([h,o],1)

d=tf.matmul(conc,b)+(-0.8)
f=tf.round(tf.nn.sigmoid(d))

sess=tf.Session()

print("output of matrix multiplication with bias with AND is",sess.run(j))
print("output of matrix multiplication with bias with NAND is",sess.run(h))
print("output of matrix multiplication with bias with OR is",sess.run(o))
print("output of matrix multiplication with bias with concatinated value is",sess.run(conc))
print("output of matrix multiplication with bias with XOR is",sess.run(f))
sess.close()