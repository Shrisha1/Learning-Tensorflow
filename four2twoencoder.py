# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:38:31 2019
This program implements FourToEncoder using tensorflow. Here we use three neurons working as OR,NAND, and AND gates
Input to be given through terminal . There re three inputs. Inputs can be binary digits i.e 0 and 1.
@author: shree
"""

import tensorflow as tf
import numpy as np

b=tf.constant([0.5,0.5],shape=[2,1],dtype=tf.float64)


entries = list(map(float, input().split())) 
y3 = np.array(entries).reshape(1, 1)

entries = list(map(float, input().split())) 
y2 = np.array(entries).reshape(1, 1)

entries = list(map(float, input().split())) 
y1 = np.array(entries).reshape(1, 1)

#implement A1, OR y3 and y2

conc=tf.concat([y3,y2],1)
d=tf.matmul(conc,b)+(-0.3)
A1=tf.round(tf.nn.sigmoid(d))


#implement A0,OR y1 and y3

conc=tf.concat([y1,y3],1)
d=tf.matmul(conc,b)+(-0.3)
A0=tf.round(tf.nn.sigmoid(d))

sess=tf.Session()

print("Output of A1 is ",sess.run(A1))
print("Output of A0 is ",sess.run(A0))


sess.close()