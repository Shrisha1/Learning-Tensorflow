# -*- coding: utf-8 -*-
"""
Spyder Editor
This program adds three numbers.First two numbers are 3 and 10. third number is 3.
"""
# importing tensorflow 
import tensorflow as tf 
  
# creating nodes in computation graph 
node1 = tf.constant(3, dtype=tf.int32) 
node2 = tf.constant(10, dtype=tf.int32)
node3 = tf.add(node1, node2)
node4 = tf.constant(1, dtype=tf.int32)
node5 = tf.add(node3,node4) 
  
# create tensorflow session object 
sess = tf.Session() 
  
# evaluating node3 and printing the result 
print("Sum of node1 and node2 is:",sess.run(node5)) 
  
# closing the session 
sess.close() 
