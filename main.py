# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:27:14 2017
Rebuild: sta arb, the pair trading, and cointergration

@author: talen
"""
import os,sys,time,math
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
starting_time = time.time()
sys.path.append(os.getcwd())

ori_data = pd.read_excel('老板华帝苏泊尔最高价2012July-2017July.xlsx')
stock1, stock2 = ori_data.columns[0], ori_data.columns[1]
'''
data1 = np.array(ori_data.ix[2:,stock1])
data2 = np.array(ori_data.ix[2:,stock2])
data11 = np.c_[np.ones((len(data1),1)), data1]
theta_best = np.linalg.inv(data11.T.dot(data11)).dot(data11.T).dot(data2)
'''
data1111 = np.array(ori_data.ix[2:,stock1])
data111 = np.array([math.log(10,data1111[i]) for i in range(len(data1111))])
data11 = np.c_[np.ones((len(data111),1)), data111]
data1 = tf.constant(data11, dtype = tf.float32, name='data1')
data22 = np.array([math.log(10,ori_data.ix[2+i,stock2]) for i in range(len(data1111))])
data2 = tf.constant(data22.reshape(-1,1),dtype = tf.float32,name='data2')
data1t = tf.transpose(data1)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(data1t,data1)),data1t),data2)

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)

plt.plot(data111,data22,'b.')
xplot = [0.5,2]
yplot = [0.5,2.5]
plt.axis([xplot[0],xplot[1],yplot[0],yplot[1]])
yline = [theta_value[0][0]+theta_value[1][0]*data111[i] for i in range(len(data111))]
plt.plot(data111,yline,'r')
plt.xlabel(stock1)
plt.ylabel(stock2)

plt.grid(b=None)
plt.show()
gap_seq = data22 - yline
plt.figure(2)
plt.plot(gap_seq,'g')
plt.grid(b=None)

gap_seq_diff = [gap_seq[i+1] - gap_seq[i] for i in range(len(gap_seq)-1)]
plt.figure(3)
plt.plot(gap_seq_diff,'r')
plt.grid(b=None)

