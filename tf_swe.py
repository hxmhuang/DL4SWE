
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
from op import *
from utils import *


# In[2]:


def get_inner_conservation(H_init, U_init, V_init):
    shape = H_init.shape
    inner_op = np.ones(shape)
    g=9.8
    for i in range(shape[0]):
        inner_op[i,0] = 0
        inner_op[i,shape[1]-1] = 0
    for i in range(shape[1]):
        inner_op[0,i] = 0
        inner_op[shape[0]-1,i] = 0
    tot_inner_h = np.sum(H_init*inner_op)
    tot_inner_energy = np.sum((0.5*(U_init**2) + 0.5*(V_init**2) + g*H_init)*inner_op)
    return tot_inner_energy, tot_inner_h


# In[3]:


n = 64
H_init = np.ones([n+2,n+2], dtype=np.float64)
U_init = np.zeros([n+2,n+2], dtype=np.float64)
V_init = np.zeros([n+2,n+2], dtype=np.float64)
dt = 0.5
dx = 1000.0
dy = 1000.0
g = 9.8
tot_step = 4000
for i in range(2,4):
    for j in range(2,4):
        H_init[i,j]  = H_init[i,j] + 0.2

#calculate total  energy 
tot_en, tot_h = get_inner_conservation(H_init, U_init, V_init)
print(tot_en)

# In[45]:


H  =tf.placeholder(tf.float64, shape=(n+2,n+2))
U  =tf.placeholder(tf.float64, shape=(n+2,n+2))
V  =tf.placeholder(tf.float64, shape=(n+2,n+2))


# In[46]:


H_left = copy_2nd_to_1st_left(H)
H_right = copy_2nd_to_1st_right(H)
H_top = copy_2nd_to_1st_top(H)
H_bottom = copy_2nd_to_1st_bottom(H)
H_inner = get_circle_inner(H)

H_ = corner_regulation(H_left + H_right + H_top + H_bottom + H_inner)

U_top = -copy_2nd_to_1st_top(U)
U_bottom= -copy_2nd_to_1st_bottom(U)
U_inner = get_tb_inner(U)
U_ = U_top + U_bottom + U_inner
U_ = lr_to_zero(U_)

V_left = -copy_2nd_to_1st_left(V)
V_right = -copy_2nd_to_1st_right(V)
V_inner = get_lr_inner(V)
V_ = V_left + V_right + V_inner
V_ = tb_to_zero(V_)


Hx = xplus_forward(H_)/2 - dt/(2*dx)*xsub_forward(U_) 
Hy = yplus_forward(H_)/2 - dt/(2*dy)*ysub_forward(V_)

Ux = xplus_forward(U_)/2 - dt/(2*dx)* xsub_forward(tf.divide(tf.square(U_), H_) + g/2*tf.square(H_)) 
Uy = yplus_forward(U_)/2 - dt/(2*dy)* ysub_forward(tf.divide(tf.multiply(U_,V_), H_))  

Vx = xplus_forward(V_)/2 - dt/(2*dx)* xsub_forward(tf.divide(tf.multiply(U_,V_),H_ ))
Vy = yplus_forward(V_)/2 - dt/(2*dy)* ysub_forward(tf.divide(tf.square(V_), H_ ) + g/2*tf.square(H_)) 

dH = (dt/dx)* xsub_backward(Ux) + (dt/dy)* ysub_backward(Vy)
dU = (dt/dx)* xsub_backward(tf.divide(tf.square(Ux),Hx) + g/2*tf.square(Hx)) +        (dt/dy)* ysub_backward(tf.divide(tf.multiply(Vy,Uy), Hy))
dV = (dt/dx)* xsub_backward(tf.divide(tf.multiply(Ux,Vx), Hx)) +        (dt/dy)* ysub_backward(tf.divide(tf.square(Vy),Hy) + g/2*tf.square(Hy))

H_next = H_ - expand_surround(dH)
U_next = U_ - expand_surround(dU)
V_next = V_ - expand_surround(dV)

iter_op = [H_next, U_next, V_next]
en_loss = tf.square(tf.reduce_sum(get_circle_inner(0.5*tf.square(U_next) + 0.5*tf.square(V_next) + g*H_next)) - tot_en)
h_loss = tf.square(tf.reduce_sum(get_circle_inner(H_next)) - tot_h)
loss = en_loss + h_loss
                    
# In[47]:

#the optimize item is energy conservation
optimizer = tf.train.AdamOptimizer(3e-4, beta1=0.5, beta2=0.9).minimize(loss)


# In[6]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result_H = []
    result_U = []
    result_V = []
    feed_dict_init = {H:H_init, U:U_init, V:V_init}
    _, H_, U_, V_ = sess.run([optimizer,H_next, U_next, V_next], feed_dict = feed_dict_init)
    #loss_, H_, U_, V_ = sess.run([loss,H_next, U_next, V_next], feed_dict = feed_dict_init)
    feed_dict_iter = {H:H_, U:U_, V:V_}
    
    for i in range(tot_step):
        if i<4000:
            _, loss_, H_, U_, V_ = sess.run([optimizer, loss, H_next, U_next, V_next], feed_dict=feed_dict_iter)
        else:
            loss_, H_, U_, V_ = sess.run([loss, H_next, U_next, V_next], feed_dict=feed_dict_iter)
        feed_dict_iter = {H:H_, U:U_, V:V_}
        result_H.append(H_)
        result_U.append(U_)
        result_V.append(V_)
        if i%200 ==0:
            print('loss of energy: {}'.format(loss_))
        """
        if i == tot_step -1:
            for var in tf.trainable_variables(): 
                if 'kernel' in var.name:
                    print("*"*100)
                    print(var.eval())
        """
        


# In[9]:


draw_3D(result_H[1990], 64)

