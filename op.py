import tensorflow as tf
import numpy as np
import os

dx = 1000.0
dy = 1000.0

def copy_2nd_to_1st_left(var):
    shape = var.shape
    op = np.zeros(shape)
    op[1,0] = 1
    op = tf.constant(op)
    return tf.matmul(var,op)

def copy_2nd_to_1st_right(var):
    shape = var.shape
    op = np.zeros(shape)
    op[shape[0]-2,shape[1]-1] = 1
    op = tf.constant(op)
    return tf.matmul(var,op)

def copy_2nd_to_1st_top(var):
    shape = var.shape
    op = np.zeros(shape)
    op[0,1] = 1
    op = tf.constant(op)
    return tf.matmul(op,var)

def copy_2nd_to_1st_bottom(var):
    shape = var.shape
    op = np.zeros(shape)
    op[shape[0]-1,shape[1]-2] = 1
    op = tf.constant(op)
    return tf.matmul(op,var)

def get_circle_inner(var):
    shape = var.shape
    op = np.ones(shape)
    op[:, 0]=0
    op[:, shape[1]-1]=0
    op[0, :]=0
    op[shape[0]-1, :]=0 
    op = tf.constant(op)
    return tf.multiply(var, op)

def get_lr_inner(var):
    shape = var.shape
    op = np.ones(shape)
    for i in range(shape[0]):
        op[i,0] = 0
        op[i,shape[1]-1] = 0
    op = tf.constant(op)
    return tf.multiply(var, op)

def get_tb_inner(var):
    shape = var.shape
    op = np.ones(shape)
    for i in range(shape[0]):
        op[0,i] = 0
        op[shape[0]-1,i] = 0
    op = tf.constant(op)
    return tf.multiply(var, op)

def corner_regulation(var):
    shape = var.shape
    op = np.ones(shape)
    op[0,0] = op[0,shape[1]-1] = op[shape[0]-1,0] = op[shape[0]-1,shape[1]-1] = 0.5
    return tf.multiply(var, op)

def lr_to_zero(var):
    shape = var.shape
    op = np.ones(shape)
    for i in range(shape[0]):
        op[i][0] = op[i][shape[1]-1] = 0
    return tf.multiply(var,op)

def tb_to_zero(var):
    shape = var.shape
    op = np.ones(shape)
    for i in range(shape[1]):
        op[0][i] = op[shape[0]-1][i] = 0
    return tf.multiply(var,op)     


# In[3]:


def expand_left(var):
    shape = var.shape
    var_left = tf.gather(var,0, axis=1) 
    var_left = tf.reshape(var_left, [shape[0],1])
    var_ = tf.concat([var_left,var], axis=1)
    return var_

def expand_right(var):
    shape = var.shape
    var_right = tf.gather(var,shape[1]-1, axis=1) 
    var_right = tf.reshape(var_right, [shape[0],1])
    var_ = tf.concat([var, var_right], axis=1)
    return var_

def expand_top(var):
    shape = var.shape
    var_top = tf.gather(var,0, axis=0) 
    var_top = tf.reshape(var_top, [1,shape[1]])
    var_ = tf.concat([var_top, var], axis=0)
    return var_

def expand_bottom(var):
    shape = var.shape
    var_bottom = tf.gather(var, shape[0]-1, axis=0) 
    var_bottom = tf.reshape(var_bottom, [1,shape[1]])
    var_ = tf.concat([var, var_bottom], axis=0)
    return var_

def expand_surround(var):
    var_ = expand_left(var)
    var_ = expand_right(var_)
    var_ = expand_top(var_)
    var_ = expand_bottom(var_)
    return var_


# In[43]:


def make_kernel(a, b=1):
    """Transform a 2D array into a convolution kernel"""
    c = np.asarray(a)/b
    c = c.reshape(list(c.shape) + [1,1])
    with tf.name_scope('kernel'):
        var = tf.Variable(c, dtype=tf.float64)
    return var

def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='VALID')
    return y[0, :, :, 0]

def AXF(x, expand_type=None):
    kernel = make_kernel([[0.0, 1.0],
                          [0.0, 1.0]],2)
    res = simple_conv(x, kernel)
    if expand_type == 'left_top':
        res = expand_left(res)
        res = expand_top(res)
    elif expand_type == 'right_bottom':
        res = expand_right(res)
        res = expand_bottom(res)
    return res

def DXF(x, expand_type=None):
    kernel = make_kernel([[0.0, -1.0],
                           [0.0, 1.0]],dx)
    res = simple_conv(x, kernel)
    if expand_type == 'left_top':
        res = expand_left(res)
        res = expand_top(res)
    elif expand_type == 'right_bottom':
        res = expand_right(res)
        res = expand_bottom(res)
    return res

def AYF(y, expand_type=None):
    kernel = make_kernel([[0.0, 0.0],
                           [1.0, 1.0]],2)
    res = simple_conv(y, kernel)
    if expand_type == 'left_top':
        res = expand_left(res)
        res = expand_top(res)
    elif expand_type == 'right_bottom':
        res = expand_right(res)
        res = expand_bottom(res)
    return res

def DYF(y, expand_type=None):
    kernel = make_kernel([[0.0, 0.0],
                           [-1.0, 1.0]],dy)
    res = simple_conv(y, kernel)
    if expand_type == 'left_top':
        res = expand_left(res)
        res = expand_top(res)
    elif expand_type == 'right_bottom':
        res = expand_right(res)
        res = expand_bottom(res)
    return res



def AXB(x, expand_type=None):
    kernel = make_kernel([[1.0, 0.0],
                           [1.0, 0.0]],2)
    res = simple_conv(x, kernel)
    if expand_type == 'left_top':
        res = expand_left(res)
        res = expand_top(res)
    elif expand_type == 'right_bottom':
        res = expand_right(res)
        res = expand_bottom(res)
    return res

def xsub_backward(x, expand_type=None):
    kernel = make_kernel([[-1.0, 0.0],
                           [1.0, 0.0]])
    res = simple_conv(x, kernel)
    if expand_type == 'left_top':
        res = expand_left(res)
        res = expand_top(res)
    elif expand_type == 'right_bottom':
        res = expand_right(res)
        res = expand_bottom(res)
    return res

def AYB(y, expand_type=None):
    kernel = make_kernel([[1.0, 1.0],
                           [0.0, 0.0]])
    res = simple_conv(y, kernel)
    if expand_type == 'left_top':
        res = expand_left(res)
        res = expand_top(res)
    elif expand_type == 'right_bottom':
        res = expand_right(res)
        res = expand_bottom(res)
    return res

def ysub_backward(y, expand_type=None):
    kernel = make_kernel([[-1.0, 1.0],
                           [0.0, 0.0]])
    res = simple_conv(y, kernel)
    if expand_type == 'left_top':
        res = expand_left(res)
        res = expand_top(res)
    elif expand_type == 'right_bottom':
        res = expand_right(res)
        res = expand_bottom(res)
    return res
