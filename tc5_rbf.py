# Author: Xiaomeng Huang
# Date:   Aug. 2, 2018
#
# Solves the fifth test case in Cartesian coordinates on the sphere from:
#     Williamson, et. al. "A Standard Test Set for Numerical Approximations to
#     the Shallow Water Equations in Spherical Geometry",  J. Comput. Phys., 
#     102 , 211-224, 1992.
#
# For details with regard to RBF-FD implemetation of the above test case, see
# Flyer et al., A guide to RBF-generated finite differences for nonlinear transport: '
# Shallow water simulations on a sphere, J. Comput. Phys. 231 (2012) 4078?095

import tensorflow as tf
import numpy as np
import math as math
from scipy.spatial import cKDTree 
from scipy.stats import mode

def setupT5(nfile):
    x =nfile[:,0]
    y =nfile[:,1]
    z =nfile[:,2]
    nd =x.size
  
    np_la, np_th, np_r = cart2sph(x, y, z)
   
    np_p_u= np.zeros([nd,3], dtype=np.float64)
    np_p_v= np.zeros([nd,3], dtype=np.float64)
    np_p_w= np.zeros([nd,3], dtype=np.float64)

    np_p_u[:,0] = 1-x**2 ;     np_p_u[:,1] = -x*y;    np_p_u[:,2] = -x*z
    np_p_v[:,0] = -x*y ;       np_p_v[:,1] = 1-y**2;  np_p_v[:,2] = -y*z
    np_p_w[:,0] = -x*z ;       np_p_w[:,1] = -y*z;    np_p_w[:,2] = 1-z**2

    # Coriolis force.
    np_f = 2*np_omega*(-x*np.sin(np_alpha) + z*np.cos(np_alpha))     

    # Computer the profile of the mountain (multiplied by gravity)
    np_r2 = (np_la-np_lam_c)**2+(np_th-np_thm_c)**2
    np_id = np_r2 < np_mR**2
    np_ghm = np_g * np_hm0 * (1 - np.sqrt(np_r2*np_id)/np_mR) * np_id
    
    return np_la, np_th ,np_r, np_p_u, np_p_v, np_p_w, np_f, np_ghm 

def cart2sph(a,b,c):
    ab = a**2 + b**2
    r = np.sqrt(ab + c**2)            # radius
    elev = np.arctan2(c, np.sqrt(ab)) # elevation
    az = np.arctan2(b, a)             # azimuth
    return az, elev, r

# Calculate the RBF-FD differentiation matrices. Requires kd-tree code.
# IN:
# file - nodes in this subroutine, NOT coordinate 'x'
# ep - shape parameter
# fdsize - stencil size (good choices: 31, 50, 74, 101)
# order - L = L^order
# dim - dimension of Laplacian formula
# OUT:
# DPx - sparse differentiation matrix
# DPy - sparse differentiation matrix
# DPy - sparse differentiation matrix
# L - sparse dissipation matrix
def rbfmatrix_fd_hyper(np_file, np_ps_u, np_ps_v, np_ps_w, np_ep, np_fdsize, np_order, np_dim):
    # get the number of points  
    np_N = len(np_file)
    #np_cords = np_file[:,0:3].reshape([np_N,3])
    np_cords = np_file[:,0:3]
    #Query the kd-tree for nearest neighbors
    np_tree = cKDTree(np_cords)
    np_dist, np_index = np_tree.query(np_cords, k=np_fdsize)
    #print('np_dist=\n', np_dist,'\n------index=\n',np_index)
   
    weightsDx =  np.zeros(np_N * np_fdsize, dtype=np.float64)
    weightsDy =  np.zeros(np_N * np_fdsize, dtype=np.float64)
    weightsDz =  np.zeros(np_N * np_fdsize, dtype=np.float64)
    weightsL  =  np.zeros(np_N * np_fdsize, dtype=np.float64)

    np_imat   =  np.zeros([1,3], dtype=np.float64)
    np_ind_i  =  np.zeros(np_N * np_fdsize, dtype=np.float64)
    np_ind_j  =  np.zeros(np_N * np_fdsize, dtype=np.float64)
   
    np_A      =  np.ones((np_fdsize+1, np_fdsize+1), dtype=np.float64) 
    np_A[np_fdsize,np_fdsize] = 0
    #np_B      =  np.zeros((np_fdsize+1, 1))
    np_B      =  np.zeros(np_fdsize+1, dtype=np.float64)

    for j in range(np_N):
        np_imat = np_index[j,:]
        np_ind_i[j*np_fdsize : (j+1)*np_fdsize] = j
        np_ind_j[j*np_fdsize : (j+1)*np_fdsize] = np_imat 

        
        np_dp = np.dot(np_file[j,0], np_file[np_imat,0]) +  \
                np.dot(np_file[j,1], np_file[np_imat,1]) +  \
                np.dot(np_file[j,2], np_file[np_imat,2])

        x1 =  np_file[np_imat,0].reshape((np_fdsize,1))
        x2 =  np_file[np_imat,1].reshape((np_fdsize,1))
        x3 =  np_file[np_imat,2].reshape((np_fdsize,1))
        np_rd2   = np.maximum(0, 2*(1-np.dot(x1, x1.T) - np.dot(x2, x2.T) - np.dot(x3, x3.T))) 
        np_rd2v  = np_rd2[:,0]
        
        np_A[0:np_fdsize, 0:np_fdsize] = rbf(np_ep, np_rd2)
        np_B[0:np_fdsize] = (np.dot(np_file[j,0], np_dp) - np_file[np_imat,0]) * drbf(np_ep, np_rd2v)
        weights = np.dot(np.linalg.inv(np_A), np_B) 
        weightsDx[j*np_fdsize : (j+1)*np_fdsize] =  weights[0 : np_fdsize]
 
        np_B[0:np_fdsize] = (np.dot(np_file[j,1], np_dp) - np_file[np_imat,1]) * drbf(np_ep, np_rd2v)
        weights = np.dot(np.linalg.inv(np_A), np_B) 
        weightsDy[j*np_fdsize : (j+1)*np_fdsize] =  weights[0 : np_fdsize]
 
        np_B[0:np_fdsize] = (np.dot(np_file[j,2], np_dp) - np_file[np_imat,2]) * drbf(np_ep, np_rd2v)
        weights = np.dot(np.linalg.inv(np_A), np_B) 
        weightsDz[j*np_fdsize : (j+1)*np_fdsize] =  weights[0 : np_fdsize]

        np_B[0:np_fdsize] = np_ep**(2*np_order) * hyper(np_ep**2 * np_rd2v, np_dim, np_order) * np.exp(-np_ep**2 * np_rd2v) 
        weights = np.dot(np.linalg.inv(np_A), np_B) 
        weightsL[j*np_fdsize : (j+1)*np_fdsize] =  weights[0 : np_fdsize]

    np_DPx = np.zeros((np_N, np_N), dtype=np.float64)
    np_DPy = np.zeros((np_N, np_N), dtype=np.float64)
    np_DPz = np.zeros((np_N, np_N), dtype=np.float64)
    np_L   = np.zeros((np_N, np_N), dtype=np.float64)
   
    for i in range(np_N):
        count = 0 
        np_imat = np_index[i,:]
        for j in np_imat:
            np_DPx[i,j] = weightsDx[i*np_fdsize+count] 
            np_DPy[i,j] = weightsDy[i*np_fdsize+count] 
            np_DPz[i,j] = weightsDz[i*np_fdsize+count] 
            np_L[i,j]   = weightsL[i*np_fdsize+count] 
            count    = count + 1
    #print('\nDPx=\n', DPx)
    #print('\nDPy=\n', DPy)
    #print('\nDPz=\n', DPz)
    #print('\nL=\n', L)
    return np_DPx, np_DPy, np_DPz, np_L

def rbf(ep, rd2):
    return np.exp(-ep**2 * rd2)

def drbf(ep, rd2):
    return -2 * ep**2 * rbf(ep, rd2) 

def hyper(ep2r2, d, k):
    n = len(ep2r2)
    P = np.zeros((n,k+1), dtype=np.float64)
    P[:,0] = 1
    P[:,1] = 4*ep2r2 - 2 * d
    for j in range(2,k+1):
        P[:,j] = 4*(ep2r2-2*(j+1)-d/2+4)*P[:,j-1] - 8*(j-1)*(2*(j+1)+d-6)*P[:,j-2] 
    return P[:,k]

# Initial condition for test case 5.  uc contains the velocity in Cartesian 
# coordinates with uc(:,1:3) the x,y,z direction, respectively.  gh contains
# the geopotential height field from the mean geopotential gh0.
def computeInitialCondition(nfile):
    x =nfile[:,0]
    y =nfile[:,1]
    z =nfile[:,2]
    N= len(x)
    
    gh = - (np_a * np_omega * np_u0+ np_u0**2 / 2)*(-x*np.sin(np_alpha) + z*np.cos(np_alpha))**2
    
    uc      = np.zeros((N, 3), dtype=np.float64)
    us      = np.zeros((N, 2), dtype=np.float64)
    c2s_u   = np.zeros((N, 2), dtype=np.float64)
    c2s_v   = np.zeros((N, 2), dtype=np.float64)
    c2s_w   = np.zeros((N, 2), dtype=np.float64)
    
    uc[:,0] = np_u0 * (-y) * np.cos(np_alpha)
    uc[:,1] = np_u0 * (x*np.cos(np_alpha) + z*np.sin(np_alpha))
    uc[:,2] = np_u0 * (-y) * np.sin(np_alpha)

    # vectors for translating the field in Cartesian coordinates to a field
    # in spherical coordinates.
    c2s_u[:,0] = -np.sin(np_la) 
    c2s_u[:,1] = -np.sin(np_th) * np.cos(np_la)
    c2s_v[:,0] =  np.cos(np_la) 
    c2s_v[:,1] = -np.sin(np_th) * np.sin(np_la)
    c2s_w[:,0] = -np.zeros(N, dtype=np.float64) 
    c2s_w[:,1] =  np.cos(np_th)

    us[:,0] = c2s_u[:,0]*uc[:,0] + c2s_v[:,0]*uc[:,1] + c2s_w[:,0]*uc[:,2];
    us[:,1] = c2s_u[:,1]*uc[:,0] + c2s_v[:,1]*uc[:,1] + c2s_w[:,1]*uc[:,2];
        
    return uc, gh, us  
#=============================Start=====================================
graph = tf.Graph()

with graph.as_default():

    with tf.name_scope("Initialization"):
        # size of RBF-FD stencil
        np_fdsize= 5
        # ending time, needs to be in days
        np_tend  = 1
        # power of Laplacian, L^order
        np_order = 4
        # dimension of stencil, on sphere dim=2
        np_dim   = 2
        # controls width of Gaussian RBF
        np_ep    = 2
        # time step, needs to be in seconds
        np_dt    = 900
        #amount of hyperviscosity applied, multiplies Laplacian^order
        np_gamma = -2.97e-16
        # Parameters for the mountain:
        np_pi   = 4 * math.atan(1)
        np_lam_c= -np_pi/2;
        np_thm_c= np_pi/6;
        np_mR   = np_pi/9;
        np_hm0  = 2000;

        # Angle of rotation measured from the equator.
        np_alpha = 0.0
        # Speed of rotation in meters/second
        np_u0    = 20.0
        # Mean radius of the earth (meters).
        np_a     = 6.37122e6
        # Rotation rate of the earth (1/seconds).
        np_omega = 7.292e-5
        # Gravitational constant (m/s^2).
        np_g     = 9.80616
        # Initial condition for the geopotential field (m^2/s^2).
        np_gh0   = np_g*5960                               
 
        # Set to plt=1 if you want to plot results at different time-steps.
        np_dsply = 1 
        np_plt   = 1
        
        AAA= tf.placeholder(tf.float64, shape=[None], name="input_placeholder_a")
        init = tf.global_variables_initializer() 

    with tf.name_scope("SetupConditions"):
        np_file =np.loadtxt("md/md002.00009")
        
        np_la, np_th, np_r, np_p_u, np_p_v, np_p_w, np_f, np_ghm = setupT5(np_file) 
        
        np_DPx, np_DPy, np_DPz, np_L = rbfmatrix_fd_hyper(np_file, np_p_u, np_p_v, np_p_w, \
                                                          np_ep, np_fdsize, np_order, np_dim)
        np_L = np_gamma * np_L

        np_uc, np_gh, np_us = computeInitialCondition(np_file)

        np_H = np.hstack((np_uc, np_gh.reshape((len(np_uc),1))))

        # Compute the projected gradient of the mountain for test case 5
        np_gradghm = np.zeros([len(np_file),3], dtype=np.float64)
        np_gradghm[:,0] = np.dot(np_DPx, np_ghm)/np_a;
        np_gradghm[:,1] = np.dot(np_DPy, np_ghm)/np_a;
        np_gradghm[:,2] = np.dot(np_DPz, np_ghm)/np_a;


    with tf.name_scope("Transformation"):
        # convert the above ndarrays into tensor
        fdsize  = tf.constant(np_fdsize,dtype=tf.int32,   name="fd")    
        tend    = tf.constant(np_tend,  dtype=tf.int32,   name="tend")      
        order   = tf.constant(np_order, dtype=tf.int32,   name="order")      
        dim     = tf.constant(np_dim ,  dtype=tf.int32,   name="dim")     
        ep      = tf.constant(np_ep,    dtype=tf.int32,   name="ep")                    
        dt      = tf.constant(np_dt,    dtype=tf.int32,   name="dt");                  
        gamma   = tf.constant(np_gamma, dtype=tf.float64, name="gamma");         
        pi      = tf.constant(np_pi,    dtype=tf.float64, name="pi")
        lam_c   = tf.constant(np_lam_c, dtype=tf.float64, name="lam_c")
        thm_c   = tf.constant(np_thm_c, dtype=tf.float64, name="thm_c")
        mR      = tf.constant(np_mR,    dtype=tf.float64, name="mR")
        hm0     = tf.constant(np_hm0,   dtype=tf.float64, name="hm0")

        la      = tf.constant(np_la,    dtype=tf.float64, name="la")
        th      = tf.constant(np_th,    dtype=tf.float64, name="th")
        r       = tf.constant(np_r,     dtype=tf.float64, name="r")
        p_u     = tf.constant(np_p_u,   dtype=tf.float64, name="p_u")
        p_v     = tf.constant(np_p_v,   dtype=tf.float64, name="p_v")
        p_w     = tf.constant(np_p_w,   dtype=tf.float64, name="p_w")
        f       = tf.constant(np_f,     dtype=tf.float64, name="f")
        ghm     = tf.constant(np_ghm,   dtype=tf.float64, name="f")
   
        DPx     = tf.constant(np_DPx,   dtype=tf.float64, name="DPx")
        DPy     = tf.constant(np_DPy,   dtype=tf.float64, name="DPy")
        DPz     = tf.constant(np_DPz,   dtype=tf.float64, name="DPz")
        L       = tf.constant(np_L,     dtype=tf.float64, name="L")

        uc      = tf.constant(np_uc,    dtype=tf.float64, name="uc")
        gh      = tf.constant(np_gh,    dtype=tf.float64, name="gh")
        us      = tf.constant(np_us,    dtype=tf.float64, name="us")

        H       = tf.constant(np_H,       dtype=tf.float64, name="H")
        gradghm = tf.constant(np_gradghm, dtype=tf.float64, name="gradghm")

    with tf.name_scope("Running"):


sess = tf.Session(graph=graph)

writer = tf.summary.FileWriter('./graph1', graph)
sess.run(init)

def run_graph(input_tensor):
    feed_dict= {AAA: input_tensor}
    print(sess.run([la,th], feed_dict= feed_dict))

run_graph([])
writer.flush()
writer.close()
sess.close()
#=============================End=======================================
