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
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def setupT5(nfile):
    np_x = nfile[:,0]
    np_y = nfile[:,1]
    np_z = nfile[:,2]
    nd   = np_x.size
  
    np_la, np_th, np_r = cart2sph(np_x, np_y, np_z)
   
    np_p_u= np.zeros([nd,3], dtype=np.float64)
    np_p_v= np.zeros([nd,3], dtype=np.float64)
    np_p_w= np.zeros([nd,3], dtype=np.float64)

    np_p_u[:,0] = 1-np_x**2 ;  np_p_u[:,1] = -np_x*np_y; np_p_u[:,2] = -np_x*np_z
    np_p_v[:,0] = -np_x*np_y ; np_p_v[:,1] = 1-np_y**2;  np_p_v[:,2] = -np_y*np_z
    np_p_w[:,0] = -np_x*np_z ; np_p_w[:,1] = -np_y*np_z; np_p_w[:,2] = 1-np_z**2

    # Coriolis force.
    np_f = 2*np_omega*(-np_x*np.sin(np_alpha) + np_z*np.cos(np_alpha))     

    # Computer the profile of the mountain (multiplied by gravity)
    np_r2 = (np_la-np_lam_c)**2+(np_th-np_thm_c)**2
    np_id = np_r2 < np_mR**2
    np_ghm = np_g * np_hm0 * (1 - np.sqrt(np_r2*np_id)/np_mR) * np_id
    
    return np_x, np_y, np_z, np_la, np_th ,np_r, np_p_u, np_p_v, np_p_w, np_f, np_ghm 

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


# Evaluates the RHS (spatial derivatives) for the Cartesian RBF 
# formulation of the shallow water equations with projected gradients.
# This function applies to Test Case 5, which contains the forcing of
# the mountain
def evalCartRhs_fd(x,y,z,f,g,a,gh0,ghm,gradghm,H,DPx,DPy,DPz,L,p_u,p_v,p_w):
    # Compute the (projected) Cartesian derivatives applied to the velocity
    # and geopotential.
    Tx = tf.matmul(DPx,H)/a 
    Ty = tf.matmul(DPy,H)/a 
    Tz = tf.matmul(DPz,H)/a 

    # This is the computation for the right hand side of the (Cartesian) 
    # momentum equations.
    p = -(H[:,0]*Tx[:,0] + H[:,1]*Ty[:,0] + H[:,2]*Tz[:,0] + f*(y*H[:,2] - z*H[:,1]) + Tx[:,3])
    q = -(H[:,0]*Tx[:,1] + H[:,1]*Ty[:,1] + H[:,2]*Tz[:,1] + f*(z*H[:,0] - x*H[:,2]) + Ty[:,3])
    s = -(H[:,0]*Tx[:,2] + H[:,1]*Ty[:,2] + H[:,2]*Tz[:,2] + f*(x*H[:,1] - y*H[:,0]) + Tz[:,3])

    # Project the momentum equations onto the surface of the sphere.
    F1 = p_u[:,0]*p + p_u[:,1]*q + p_u[:,2]*s;
    F2 = p_v[:,0]*p + p_v[:,1]*q + p_v[:,2]*s;
    F3 = p_w[:,0]*p + p_w[:,1]*q + p_w[:,2]*s;

    # Right-hand side for the geopotential (Does not need to be projected, this
    # has already been accounted for in the DPx, DPy, and DPz operators for
    # this equation).
    F4 = -( H[:,0]*(Tx[:,3] - gradghm[:,0]) + \
            H[:,1]*(Ty[:,3] - gradghm[:,1]) + \
            H[:,2]*(Tz[:,3] - gradghm[:,2]) + \
           (H[:,3]+gh0-ghm)*(Tx[:,0] + Ty[:,1] + Tz[:,2]))
    
    F = tf.concat([tf.reshape(F1,[-1,1]), \
                  tf.reshape(F2,[-1,1]), \
                  tf.reshape(F3,[-1,1]), \
                  tf.reshape(F4,[-1,1])], 1)
    
    # Apply the hyper-viscosity, either once or twice.
    G = F + tf.matmul(L,H);
    return G

#=============================Define Graph==============================
graph = tf.Graph() 
with graph.as_default():
    with tf.name_scope("Input"):
        x       = tf.placeholder(dtype=tf.float64, name="x")
        y       = tf.placeholder(dtype=tf.float64, name="y")
        z       = tf.placeholder(dtype=tf.float64, name="z")
        f       = tf.placeholder(dtype=tf.float64, name="f")
        g       = tf.placeholder(dtype=tf.float64, name="g")
        a       = tf.placeholder(dtype=tf.float64, name="a")
        gh0     = tf.placeholder(dtype=tf.float64, name="gh0")
        ghm     = tf.placeholder(dtype=tf.float64, name="ghm")
        gradghm = tf.placeholder(dtype=tf.float64, name="gradghm")
        H       = tf.placeholder(dtype=tf.float64, name="H")
        DPx     = tf.placeholder(dtype=tf.float64, name="DPx")
        DPy     = tf.placeholder(dtype=tf.float64, name="DPy")
        DPz     = tf.placeholder(dtype=tf.float64, name="DPz")
        L       = tf.placeholder(dtype=tf.float64, name="L")
        p_u     = tf.placeholder(dtype=tf.float64, name="p_u")
        p_v     = tf.placeholder(dtype=tf.float64, name="p_v")
        p_w     = tf.placeholder(dtype=tf.float64, name="p_w")
        dt      = tf.placeholder(dtype=tf.float64, name="dt")    
        
        #gamma   = tf.placeholder(dtype=tf.float64,   name="gamma")    

        #K       = tf.get_variable("K", shape=[9,4], initializer=H)
        K1      = H
        
        d1      = dt*evalCartRhs_fd(x,y,z,f,g,a,gh0,ghm,gradghm,K1,DPx,DPy,DPz,L,p_u,p_v,p_w)
        K2      = H + 0.5*d1;
        d2      = dt*evalCartRhs_fd(x,y,z,f,g,a,gh0,ghm,gradghm,K2,DPx,DPy,DPz,L,p_u,p_v,p_w)
        K3      = H + 0.5*d2;
        d3      = dt*evalCartRhs_fd(x,y,z,f,g,a,gh0,ghm,gradghm,K3,DPx,DPy,DPz,L,p_u,p_v,p_w)
        K4      = H + d3;
        d4      = dt*evalCartRhs_fd(x,y,z,f,g,a,gh0,ghm,gradghm,K4,DPx,DPy,DPz,L,p_u,p_v,p_w)
        Res     = H + 1/6*(d1 + 2*d2 + 2*d3 + d4)

        init = tf.global_variables_initializer() 

#=============================Start Session=============================
with tf.Session(graph=graph) as sess:
    # size of RBF-FD stencil
    #np_fdsize= 5
    np_fdsize= 31 
    # ending time, needs to be in days
    np_tend  = 1
    # power of Laplacian, L^order
    np_order = 4
    # dimension of stencil, on sphere dim=2
    np_dim   = 2
    # controls width of Gaussian RBF
    np_ep    = 2.0
    # time step, needs to be in seconds
    np_dt    = 900.0
    #amount of hyperviscosity applied, multiplies Laplacian^order
    np_gamma = -2.97e-16
    # Parameters for the mountain:
    np_pi   = 4.0 * math.atan(1)
    np_lam_c= -np_pi/2;
    np_thm_c= np_pi/6;
    np_mR   = np_pi/9;
    np_hm0  = 2000.0;
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
    np_dsply = 50 
    np_plt   = 1

    #np_file =np.loadtxt("md/md002.00009")
    np_file =np.loadtxt("md/md059.03600")
    # Setup tc5 case
    np_x, np_y, np_z, np_la, np_th, np_r, np_p_u, \
        np_p_v, np_p_w, np_f, np_ghm = setupT5(np_file) 
    # Compute the rbf difference matrix 
    np_DPx, np_DPy, np_DPz, np_L = rbfmatrix_fd_hyper(np_file, np_p_u, np_p_v, np_p_w, \
                                                  np_ep, np_fdsize, np_order, np_dim)
    np_L = np_gamma * np_L
    # compute the initial condition
    np_uc, np_gh, np_us = computeInitialCondition(np_file)
    np_H = np.hstack((np_uc, np_gh.reshape((len(np_uc),1))))

    # Compute the projected gradient of the mountain for test case 5
    np_gradghm = np.zeros([len(np_file),3], dtype=np.float64)
    np_gradghm[:,0] = np.dot(np_DPx, np_ghm)/np_a;
    np_gradghm[:,1] = np.dot(np_DPy, np_ghm)/np_a;
    np_gradghm[:,2] = np.dot(np_DPz, np_ghm)/np_a;

    sess.run(init)

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.scatter(np_la, np_th)
    plt.ion() 
    plt.show()


    feed_dict={ x       : np_x       ,\
                y       : np_y       ,\
                z       : np_z       ,\
                f       : np_f       ,\
                g       : np_g       ,\
                a       : np_a       ,\
                gh0     : np_gh0     ,\
                ghm     : np_ghm     ,\
                gradghm : np_gradghm ,\
                H       : np_H	  ,\
                DPx     : np_DPx     ,\
                DPy     : np_DPy     ,\
                DPz     : np_DPz     ,\
                L       : np_L       ,\
                p_u     : np_p_u     ,\
                p_v     : np_p_v     ,\
                p_w     : np_p_w     ,\
                dt      : np_dt      } 

    for nt in range(np_tend*24*3600):
    #for nt in range(900):
    #for nt in range(900):
        #print("\n----------------------step:",nt+1,"-----------------------------\n")
        [H_next] = sess.run([Res], feed_dict= feed_dict)
        feed_dict={ x       : np_x       ,\
                    y       : np_y       ,\
                    z       : np_z       ,\
               	    f       : np_f       ,\
               	    g       : np_g       ,\
               	    a       : np_a       ,\
               	    gh0     : np_gh0     ,\
               	    ghm     : np_ghm     ,\
               	    gradghm : np_gradghm ,\
               	    H       : H_next ,\
                    DPx     : np_DPx     ,\
               	    DPy     : np_DPy     ,\
               	    DPz     : np_DPz     ,\
               	    L       : np_L       ,\
               	    p_u     : np_p_u     ,\
               	    p_v     : np_p_v     ,\
               	    p_w     : np_p_w     ,\
               	    dt      : np_dt      } 

        #print(H_next)
        if np_plt == 1:
            #if nt % np_dsply ==0:
            if nt % np_dsply  ==0:
                value = (H_next[:,3]+np_gh0)/np_g
                #print(z)
                ax.cla()
                ax.tricontour(np_la, np_th, value, 14, linewidths=0.5, colors='k')
                cntr  = ax.tricontourf(np_la, np_th, value, 14, cmap="RdBu_r")
                #fig.colorbar(cntr, ax=ax)
                #ax.plot(np_la, np_th, 'ko', ms=3)
                ax.axis((-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
                ax.set_title('tricontour (%d points)' % nt)
                plt.pause(0.01)

sess.close()
#=============================End=======================================
