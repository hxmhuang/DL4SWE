# Solves the fifth test case in Cartesian coordinates on the sphere from:
#     Williamson, et. al. "A Standard Test Set for Numerical Approximations to
#     the Shallow Water Equations in Spherical Geometry",  J. Comput. Phys., 
#     102 , 211-224, 1992.

# For details with regard to RBF-FD implemetation of the above test case, see
# Flyer et al., A guide to RBF-generated finite differences for nonlinear transport: '
# Shallow water simulations on a sphere, J. Comput. Phys. 231 (2012) 4078?095

import tensorflow as tf
import numpy as np
import math as math
 
def test_case_5_cart_rk4_fd(nfile,ep,fdsize,order,dim,gamma,dt,tend,dsply,plt):
    # Initialize the constants for the Williamson test case 5.
    x, y, z, la, th ,r, p_u, p_v, p_w, f, ghm = setupT5(nfile)
    #DPx,DPy,DPz,L = rbfmatrix_fd_hyper([x y z],ep,fdsize,order,dim)
    rbfmatrix_fd_hyper(np.hstack((x,y,z)),ep,fdsize,order,dim)
    #L = gamma * L

    return 0,0

# Calculate the RBF-FD differentiation matrices
def rbfmatrix_fd_hyper(x,ep,fdsize,order,dim):
#    N=len(x[:,1])
#    print("N=\n", N, "fdsize=",fdsize)
#    srange = math.sqrt(6 * fdsize/N)
#    print("srange=\n", srange, "x=",x)
#    rbf =  lambda ep, rd2: np.exp(-ep**2 * rd2)
#    drbf =  lambda ep, rd2: -2 * ex**2 * np.exp( -ep**2 * rd2)

# Requires kd-tree code.
# IN:
# x - nodes in this subroutine, NOT coordinate 'x'
# ep - shape parameter
# fdsize - stencil size (good choices: 31, 50, 74, 101)
# order - L = L^order
# dim - dimension of Laplacian formula
# OUT:
# DPx - sparse differentiation matrix
# DPy - sparse differentiation matrix
# DPy - sparse differentiation matrix
# L - sparse dissipation matrix

    return 0

# Set up for the Williamson test case 5.
def setupT5(nfile):
    x=nfile[:,0]
    y=nfile[:,1]
    z=nfile[:,2]
    nd=x.size
    la, th, r = cart2sph(x, y, z)
    p_u= np.zeros([nd,3], dtype=np.float64)
    p_v= np.zeros([nd,3], dtype=np.float64)
    p_w= np.zeros([nd,3], dtype=np.float64)

    p_u[:,0] = 1-x**2 ;     p_u[:,1] = -x*y;    p_u[:,2] = -x*z
    p_v[:,0] = -x*y ;       p_v[:,1] = 1-y**2;  p_v[:,2] = -y*z
    p_w[:,0] = -x*z ;       p_w[:,1] = -y*z;    p_w[:,2] = 1-z**2
    
    # Coriolis force.
    f = 2*omega*(-x*np.sin(alpha) + z*np.cos(alpha)); 
    
    # Parameters for the mountain:
    pi =  4 * math.atan(1) 
    lam_c = -pi/2;
    thm_c = pi/6;
    mR = pi/9;
    hm0 = 2000;
   
    # Computer the profile of the mountain (multiplied by gravity)
    ghm = np.zeros([nd, 1])
    r2 = (la-lam_c)**2+(th-thm_c)**2
    id = r2 < mR**2
    ghm = g * hm0 * (1 - np.sqrt(r2*id)/mR) * id

    return x, y, z, la, th ,r, p_u, p_v, p_w, f, ghm

def cart2sph(x,y,z):
    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)            # radius
    elev = np.arctan2(z, np.sqrt(xy)) # elevation
    az = np.arctan2(y, x)             # azimuth
    return az, elev, r

   
fd    = tf.constant(31, dtype=tf.int32)     # size of RBF-FD stencil
tend  = tf.constant(1,  dtype=tf.int32)      # ending time, needs to be in days
order = tf.constant(4,  dtype=tf.int32)      # power of Laplacian, L^order
dim   = tf.constant(2 , dtype=tf.int32)     # dimension of stencil, on sphere dim=2

# RBF-FD parameters for N = 3600 node set
ep = tf.constant(2,     dtype=tf.int32)            # controls width of Gaussian RBF
dt = tf.constant(1000,  dtype=tf.int32);         # time step, needs to be in seconds
gamma = tf.constant(-2.97e-16, dtype=tf.float64); #amount of hyperviscosity applied, multiplies Laplacian^order 

alpha = tf.constant(0.0, dtype=tf.float64)         # Angle of rotation measured from the equator.
u0    = tf.constant(20.0, dtype=tf.float64);       # Speed of rotation in meters/second
a     = tf.constant(6.37122e6, dtype=tf.float64);  # Mean radius of the earth (meters).
omega = tf.constant(7.292e-5, dtype=tf.float64);   # Rotation rate of the earth (1/seconds).
g     = tf.constant(9.80616, dtype=tf.float64);    # Gravitational constant (m/s^2).
gh0   = g*5960                               # Initial condition for the geopotential field (m^2/s^2).


# Set to plt=1 if you want to plot results at different time-steps.
dsply = tf.constant(1) 
plt = tf.constant(1)
                  
data=np.loadtxt("md/md001.00004")
fname = tf.constant(data, dtype=tf.float64) 
H,atm = test_case_5_cart_rk4_fd(fname,ep,fd,order,dim,gamma,dt,tend,dsply,plt);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   
    sess.run(gh0)

