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
import matplotlib.pyplot as plt 
import matplotlib.tri as tri 
from scipy.spatial import cKDTree 
import time

def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def setupT5(nfile):
    N    = nfile.shape[0]   
    
    x = nfile[:,0].reshape((N,1))
    y = nfile[:,1].reshape((N,1))
    z = nfile[:,2].reshape((N,1))   
    
    la, th, r = cart2sph(x, y, z)
    
    p_u = np.hstack((1-x**2, -x*y  , -x*z  ))
    p_v = np.hstack((-x*y  , 1-y**2, -y*z  )) 
    p_w = np.hstack((-x*z  , -y*z,   1-z**2)) 
   
    X = nfile[:,0:3]
   # P = np.eye(
    # Coriolis force.
    f = 2*omega*(-x*np.sin(alpha) + z*np.cos(alpha))
    
    # Computer the profile of the mountain (multiplied by gravity)
    r2 = (la-lam_c)**2+(th-thm_c)**2
    id = r2 < mR**2
    ghm = g * hm0 * (1 - np.sqrt(r2*id)/mR) * id

    return x, y, z, la, th ,r, p_u, p_v, p_w, f, ghm    


def cart2sph(a,b,c):
    ab = a**2 + b**2
    r = np.sqrt(ab + c**2)            # radius
    elev = np.arctan2(c, np.sqrt(ab)) # elevation
    az = np.arctan2(b, a)             # azimuth
    return az, elev, r

# Calculate the RBF-FD differentiation matrices. Requires kd-tree code.
def rbfmatrix_fd_hyper(nfile, ps_u, ps_v, ps_w, ep, fdsize, order, dim):
    # get the number of points  
    N = nfile.shape[0]
    X = nfile[:,0:3]
    #Query the kd-tree for nearest neighbors
    tree = cKDTree(X)
    dist, index = tree.query(X, k=fdsize)
    #print('dist=\n', dist,'\n------index=\n',index)

    weightsDx =  np.zeros(N * fdsize, dtype=np.float64)
    weightsDy =  np.zeros(N * fdsize, dtype=np.float64)
    weightsDz =  np.zeros(N * fdsize, dtype=np.float64)
    weightsL  =  np.zeros(N * fdsize, dtype=np.float64)

    imat   =  np.zeros([1,3], dtype=np.float64)
    ind_i  =  np.zeros(N * fdsize, dtype=np.int64)
    ind_j  =  np.zeros(N * fdsize, dtype=np.int64)

    A      =  np.ones((fdsize+1, fdsize+1), dtype=np.float64)
    A[fdsize,fdsize] = 0
    B      =  np.zeros(fdsize+1, dtype=np.float64)
    for j in range(N):
        imat = index[j,:]
        ind_i[j*fdsize : (j+1)*fdsize] = j
        ind_j[j*fdsize : (j+1)*fdsize] = imat

        dp = np.dot(nfile[j,0], nfile[imat,0]) +  \
             np.dot(nfile[j,1], nfile[imat,1]) +  \
             np.dot(nfile[j,2], nfile[imat,2])
        
        x1 =  nfile[imat,0].reshape((fdsize,1))
        x2 =  nfile[imat,1].reshape((fdsize,1))
        x3 =  nfile[imat,2].reshape((fdsize,1))
        rd2   = np.maximum(0, 2*(1-np.dot(x1, x1.T) - np.dot(x2, x2.T) - np.dot(x3, x3.T)))
        rd2v  = rd2[:,0]

        A[0:fdsize, 0:fdsize] = rbf(ep, rd2)
        B[0:fdsize] = (np.dot(nfile[j,0], dp) - nfile[imat,0]) * drbf(ep, rd2v)
        weights = np.dot(np.linalg.inv(A), B)
        weightsDx[j*fdsize : (j+1)*fdsize] =  weights[0 : fdsize]

        B[0:fdsize] = (np.dot(nfile[j,1], dp) - nfile[imat,1]) * drbf(ep, rd2v)
        weights = np.dot(np.linalg.inv(A), B)
        weightsDy[j*fdsize : (j+1)*fdsize] =  weights[0 : fdsize]

        B[0:fdsize] = (np.dot(nfile[j,2], dp) - nfile[imat,2]) * drbf(ep, rd2v)
        weights = np.dot(np.linalg.inv(A), B)
        weightsDz[j*fdsize : (j+1)*fdsize] =  weights[0 : fdsize]

        B[0:fdsize] = ep**(2*order) * hyper(ep**2 * rd2v, dim, order) * np.exp(-ep**2 * rd2v)

        weights = np.dot(np.linalg.inv(A), B)
        weightsL[j*fdsize : (j+1)*fdsize] =  weights[0 : fdsize]

    indices=np.concatenate((np.expand_dims(ind_i, 1), np.expand_dims(ind_j, 1)), 1)
    #print('indices=\n', indices)

    return indices, weightsDx, weightsDy, weightsDz, weightsL

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
    N = nfile.shape[0]
    x = nfile[:,0].reshape((N,1))
    y = nfile[:,1].reshape((N,1))
    z = nfile[:,2].reshape((N,1))


    gh = - (a * omega * u0+ u0**2.0 / 2.0)*(-x*np.sin(alpha) + z*np.cos(alpha))**2.0

    uc= u0 * np.hstack((-y * np.cos(alpha), \
                        x*np.cos(alpha) + z*np.sin(alpha),\
                        -y*np.sin(alpha)))   
    # vectors for translating the field in Cartesian coordinates to a field
    # in spherical coordinates.
    c2s_u = np.hstack((-np.sin(la), -np.sin(th) * np.cos(la)))
    c2s_v = np.hstack(( np.cos(la), -np.sin(th) * np.sin(la)))
    c2s_w = np.hstack((-np.zeros((N,1)), np.cos(th)             ))
    
    us      = np.zeros((N, 2), dtype=np.float64)     
    us[:,0] = c2s_u[:,0]*uc[:,0] + c2s_v[:,0]*uc[:,1] + c2s_w[:,0]*uc[:,2];
    us[:,1] = c2s_u[:,1]*uc[:,0] + c2s_v[:,1]*uc[:,1] + c2s_w[:,1]*uc[:,2];
    
    return uc, gh, us

# Evaluates the RHS (spatial derivatives) for the Cartesian RBF 
# formulation of the shallow water equations with projected gradients.
# This function applies to Test Case 5, which contains the forcing of
# the mountain
def evalCartRhs_fd(U,H,DPx,DPy,DPz,L,X,f,g,a,gh0,p_u,p_v,p_w,gradghm,dt):
    Ux = tf.sparse_tensor_dense_matmul(DPx,U)/a
    Uy = tf.sparse_tensor_dense_matmul(DPy,U)/a
    Uz = tf.sparse_tensor_dense_matmul(DPz,U)/a
    Hx = tf.sparse_tensor_dense_matmul(DPx,H)/a
    Hy = tf.sparse_tensor_dense_matmul(DPy,H)/a
    Hz = tf.sparse_tensor_dense_matmul(DPz,H)/a    
    
    U_ = tf.stack((Ux,Uy,Uz),axis=0)
    H_ = tf.concat((Hx,Hy,Hz),axis=1) 
    RHS= -(tf.einsum('ik,kij->ij', U, U_) + f*tf.cross(X,U) + H_)

    # Project the momentum equations onto the surface of the sphere.
    #p=RHS[:,0]; q=RHS[:,1] ; s=RHS[:,2]
    #F1 = p_u[:,0]*p + p_u[:,1]*q + p_u[:,2]*s;
    #F2 = p_v[:,0]*p + p_v[:,1]*q + p_v[:,2]*s;
    #F3 = p_w[:,0]*p + p_w[:,1]*q + p_w[:,2]*s;
    #comments :
    #  let P=[p_u p_v p_w] RHS=[p q s]
    #for j in range(3)
    #    for i in range(9)
    #        for k in range(3)
                #F[i, j] = sum_k P[i, j, k] * RHS[i, k]        
    P  = tf.stack((p_u,p_v,p_w),axis=2)   
    F  = tf.einsum('ijk,ik->ij', P, RHS)

    # Right-hand side for the geopotential (Does not need to be projected, this
    # has already been accounted for in the DPx, DPy, and DPz operators for
    # this equation).
    #G  = -( U_[:,0]*(H_ - gradghm[:,0]) + \
    #        U_[:,1]*(H_ - gradghm[:,1]) + \
    #        U_[:,2]*(H_ - gradghm[:,2]) + \
    #       (H+gh0-ghm)*(U_[:,0] + U_[:,1] + U_[:,2]))
    G1  = tf.reshape(tf.einsum('ij,ij->i', U, H_ - gradghm), shape=(-1,1))
    G2  = (H + gh0 - ghm) * tf.reshape((Ux[:,0] + Uy[:,1] + Uz[:,2]),shape=(-1,1))
    G = -(G1+G2)
    
    # Still have some precision problem
    F_adjust = F + tf.sparse_tensor_dense_matmul(L, U)
    G_adjust = G + tf.sparse_tensor_dense_matmul(L, H)
    
    #return F, tf.sparse_tensor_dense_matmul(L, U)
    return F_adjust, G_adjust

def computeMetric(Vel, gH):
    global th
    #energy = np.sum(0.5*(Vel**2) - gH)
    energy = np.sum(-gH*(Vel**2 - gH)/2)
    mass   = np.sum(-gH)
    #mass   = np.sum(-gH*np.cos(th))
    #energy = np.sum((Vel**2 + gH**2)*np.cos(th))
    return energy, mass

def computeMetricTensor(Vel, gH):
    global th
    #energy = tf.reduce_sum(0.5*tf.square(Vel) - gH)
    energy = tf.reduce_sum(-gh*(Vel**2 - gH)/2)
    mass   = tf.reduce_sum(-gH)
    #mass   = tf.reduce_sum(-gH*np.cos(th))
    #energy = tf.reduce_sum((Vel**2 + gH**2)*np.cos(th))
    return energy, mass

#=============================Define Parameters==============================      
# size of RBF-FD stencil
fdsize= 0 
#nfile =np.loadtxt("md/md002.00009")
#nfile =np.loadtxt("md/md019.00400")
nfile  =np.loadtxt("md/md059.03600")
#nfile =np.loadtxt("md/md079.06400")
#nfile =np.loadtxt("md/md164.27225")
N = nfile.shape[0]
# time step, needs to be in seconds
#dt    = 900.0
dt    = 0.0
#amount of hyperviscosity applied, multiplies Laplacian^order
#gamma = -2.97e-16
gamma = 0

learning_rate = 0.0 
beta1 = 0.0
weight = 0.0
if   N == 9:
    fdsize = 5 ; gamma = -2.97e-15 ; dt = 2000.0 ; learning_rate = 2e-6 ;  beta1 = 1e-1 ; weight = 1e3
elif N == 400:
    fdsize = 31; gamma = -2.97e-15 ; dt = 2000.0 ; learning_rate = 1e-7 ;  beta1 = 1e-1 ; weight = 1e3
elif N == 3600:
    fdsize = 31; gamma = -2.97e-16 ; dt = 2000.0 ; learning_rate = 1e-7 ;  beta1 = 1e-1 ; weight = 1e3
elif N == 4900:
    fdsize = 31; gamma = -2.97e-15 ; dt = 1000.0 ; learning_rate = 1e-7 ;  beta1 = 1e-1 ; weight = 1e3
elif N == 6400:
    fdsize = 31; gamma = -2.97e-16 ; dt = 1000.0 ; learning_rate = 1e-7 ;  beta1 = 1e-1 ; weight = 1e3

# ending time, needs to be in days
tend  = 1
# power of Laplacian, L^order
order = 4
# dimension of stencil, on sphere dim=2
dim   = 2
# controls width of Gaussian RBF
ep    = 2.0

#gamma =0 
# Parameters for the mountain:
lam_c= -np.pi/2;
thm_c= np.pi/6;
mR   = np.pi/9;
hm0  = 2000.0;
# Angle of rotation measured from the equator.
alpha = 0.0
# Speed of rotation in meters/second
u0    = 20.0
# Mean radius of the earth (meters).
a     = 6.37122e6
# Rotation rate of the earth (1/seconds).
omega = 7.292e-5
g     = 9.80616
# Initial condition for the geopotential field (m^2/s^2).
gh0   = g*5960
# Set to nplt=1 if you want to plot results at different time-steps.
ndsply = 1
nplt   = 0

# Setup tc5 case
print(">>Setup testcase ...")
x, y, z, la, th, r, p_u, p_v, p_w, f, ghm = setupT5(nfile)

print(">>Compute RBF matrix ...")
indices, weightsDx, weightsDy, weightsDz, weightsL =\
    rbfmatrix_fd_hyper(nfile, p_u, p_v, p_w, ep, fdsize, order, dim)

weightsL = gamma * weightsL

print(">>Compute initial condition ...")
uc, gh, us = computeInitialCondition(nfile)

# Compute the inner energy. Note that gh is negative
#tot_inner_energy = np.sum((0.5*(uc**2) - gh))
tot_inner_energy, tot_mass = computeMetric(uc, gh)
#tot_mass         = np.sum( - gh))
print("Total inner energy=",tot_inner_energy, "\tTotal mass=",tot_mass)
#print("Total dynamic energy=", np.sum(uc**2))
#print("Total potential energy=", np.sum(gh**2))

if nplt == 1:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.ion()
    plt.show() 
 
#=============================Define Graph=============================
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("Input"):   
        U       = tf.placeholder(dtype=tf.float64, shape=(N,3), name="U")
        H       = tf.placeholder(dtype=tf.float64, shape=(N,1), name="H")
        
        X       = tf.constant(nfile[:,0:3], shape=(N,3), dtype=tf.float64, name="X")        
        f       = tf.constant(f, dtype=tf.float64, name="f")
        g       = tf.constant(g, dtype=tf.float64, name="g")
        a       = tf.constant(a, dtype=tf.float64, name="a")
        gh0     = tf.constant(gh0, dtype=tf.float64, name="gh0")
        ghm     = tf.constant(ghm, dtype=tf.float64, name="ghm")
        indices = tf.constant(indices, dtype=tf.int64, name="indices")
        weightsDx=tf.Variable(weightsDx, dtype=tf.float64, trainable=True, name="weightsDx")
        weightsDy=tf.Variable(weightsDy, dtype=tf.float64, trainable=True,name="weightsDy")
        weightsDz=tf.Variable(weightsDz, dtype=tf.float64, trainable=True,name="weightsDz")
        weightsL =tf.Variable(weightsL, dtype=tf.float64, trainable=False,name="weightsL")
        p_u     = tf.constant(p_u, dtype=tf.float64, name="p_u")
        p_v     = tf.constant(p_v, dtype=tf.float64, name="p_v")
        p_w     = tf.constant(p_w,dtype=tf.float64, name="p_w")
        dt      = tf.constant(dt, dtype=tf.float64, name="dt")
        learning_rate= tf.constant(learning_rate, dtype=tf.float64, name="learning_rate")
        beta1   = tf.constant(beta1, dtype=tf.float64, name="beta1")
        weight  = tf.constant(weight, dtype=tf.float64, name="weight4loss")


        #dt      = tf.Variable(dt, dtype=tf.float64, name="dt")
        tot_en  = tf.constant(tot_inner_energy, dtype=tf.float64, name="energy")
        tot_ma  = tf.constant(tot_mass, dtype=tf.float64, name="mass")

        DPx     = tf.SparseTensor(indices=indices, values=weightsDx, dense_shape=[N,N])
        DPy     = tf.SparseTensor(indices=indices, values=weightsDy, dense_shape=[N,N])
        DPz     = tf.SparseTensor(indices=indices, values=weightsDz, dense_shape=[N,N])
        L       = tf.SparseTensor(indices=indices, values=weightsL,  dense_shape=[N,N])       

        # Compute the projected gradient of the mountain for test case 5        
        #gradghm= tf.stack(tf.sparse_tensor_dense_matmul(DPx,ghm)/a,\
        #                  tf.sparse_tensor_dense_matmul(DPy,ghm)/a,\
        #                  tf.sparse_tensor_dense_matmul(DPz,ghm)/a)   
        gradghm1 = tf.sparse_tensor_dense_matmul(DPx,ghm)/a
        gradghm2 = tf.sparse_tensor_dense_matmul(DPy,ghm)/a
        gradghm3 = tf.sparse_tensor_dense_matmul(DPz,ghm)/a
        gradghm  = tf.concat((gradghm1, gradghm2, gradghm3), axis=1)
       

        U1 = U
        H1 = H
        d1_u, d1_h = evalCartRhs_fd(U1,H1,DPx,DPy,DPz,L,X,f,g,a,gh0,p_u,p_v,p_w,gradghm,dt)
        
        U2 = U + 0.5*d1_u*dt
        H2 = H + 0.5*d1_h*dt
        d2_u, d2_h = evalCartRhs_fd(U2,H2,DPx,DPy,DPz,L,X,f,g,a,gh0,p_u,p_v,p_w,gradghm,dt)

        
        U3 = U + 0.5*d2_u*dt
        H3 = H + 0.5*d2_h*dt
        d3_u, d3_h = evalCartRhs_fd(U3,H3,DPx,DPy,DPz,L,X,f,g,a,gh0,p_u,p_v,p_w,gradghm,dt)


        U4 = U + d3_u*dt
        H4 = H + d3_h*dt
        d4_u, d4_h = evalCartRhs_fd(U4,H4,DPx,DPy,DPz,L,X,f,g,a,gh0,p_u,p_v,p_w,gradghm,dt)
        
        U_next = U + 1/6*(d1_u + 2*d2_u + 2*d3_u + d4_u)*dt;
        H_next = H + 1/6*(d1_h + 2*d2_h + 2*d3_h + d4_h)*dt;
        
        value = (H_next+gh0)/g

        #loss = tf.abs((tot_en - tf.reduce_sum(0.5*tf.square(U_next) - H_next))/tot_en)
        #loss = tf.square((tot_en - tf.reduce_sum(0.5*tf.square(U_next) - H_next)))
        t_energy,t_mass=computeMetricTensor(U_next, H_next)
        delta_en= tf.abs(tot_en-t_energy)
        delta_ma= tf.abs(tot_ma-t_mass)
        #flag = tf.cast(delta_en>1.0, tf.float64)
        #loss = flag*tf.square(delta) +(1-flag) * tf.sqrt(delta)
        #loss = flag*tf.square(delta) +(1-flag) * delta
        #loss = delta ** 4
        
        #loss = tf.square(delta_en+delta_ma)
        #loss = tf.square(delta_en)+tf.square(delta_ma)*100
        #loss = tf.square(delta_en)+100000*tf.square(delta_ma)
        #weight = tot_en/tot_ma
        loss = tf.square(delta_en)+tf.square(weight*delta_ma)
        #loss = tf.square(delta_en+weight*delta_ma)
        #loss = tf.square(delta_en)+tf.square(delta_ma)
        #loss = tf.square(delta_en)
        #loss = tf.square(delta_en)
        #loss = tf.abs(delta_en)
        #loss = tf.square(delta_ma)
        #loss = tf.square(tot_en - t_energy)
        #the optimize item is energy conservation
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)

        init = tf.global_variables_initializer()

#=============================Start Session=============================
with tf.Session(graph=graph) as sess:
    sess.run(init)
    
    feed_dict_train={ U: uc, H: gh}
    #feed_dict_predict={ U: uc, H: gh}
    feed_dict_test={ U: uc, H: gh}
  
    U_train = uc
    H_train = gh
    tic()

    #for nt in range(tend*24*3600):
    for nt in range(1,100):
        #=============Train Phase==================
        for train_step in range(1,500):
            [optimizer_train,delta_en_train,delta_ma_train,loss_train,U_train,H_train,value_train] = sess.run([optimizer, delta_en, delta_ma, loss, U_next, H_next, value], feed_dict= feed_dict_train)
            #print(">>>>en_train:",train_step, "\tloss_train=",loss_train, "\tdelta_en_train=",delta_en_train, "\tdelta_ma_train=",delta_ma_train)
            #if loss_train < 2e-2:
            #if loss_train < 2e-2:
            if delta_en_train < 1e0 and delta_ma_train < 1e0:
                break
       
        #=============One step prediction==================
        #[delta_en_train,delta_ma_train,loss_train,U_train,H_train,value_train] = sess.run([delta_en, delta_ma, loss, U_next, H_next, value], feed_dict= feed_dict_train)

        feed_dict_train={ U: U_train, H: H_train}
        #print(">>step:",nt, "\tloss=",loss_train, "\tdelta_en=",delta_en_train, "\tdelta_ma=",delta_ma_train)
        t_energy, t_mass = computeMetric(U_train, H_train)
        print(">>step:",nt, "\ttrain:",train_step, "\tloss=",loss_train, "\tdelta_en=",delta_en_train, "\tdelta_ma=",delta_ma_train)

        #t_energy,t_mass = computeMetric(U_train, H_train)
        #print(">>step:",nt, "\tt_energy=",t_energy, "\tloss=",loss_train, "\tdelta_en=",delta_en_train, "\tdelta_ma=",delta_ma_train)
    
        # plot the results
        if nplt == 1 and nt % ndsply==0:
                ax.cla()
                ax.tricontour(la.reshape(N), th.reshape(N), value_train.reshape(N), 14, linewidths=0.5, colors='k')
                cntr  = ax.tricontourf(la.reshape(N), th.reshape(N), value_train.reshape(N), 14, cmap="RdBu_r")
                #fig.colorbar(cntr, ax=ax)
                #ax.plot(la, th, 'ko', ms=3)
                ax.axis((-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
                ax.set_title('tricontour (%d points)' % nt)
                plt.pause(0.01)

    toc()
        
    #=============Predict Phase==================
    #for nt in range(1000):
    #    [loss_test,U_test,H_test,value_test] = sess.run([loss, U_next, H_next, value], feed_dict= feed_dict_test)
    #    feed_dict_test={ U: U_test, H: H_test}
    #    t_energy = computeMetric(U_test, H_test)
    #    print("step_test:",nt, "\tt_energy=",t_energy, "\tloss_test=",loss_test)
 
sess.close()
