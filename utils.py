
# coding: utf-8

# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def draw_3D(y, n):


    fig = plt.figure(1)
    ax = Axes3D(fig)
    X = np.arange(0, n+2, 1)
    #X = np.arange(-4, 4, 0.25)
    Y = np.arange(0, n+2, 1)
    X, Y = np.meshgrid(X, Y)
    Z = y

    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

