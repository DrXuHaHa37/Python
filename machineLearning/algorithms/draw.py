from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from functions import UNIMODAL, MULTIMODAL


def draw_convergence(npArr):
    plt.plot(npArr)
    plt.ylabel('convergence')
    plt.show()


def f(x, y, r):
    for i in range(x.shape[0]):
        r[i] = x[i] ** 2 + y[i] ** 2
    return r


def draw_function_plot():
    x = np.linspace(-5, 5, 30)
    y = np.linspace(-5, 5, 30)
    X, Y = np.meshgrid(x, y)
    result = X.copy()
    fx = f(X, Y, result)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, fx, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def draw_scatter_plot(matrix, modal, k):
    # {matrix of (x, y), 2-dimensions},
    # {function type in class MODAL},
    # {k-th function in MODAL}
    mtx = np.asarray(matrix)
    xdata = mtx[:, 0]
    ydata = mtx[:, 1]
    zdata = modal(mtx, k).returnValue
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
    plt.show()






