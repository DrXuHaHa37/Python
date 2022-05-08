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


def draw_with_text(x, y, title, additionList):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, y)
    ax.set_title(title)
    # 设置x和y轴标签
    ax.set_xlabel('generations')
    ax.set_ylabel('loss')

    # 在子图上添加文本
    ax.text(1, len(additionList), '\n'.join(additionList),
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    # ax.text(3, 8, 'boxed italics text in data coords', style='italic',
    #         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.show()


x = np.linspace(1, 10, 100)
y = x * x
draw_with_text(x, y, 'this is title', ['111', '222', '333'])




