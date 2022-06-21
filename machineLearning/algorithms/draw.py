from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from algorithms.functions import UNIMODAL, MULTIMODAL
from matplotlib.pyplot import plot, savefig
import datetime


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


def draw_with_text(x, y, title= '111', additionList=[]):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, y)
    ax.set_title(title)
    # 设置x和y轴标签
    ax.set_xlabel('generations')
    ax.set_ylabel('loss')
    # plt.ylim(0, int(2000))

    # 在子图上添加文本
    ax.text(1, len(additionList), '\n'.join(additionList),
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})
    plt.show()


# {method1:[x1, y1], method2:[x2,y2]}
def draw_multi_lines(method, title):
    fig, ax = plt.subplots()
    for i in method:
        ax.plot(method[i][0], method[i][1], label=i)

    ax.set_xlabel('sample')
    ax.set_ylabel('vpc')
    ax.set_title(title)

    ax.legend()
    t = datetime.datetime.now()
    timeString = str(t).split('.')[0].split(' ')
    timeString[1] = ''.join(timeString[1].split(':'))
    timeString = '_'.join(timeString) + '.jpg'
    savefig("D:/Documents/gPaper/predict/" + timeString)
    # plt.show()
    plt.clf()


# x = np.linspace(-1, 1, 10)
# y1 = x
# y2 = x ** 2
#
# d = dict()
#
# d['line'] = [x, y1]
# d['pow'] = [x, y2]
#
# draw_multi_lines(d)




