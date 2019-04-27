# coding: utf-8
from np import *

def activation(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y-delta)) / batch_size



def to_one_hot_label(X, outsize, classsize):
    if outsize < classsize:
        T = to_one_hot_label2(X, outsize)
    else:
        T = to_one_hot_label1(X, outsize)
    return T

def to_one_hot_label2(X, outsize):
    T = np.zeros((X.size, outsize))
    for i in range(X.size):
        if X[i] == 0:
            T[i] = 0.01
        else:
            T[i, X[i]-1] = 1
    return T

def to_one_hot_label1(X, outsize):
    T = np.zeros((X.size, outsize))
    for i in range(X.size):
        T[i, X[i]] = 1
    return T


def mean_squared_error(y, t):
    batch_size = y.shape[1]
    output_size = y.shape[0]
    return np.sum( (y - t) **2 ) / output_size / batch_size

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1), dtype='complex128')
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def normalize_ax1(x): # x.shape must be 2 dimentional
    x_max = np.max(x, axis=1).reshape((x.shape[0], -1)) + 1e-5
    x_min = np.min(x, axis=1).reshape((x.shape[0], -1))
    x_n = (x - x_min) / (x_max - x_min)
    return x_n

def autocorr(x, normalize=True): # x.shape = (N,length)
    if normalize:
        norm = np.sum(x**2, axis=1)
    acorr = []
    for n in range(x.shape[0]):
        cor = np.correlate(x[n], x[n], mode='same')
        if normalize:
            cor /= norm[n]
        acorr.append(cor)
    acorr = np.array(acorr)
    return acorr
