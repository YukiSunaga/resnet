# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import numpy.matlib
from functions import *
from util import im2col, col2im
import optimizer as op


pool = True
backpro = 0
beta1=0.9
beta2=0.999


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False, lr=0.01, outdim1=True):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.hs = None
        self.stateful = stateful

        self.optimizer = op.Adam(lr=lr)
        self.outdim1 = outdim1

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        self.hs = hs

        if self.outdim1:
            return self.h
        else:
            return hs

    def backward(self, dhs):
        if self.outdim1:
            dhse = np.zeros_like(self.hs)
            dhse[:,-1,:] = dhs
            dhs = dhse
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]



        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        self.optimizer.update(self.params, self.grads)

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Fourier_unit_sin:
    def __init__(self, T):
        self.first_flg = True
        self.T = T
        self.bwave = None

    def forward(self, x, train_flg):
        if self.first_flg:
            self.bwave = np.sin( 2*np.pi / self.T * np.arange(x.shape[1]))
            self.first_flg = False
        y = x * self.bwave
        y = np.sum(y, axis=1).reshape((x.shape[0], 1))
        y = np.abs(y)

        return y

    def backward(self, dout):
        return dout


class Fourier_unit_cos:
    def __init__(self, T):
        self.first_flg = True
        self.T = T
        self.bwave = None

    def forward(self, x, train_flg):
        if self.first_flg:
            self.bwave = np.cos( 2*np.pi / self.T * np.arange(x.shape[1]))
            self.first_flg = False
        y = x * self.bwave
        y = np.sum(y, axis=1).reshape((x.shape[0], 1))
        y = np.abs(y)

        return y

    def backward(self, dout):
        return dout

class Fourier:
    def __init__(self, T_l, T_h, T_step=1, normalize=False):
        self.T_l = T_l          # sin wave of the fourier units have cycles [T_l, T_h]
        self.T_h = T_h
        self.T_step = T_step
        self.F_units = []
        for i in range( int((T_h - T_l + 1) / T_step) ):
            T = T_l + i * T_step
            self.F_units.append(Fourier_unit_sin(T))
            self.F_units.append(Fourier_unit_cos(T))
        self.normalize=normalize

    def forward(self, x, train_flg=False):
        first = True
        for i, unit in enumerate(self.F_units):
            if first:
                y = unit.forward(x, train_flg)
                first = False
            else:
                y = np.concatenate((y, unit.forward(x, train_flg)), axis=1)
        if self.normalize:
            y = normalize_ax1(y)
        return y

    def backward(self, dout):
        return dout


class Fourier_np: # If input shape is (N, l), output shape is (N, int(l/2) + 1)
    def forward(self, x, train_flg=False):
        y = np.fft.rfft(x)
        y = np.abs(y)
        return y

    def backward(self, dout):
        return dout



class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None, optimizer='SGD', lr=0.01):
        self.gamma = gamma.astype(dtype)
        self.beta = beta.astype(dtype)
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

        self.optimizer = optimizer
        self.lr = lr

        self.iter = 0
        self.gm = None
        self.gv = None
        self.bm = None
        self.bv = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)

        if self.optimizer == 'SGD':
            self.gamma -= self.lr * self.dgamma
            if self.bias:
                self.beta -= self.lr * self.dbeta

            if self.optimizer == 'Adam':
                if self.gm is None:
                    self.gm = np.zeros_like(self.dgamma)
                    self.gv = np.zeros_like(self.dgamma)
                self.iter += 1
                lr_t  = self.lr * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
                self.gm += (1 - beta1) * (self.dgamma - self.gm)
                self.gv += (1 - beta2) * (self.dgamma**2 - self.gv)
                self.gamma = self.gamma - lr_t * self.gm / (np.sqrt(self.gv) + 1e-7)


                if self.bm is None:
                    self.bm = np.zeros_like(self.dbeta)
                    self.bv = np.zeros_like(self.dbeta)
                self.bm += (1 - beta1) * (self.dbeta - self.bm)
                self.bv += (1 - beta2) * (self.dbeta**2 - self.bv)
                self.beta = self.beta - lr_t * self.bm / (np.sqrt(self.bv) + 1e-7)


        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class MaxNorm:
    def __init__(self, max=10):
        self.m = max

    def maxnorm(self, W):
        if W.ndim == 2:
            S = np.sum(W**2,axis=0)
            W[:,S>self.m] = W[:,S>self.m] * np.sqrt( self.m / S[S>self.m] )
            return W
        else:
            S = np.sum(np.sum(np.sum(W**2, axis=3), axis=2), axis=1)
            W[S>self.m] = W[S>self.m] * np.sqrt( self.m / S[S>self.m] )
            return W


class GAP:  # Global Average Pooling : connection from the last conv. layer to the first MLP layer
    def __init__(self):
        self.x = None
        self.x_shape = None

    def forward(self,x):
        self.x = x
        if self.x.ndim != 4:
            return x

        self.x_shape = x.shape
        y = x.mean(axis=2).mean(axis=2)

        return y

    def backward(self,dout):
        if len(self.x_shape) != 4:
            return dout
        dout = dout.reshape((dout.shape[0], dout.shape[1], 1, 1))
        dx = np.zeros(self.x_shape)
        dx += 1 / (self.x_shape[2]*self.x_shape[3]) * dout

        return dx


class GMP:  # Global Max Pooling : connection from the last conv. layer to the first MLP layer
    def __init__(self):
        self.x_shape = None
        self.arg_max = None

    def forward(self,x):
        if self.x.ndim != 4:
            return x

        self.x_shape = x.shape
        x = x.reshape((self.x_shape[0],self.x_shape[1],-1))
        self.arg_max = x.argmax(axis=2)
        y = x.max(axis=2).reshape(((self.x_shape[0],self.x_shape[1],1,1)))

        return y

    def backward(self,dout):
        if len(self.x_shape) != 4:
            return dout
        dx = np.zeros(self.x_shape)
        dx = dx.reshape((dx.shape[0], dx.shape[1], -1))
        for n in range(dx.shape[0]):
            dx[c, np.arange(self.argmax.shape[1]), self.argmax[c]] = dout[c].flatten()

        return dx


class DiffLayer:
    def forward(self, x):
        ax = len(x.shape) - 1
        x_r = np.roll(x, -1, axis=ax)
        x_diff = x_r - x
        if ax == 1:
            x_diff[:,-1] = 0
        elif ax == 3:
            x_diff[:,:,:,-1] = 0
        return x_diff
    def backward(self, dout):
        return dout

class Diff2Layer:
    def __init__(self):
        self.diff = DiffLayer()
    def forward(self, x):
        ax = len(x.shape) - 1
        x_diff2 = self.diff.forward( self.diff.forward(x) )
        if ax == 1:
            x_diff2[:,-1] = 0
            x_diff2[:,-2] = 0
        elif ax == 3:
            x_diff2[:,:,:,-1] = 0
            x_diff2[:,:,:,-2] = 0
        return x_diff2
    def backward(self, dout):
        return dout



'''
class LSTM: # T blocks of LSTM units
    def __init__(self, T, W):
        self.W = W #W: np.array( (T, ) )

        self.x = None
        self.y = None
        self.c = None
        self.i = None

        self.T = T

    def forward(self, x):
        for t in range(self.T):
            self.i[t] = sigmoid(  )
'''

class Flatten_layer:
    def __init__(self):
        self.x = None
        self.original_x_shape = None

    def forward(self, x, train_flg=False):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        return x

    def backward(self, dout): # この層の教師出力tから前層の教師出力t_0を計算
        dout = dout.reshape(*self.original_x_shape)

        return dout



class FreqUnit:
    def __init__(self, W_unit, T, bias=False, b=None, lr=0.01, activation='Relu', optimizer='SGD', batchnorm=False):
        self.W_unit = W_unit
        self.W_unit[self.W_unit==0] += 1e-7
        self.win_size = W_unit.shape[2]
        self.bias = bias
        if self.bias:
            self.b = b
        else:
            self.b = 0

        self.T = T

        self.N = int(T - 1)
        self.first_flg = True

        self.W = np.zeros((self.N, W_unit.shape[0], 1, self.T))   # similar to conv. kernels
        for n in range(self.N):
            W_uni = self.W_unit
            if n+self.win_size > self.W.shape[3]:
                W_uni = 0
            self.W[n, :, :, n:n+self.win_size] = W_uni
        self.W_mask = (self.W==0)     # index that its weights fix to zero


        self.activation = activation
        self.mask=None

        self.x = None       # 入力
        self.x_col = None   # im2colしたx
        self.u = None       # Convolutionを取った後
        self.u_col = None   # im2colしたu
        self.u_sum = None
        self.conv_y = None  # self.uを活性化関数に通した後
        self.conv_y_col = None # im2colしたself.conv_y
        self.pool_x_col = None # Poolingの入力(=conv_y)を（Pooling用の形状に）im2colしたもの
        self.pool_y = None  # self.conv_yをPoolingした後
        self.arg_max = None
        self.y = None

        self.batchnorm = batchnorm

        self.lr = lr
        self.optimizer = optimizer
        self.dW = None
        if self.bias:
            self.db = None
        self.iter = 0
        self.m = None
        self.v = None
        self.bm = None
        self.bv = None


    def W_update0(self, amax):
        w_unit = self.W[amax, 0, 0, amax:amax+self.win_size]
        for n in range(self.N):
            W_uni = w_unit
            if n+self.win_size > self.W.shape[3]:
                W_uni = 0
            self.W[n, 0, 0, n:n+self.win_size] = W_uni

    def W_update(self, amax):
        w_unit = np.zeros_like(self.W_unit)
        for n in range(amax.size):
            if not amax[n]+self.win_size > self.W.shape[3]:
                w_unit += self.W[amax[n], :, :, amax[n]:amax[n]+self.win_size]
        w_unit /= amax.size
        for n in range(self.N):
            W_uni = w_unit
            if n+self.win_size > self.W.shape[3]:
                W_uni = 0
            self.W[n, :, :, n:n+self.win_size] = W_uni

    def forward(self, x, train_flg=False):
        # Convolution
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        if self.first_flg:
            self.W /= np.sqrt(self.T / self.win_size)
            self.first_flg = False

        out_h = 1 + int((H - FH)/self.T)
        out_w = 1 + int((W - FW)/self.T)

        col = im2col(x, FH, FW, self.T, 0)
        col_W = self.W.reshape(FN, -1).T

        self.u_col = np.dot(col, col_W) + self.b
        self.u = self.u_col.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.x_col = col
        self.W_col = col_W

        self.u_sum = np.sum(self.u, axis=3).reshape((N,self.N))

        if self.activation == 'tanh':
            self.conv_y = np.tanh(self.u_sum)
        elif self.activation == 'Relu':
            self.mask = (self.u_sum <= 0)
            self.conv_y = self.u_sum
            self.conv_y[self.mask] = 0
        elif self.activation == 'None':
            self.conv_y = self.u_sum
        else:
            print('Error : Activation function ' + self.activation + ' is undefined.')


        self.arg_max = np.argmax(self.conv_y, axis=1)
        y = np.max(self.conv_y, axis=1)
        self.pool_y = y
        self.y = self.u

        return y

    def backward(self, dout):
        dout2 = np.zeros((dout.size, self.N))
        dout2[np.arange(dout.size), self.arg_max] = dout.flatten()
        dout3 = np.zeros(self.u.shape)
        dout3[:, np.arange(self.u.shape[1])] = dout2[:, np.arange(self.u.shape[1])].T.reshape(dout2.shape+(1,1))
        dout2 = dout3


        # Conv backward
        FN, C, FH, FW = self.W.shape

        if self.activation == 'tanh':
            dout2 = dout2 * (1 - (np.tanh(self.u))**2)
        elif self.activation == 'Relu':
            dout2[self.mask] = 0
        elif self.activation == 'None':
            pass


        FN, C, FH, FW = self.W.shape
        dout2 = dout2.transpose(0,2,3,1).reshape(-1, FN)

        if self.bias:
            self.db = np.sum(dout2, axis=0)
        self.dW = np.dot(self.x_col.T, dout2)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dconv = np.dot(dout2, self.W_col.T)
        dx = col2im(dconv, self.x.shape, FH, FW, self.T, 0)

        # update
        if self.optimizer == 'SGD':
            self.dW = np.sum(self.dW, axis=0)
            self.W = self.W - self.lr * self.dW
            if self.bias:
                self.b = self.b - self.lr * self.db
        if self.optimizer == 'Adam':
            if self.m is None:
                self.m = np.zeros_like(self.dW)
                self.v = np.zeros_like(self.dW)
            self.iter += 1
            lr_t  = self.lr * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
            self.m += (1 - beta1) * (self.dW - self.m)
            self.v += (1 - beta2) * (self.dW**2 - self.v)
            deltaW = np.sum(self.m / (np.sqrt(self.v) + 1e-7), axis=0) / self.m.shape[0]
            self.W = self.W - lr_t * deltaW
        self.W[self.W_mask] = 0
        self.W_update(self.arg_max)

        if self.bias:
            if self.bm is None:
                self.bm = np.zeros_like(self.db)
                self.bv = np.zeros_like(self.db)
            self.bm += (1 - beta1) * (self.db - self.bm)
            self.bv += (1 - beta2) * (self.db**2 - self.bv)
            self.b = self.b - lr_t * self.bm / (np.sqrt(self.bv) + 1e-7)


        return dx


    def return_W(self):
        return self.W[0, :, :, 0:self.win_size]

    def return_b(self):
        return self.b

    def set_W(self, W):
        if W.ndim == 1:
            W = W.reshape((1,1,-1))
        self.W_unit = W
        self.W_unit[self.W_unit==0] += 1e-7
        self.win_size = W.shape[2]
        self.W = np.zeros((self.N, W.shape[0], 1, self.T))   # similar to conv. kernels
        for n in range(self.N):
            W_uni = self.W_unit
            if n+self.win_size > self.W.shape[3]:
                W_uni = 0
            self.W[n, :, :, n:n+self.win_size] = W_uni
        self.W_mask = (self.W==0)     # index that its weights fix to zero

    def set_b(self,b):
        self.b = b



class FreqLayer:
    def __init__(self, Tl, Th, T_step, W_unit, unit_num, bias=False, b=None, lr=0.01, activation='Relu', optimizer='SGD', batchnorm=False):
        self.unit_num = unit_num #int((Th - Tl) / T_step + 1)
        self.W_unit = W_unit # (unit_num, W_unit_size)
        self.unit_list = []
        for u in range(self.unit_num):
            T = Tl + u * T_step
            self.unit_list.append(FreqUnit(W_unit[u], T, bias, b ,lr, activation, optimizer, batchnorm))
        self.x_shape = None

    def forward(self, x, train_flg):
        N, C, H, W = x.shape
        self.x_shape = x.shape
        y = np.zeros((self.unit_num, N))
        for u in range(self.unit_num):
            y[u] = self.unit_list[u].forward(x, train_flg)
        y = y.T

        return y

    def backward(self, dout):
        dx = np.zeros(self.x_shape)
        for u in range(self.unit_num):
            dx = self.unit_list[u].backward(dout[:,u])

        return dx

    def return_W(self):
        W = []
        for u in range(self.unit_num):
            W.append(self.unit_list[u].return_W())
        return W

    def return_b(self):
        b = []
        for u in range(self.unit_num):
            b.append(self.unit_list[u].return_b())
        return b

    def set_W(self, W_list):
        for u in range(self.unit_num):
            self.unit_list[u].set_W(W_list[u])

    def set_b(self, b_list):
        for u in range(self.unit_num):
            self.unit_list[u].set_b(b_list[u])



class MulFreqLayer:
    def __init__(self, Tl, Th, T_step, W_units, unit_num, K_num, bias=False, b=None, lr=0.01, activation='Relu', optimizer='SGD', batchnorm=False):
        self.unit_num = unit_num #int((Th - Tl) / T_step + 1)
        self.W_units = W_units
        self.K_num = K_num
        self.F_layers = []
        if bias:
            for i in range(K_num):
                self.F_layers.append(FreqLayer(Tl,Th,T_step, W_units[i], unit_num, bias, b[i], lr, activation, optimizer))
        else:
            for i in range(K_num):
                self.F_layers.append(FreqLayer(Tl,Th,T_step, W_units[i], unit_num, bias, None, lr, activation, optimizer))



    def forward(self, x, train_flg):
        N, C, H, W = x.shape
        self.x_shape = x.shape
        y = np.zeros((N, self.unit_num * self.K_num))
        flg = True
        for k in range(self.K_num):
            if flg:
                y = self.F_layers[k].forward(x, train_flg)
                flg = False
            else:
                y = np.concatenate((y, self.F_layers[k].forward(x, train_flg)), axis=1)

        return y

    def backward(self, dout):
        dx = np.zeros(self.x_shape)
        for k in range(self.K_num):
            dx += self.F_layers[k].backward(dout[:,k*self.unit_num:(k+1)*self.unit_num])

        return dx

    def return_W(self):
        W = []
        for u in range(len(self.F_layers)):
            W.append(self.F_layers[u].return_W())
        return W

    def return_b(self):
        b = []
        for u in range(len(self.F_layers)):
            b.append(self.F_layers[u].return_b())
        return b

    def set_W(self, W_list):
        for u in range(len(self.F_layers)):
            self.F_layers[u].set_W(W_list[u])

    def set_b(self, b_list):
        for u in range(len(self.F_layers)):
            self.F_layers[u].set_b(b_list[u])


class RBF:
    def __init__(self, w, lr=0.01, optimizer='SGD', batchnorm=False, dropout=False, maxnorm = False):
        self.w = w

        self.x = None
        self.original_x_shape = None
        self.u = None
        self.y = None    #x->(Affine)->u->(Tanh)->y
        # 重みの微分
        self.dW = None


        self.optimizer = optimizer

        self.iter = 0
        self.m = None
        self.v = None

    def forward(self, x, train_flg=False):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)


        self.x = x

        u = (x - self.w)**2
        self.u = u

        self.y = np.exp( -np.sum(u, axis=1).reshape((u.shape[0], -1)) )
        y = self.y


        return y

    def backward(self, dout): # この層の教師出力tから前層の教師出力t_0を計算
        dout[self.mask] = 0
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)


        if self.optimizer == 'SGD':
            self.W -= self.lr * self.dW

            if self.optimizer == 'Adam':
                if self.m is None:
                    self.m = np.zeros_like(self.dW)
                    self.v = np.zeros_like(self.dW)
                self.iter += 1
                lr_t  = self.lr * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
                self.m += (1 - beta1) * (self.dW - self.m)
                self.v += (1 - beta2) * (self.dW**2 - self.v)
                self.W = self.W - lr_t * self.m / (np.sqrt(self.v) + 1e-7)


        dx = dx.reshape(*self.original_x_shape)

        return dx

    def return_W(self):
        return self.W

    def return_b(self):
        return -1

    def set_W(self, W):
        self.W = W

    def set_b(self, b):
        a=1


class SelfCorrelation:
    def __init__(self, normalize=True):
        self.original_x_shape = None
        self.normalize = normalize
    def forward(self, x, train_flg=False): # x.shape = (N,L)
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        y = x - np.mean(x, axis=1).reshape((x.shape[0], 1))
        if self.normalize:
            norm = np.sum(y**2, axis=1) + 1e-7
        corr = []
        for n in range(x.shape[0]):
            ac = np.correlate(y[n], y[n], 'same')
            if self.normalize:
                ac /= norm[n]
            else:
                ac /= self.original_x_shape[-1]
            corr.append(ac)
        corr = np.array(corr)
        return corr
    def backward(self, dout):
        return dout



class AffineRelu:
    def __init__(self, W, bias=False, b=None, lr=0.01, optimizer='SGD', batchnorm=False, dropout=False, maxnorm = False):
        self.W = W
        self.bias = bias
        if self.bias:
            self.b = b
        else:
            self.b = 0

        self.dropout=dropout
        if self.dropout:
            self.DO = Dropout()
        self.maxnorm = maxnorm
        if self.maxnorm:
            self.MN = MaxNorm()

        self.x = None
        self.original_x_shape = None
        self.u = None
        self.y = None    #x->(Affine)->u->(Tanh)->y
        # 重みの微分
        self.dW = None
        if self.bias:
            self.db = None

        self.mask = None

        self.optimizer = optimizer

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.BN = None
            self.BN_firstflg = True

        self.iter = 0
        self.m = None
        self.v = None
        self.bm = None
        self.bv = None

    def forward(self, x, train_flg=False):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        if self.batchnorm:
            if self.BN_firstflg:
                gamma = np.ones(self.W.shape[0])
                beta = np.zeros(self.W.shape[0])
                self.BN = BatchNormalization(gamma, beta, optimizer=self.optimizer)
                self.BN_firstflg = False
                x = self.BN.forward(x, train_flg)
            else:
                x = self.BN.forward(x, train_flg)


        self.x = x

        u = np.dot(self.x, self.W) + self.b
        self.u = u

        self.mask = (u <= 0)
        self.y = self.u.copy()
        self.y[self.mask] = 0
        y = self.y

        if self.dropout:
            y = self.DO.forward(y, train_flg)

        return y

    def backward(self, dout): # この層の教師出力tから前層の教師出力t_0を計算
        if self.dropout:
            dout = self.DO.backward(dout)
        dout[self.mask] = 0
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        if self.bias:
            self.db = np.sum(dout, axis=0)


        if self.optimizer == 'SGD':
            self.W -= self.lr * self.dW
            if self.bias:
                self.b -= self.lr * self.db

            if self.optimizer == 'Adam':
                if self.m is None:
                    self.m = np.zeros_like(self.dW)
                    self.v = np.zeros_like(self.dW)
                self.iter += 1
                lr_t  = self.lr * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
                self.m += (1 - beta1) * (self.dW - self.m)
                self.v += (1 - beta2) * (self.dW**2 - self.v)
                self.W = self.W - lr_t * self.m / (np.sqrt(self.v) + 1e-7)

                if self.bias:
                    if self.bm is None:
                        self.bm = np.zeros_like(self.db)
                        self.bv = np.zeros_like(self.db)
                    self.bm += (1 - beta1) * (self.db - self.bm)
                    self.bv += (1 - beta2) * (self.db**2 - self.bv)
                    self.b = self.b - lr_t * self.bm / (np.sqrt(self.bv) + 1e-7)

        if self.maxnorm:
            self.W = self.MN.maxnorm(self.W)

        if self.batchnorm:
            dx = self.BN.backward(dx)

        dx = dx.reshape(*self.original_x_shape)

        return dx

    def return_W(self):
        return self.W

    def return_b(self):
        return self.b

    def set_W(self, W):
        self.W = W

    def set_b(self, b):
        self.b = b


class AffineTanh:
    def __init__(self, W, bias=False, b=None, lr=0.01, optimizer='SGD', batchnorm=False, dropout=False, maxnorm = False):
        self.W = W
        self.bias = bias
        if self.bias:
            self.b = b
        else:
            self.b = 0

        self.dropout=dropout
        if self.dropout:
            self.DO = Dropout()
        self.maxnorm = maxnorm
        if self.maxnorm:
            self.MN = MaxNorm()

        self.x = None
        self.original_x_shape = None
        self.u = None
        self.y = None    #x->(Affine)->u->(Tanh)->y
        # 重みの微分
        self.dW = None
        if self.bias:
            self.db = None


        self.optimizer = optimizer

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.BN = None
            self.BN_firstflg = True

        self.iter = 0
        self.m = None
        self.v = None
        self.bm = None
        self.bv = None

    def forward(self, x, train_flg=False):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        if self.batchnorm:
            if self.BN_firstflg:
                gamma = np.ones(self.W.shape[0])
                beta = np.zeros(self.W.shape[0])
                self.BN = BatchNormalization(gamma, beta, optimizer=self.optimizer)
                self.BN_firstflg = False
                x = self.BN.forward(x, train_flg)
            else:
                x = self.BN.forward(x, train_flg)


        self.x = x

        u = np.dot(self.x, self.W) + self.b
        self.u = u

        self.y = np.tanh(self.u)
        y = self.y

        if self.dropout:
            y = self.DO.forward(y, train_flg)

        return y

    def backward(self, dout): # この層の教師出力tから前層の教師出力t_0を計算
        if self.dropout:
            dout = self.DO.backward(dout)
        dout = dout * (1 - self.y ** 2)
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        if self.bias:
            self.db = np.sum(dout, axis=0)


        if self.optimizer == 'SGD':
            self.W -= self.lr * self.dW
            if self.bias:
                self.b -= self.lr * self.db

            if self.optimizer == 'Adam':
                if self.m is None:
                    self.m = np.zeros_like(self.dW)
                    self.v = np.zeros_like(self.dW)
                self.iter += 1
                lr_t  = self.lr * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
                self.m += (1 - beta1) * (self.dW - self.m)
                self.v += (1 - beta2) * (self.dW**2 - self.v)
                self.W = self.W - lr_t * self.m / (np.sqrt(self.v) + 1e-7)

                if self.bias:
                    if self.bm is None:
                        self.bm = np.zeros_like(self.db)
                        self.bv = np.zeros_like(self.db)
                    self.bm += (1 - beta1) * (self.db - self.bm)
                    self.bv += (1 - beta2) * (self.db**2 - self.bv)
                    self.b = self.b - lr_t * self.bm / (np.sqrt(self.bv) + 1e-7)

        if self.maxnorm:
            self.W = self.MN.maxnorm(self.W)

        if self.batchnorm:
            dx = self.BN.backward(dx)

        dx = dx.reshape(*self.original_x_shape)

        return dx

    def return_W(self):
        return self.W

    def return_b(self):
        return self.b

    def set_W(self, W):
        self.W = W

    def set_b(self, b):
        self.b = b


class AffineSoftmaxCE:
    def __init__(self, W, bias=False, b=None, lr=0.01, optimizer = 'SGD', dropout=False):
        self.W = W
        self.bias = bias
        if self.bias:
            self.b = b
        else:
            self.b = 0

        self.dropout = dropout
        if self.dropout:
            self.DO = Dropout()

        self.x = None
        self.original_x_shape = None
        self.u = None
        self.y = None    #x->(Affine)->u->(Tanh)->y
        # 重みの微分
        self.dW = None
        if self.bias:
            self.db = None

        self.lr = lr
        self.optimizer = optimizer
        self.iter = 0
        self.m = None
        self.v = None
        self.bm = None
        self.bv = None


    def forward(self, x, train_flg=False):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x

        u = np.dot(self.x, self.W) + self.b
        self.u = u

        if self.dropout:
            u = self.DO.forward(u, train_flg)


        self.y = softmax(u)

        return self.y

    def backward(self, t): # この層の教師出力tから前層の教師出力t_0を計算
        # Cross Entropy Error の微分も含んでいる
        batch_size = t.shape[0]
        dout = (self.y - t) / batch_size

        if self.dropout:
            dout = self.DO.backward(dout)

        dx = np.dot(dout, self.W.T)
        if self.bias:
            self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.x.T, dout)

        if self.optimizer == 'SGD':
            self.W = self.W - self.lr * self.dW
            if self.bias:
                self.b = self.b - self.lr * self.db
        if self.optimizer == 'Adam':
            if self.m is None:
                self.m = np.zeros_like(self.dW)
                self.v = np.zeros_like(self.dW)
            self.iter += 1
            lr_t  = self.lr * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
            self.m += (1 - beta1) * (self.dW - self.m)
            self.v += (1 - beta2) * (self.dW**2 - self.v)
            self.W = self.W - lr_t * self.m / (np.sqrt(self.v) + 1e-7)

            if self.bias:
                if self.bm is None:
                    self.bm = np.zeros_like(self.db)
                    self.bv = np.zeros_like(self.db)
                self.bm += (1 - beta1) * (self.db - self.bm)
                self.bv += (1 - beta2) * (self.db**2 - self.bv)
                self.b = self.b - lr_t * self.bm / (np.sqrt(self.bv) + 1e-7)


        dx = dx.reshape(*self.original_x_shape)

        return dx

    def return_W(self):
        return self.W

    def return_b(self):
        return self.b

    def set_W(self, W):
        self.W = W

    def set_b(self, b):
        self.b = b

class AffineSoftmaxMS:
    def __init__(self, W, lr=0.01, optimizer = 'SGD'):
        self.W = W

        self.x = None
        self.original_x_shape = None
        self.u = None
        self.y = None    #x->(Affine)->u->(Tanh)->y
        # 重みの微分
        self.dW = None

        self.lr = lr
        self.optimizer = optimizer
        self.iter = 0
        self.m = None
        self.v = None



    def forward(self, x, train_flg=False):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x

        u = np.dot(self.x, self.W)
        self.u = u

        self.y = softmax(self.u)

        return self.y

    def backward(self, t): # この層の教師出力tから前層の教師出力t_0を計算
        # Mean Squared Error の微分も含んでいる
        batch_size, out_size = t.shape
        S = np.matlib.repmat(np.sum(self.u*(t - self.u), axis=1).reshape(batch_size,-1), 1, out_size)
        dout = 2*self.u * (S + (self.u - t)) / out_size / batch_size

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)

        if self.optimizer == 'SGD':
            self.W = self.W - self.lr * self.dW
        if self.optimizer == 'Adam':
            if self.m is None:
                self.m = np.zeros_like(self.dW)
                self.v = np.zeros_like(self.dW)
            self.iter += 1
            lr_t  = self.lr * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
            self.m += (1 - beta1) * (self.dW - self.m)
            self.v += (1 - beta2) * (self.dW**2 - self.v)
            self.W = self.W - lr_t * self.m / (np.sqrt(self.v) + 1e-7)


        dx = dx.reshape(*self.original_x_shape)

        return dx

    def return_W(self):
        return self.W

    def return_b(self):
        return self.b

    def set_W(self, W):
        self.W = W

    def set_b(self,b):
        self.b = b



class ConvPool:
    def __init__(self, W, bias=False, b=None, lr=0.01, activation='Relu', optimizer = 'SGD',
                pool='max', pool_h=2, pool_w=2, conv_stride=1, conv_pad=0, pool_stride=2, pool_pad=0,
                batchnorm=False, pool_or_not=True, gap_layer=False, useDO=False, DO_rate=0.0):
        self.W = W
        self.W_col = None
        self.bias = bias
        if self.bias:
            self.b = b
        else:
            self.b = 0
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad
        self.pool = pool
        self.pool_h = 1
        self.pool_w = pool_w
        self.pool_stride = pool_stride
        self.pool_pad = pool_pad
        self.pool_or_not = pool_or_not

        self.activation = activation
        self.mask=None

        self.x = None       # 入力
        self.x_col = None   # im2colしたx
        self.u = None       # Convolutionを取った後
        self.u_col = None   # im2colしたu
        self.conv_y = None  # self.uを活性化関数に通した後
        self.conv_y_col = None # im2colしたself.conv_y
        self.pool_x_col = None # Poolingの入力(=conv_y)を（Pooling用の形状に）im2colしたもの
        self.pool_y = None  # self.conv_yをPoolingした後
        self.arg_max = None

        self.y = None

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.BN = None
            self.BN_firstflg = True

        self.dropout = useDO
        if self.dropout:
            self.DO = Dropout(dropout_ratio=DO_rate)

        self.gap_layer = gap_layer
        if self.gap_layer:
            self.GAP = GAP()



        self.lr = lr
        self.optimizer = optimizer
        self.dW = None
        if self.bias:
            self.db = None
        self.iter = 0
        self.m = None
        self.v = None
        self.bm = None
        self.bv = None


    def forward(self, x, train_flg=False):
        # Convolution
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        if self.batchnorm:
            if self.BN_firstflg:
                gamma = np.ones((C*H*W))
                beta = np.zeros((C*H*W))
                self.BN = BatchNormalization(gamma, beta, optimizer=self.optimizer)
                self.BN.forward(x, train_flg)
                self.BN_firstflg = False
            else:
                self.BN.forward(x, train_flg)

        out_h = 1 + int((H + 2*self.conv_pad - FH) / self.conv_stride)
        out_w = 1 + int((W + 2*self.conv_pad - FW) / self.conv_stride)

        col = im2col(x, FH, FW, self.conv_stride, self.conv_pad)
        col_W = self.W.reshape(FN, -1).T

        self.u_col = np.dot(col, col_W) + self.b
        self.u = self.u_col.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.x_col = col
        self.W_col = col_W

        if self.activation == 'tanh':
            self.conv_y = np.tanh(self.u)
        elif self.activation == 'Relu':
            self.mask = (self.u <= 0)
            self.conv_y = self.u
            self.conv_y[self.mask] = 0
        else:
            print('Error : Activation function ' + self.activation + ' is undefined.')

        if not self.pool_or_not: y = self.conv_y

        if self.pool_or_not:
            # Pooling
            N, C, H, W = self.conv_y.shape
            out_h = int(1 + (H - self.pool_h) / self.pool_stride)
            out_w = int(1 + (W - self.pool_w) / self.pool_stride)

            self.conv_y_col = im2col(self.conv_y, self.pool_h, self.pool_w, self.pool_stride, self.pool_pad)
            self.conv_y_col = self.conv_y_col.reshape(-1, self.pool_h*self.pool_w)

            if self.pool == 'max':
                arg_max = np.argmax(self.conv_y_col, axis=1)
                self.pool_y = np.max(self.conv_y_col, axis=1)
                self.arg_max = arg_max

            elif self.pool =='avg':
                self.pool_y = np.mean(self.conv_y_col, axis=1)
            self.pool_y = self.pool_y.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
            y = self.pool_y

        self.y = y

        if self.dropout:
            y = self.DO.forward(y, train_flg)
        if self.gap_layer:
            y = self.GAP.forward(y)


        return y

    def backward(self, dout):
        if self.gap_layer:
            dout = self.GAP.backward(dout)
        if self.dropout:
            dout = self.DO.backward(dout)

        if self.pool_or_not:
            # Pooling backward
            dout = dout.transpose(0, 2, 3, 1)
            pool_size = self.pool_h * self.pool_w

            if self.pool == 'max':
                dmax = np.zeros((dout.size, pool_size))
                dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
                dmax = dmax.reshape(dout.shape + (pool_size,))
            elif self.pool == 'avg':
                dmax = np.zeros((dout.size, pool_size))
                for i in range(pool_size):
                    dmax[np.arange(self.conv_y_col.shape[0]), i ] = self.conv_y_col.shape[1] * dout.flatten()
                dmax = dmax.reshape(dout.shape + (pool_size,))

            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            dx = col2im(dcol, self.conv_y.shape, self.pool_h, self.pool_w, self.pool_stride, self.pool_pad)
        else:
            dx = dout
        # Conv backward
        FN, C, FH, FW = self.W.shape

        if self.activation == 'tanh':
            dout2 = dx * (1 - (np.tanh(self.u))**2)
        elif self.activation == 'Relu':
            dout2 = dx
            dout2[self.mask] = 0


        FN, C, FH, FW = self.W.shape
        dout2 = dout2.transpose(0,2,3,1).reshape(-1, FN)

        if self.bias:
            self.db = np.sum(dout2, axis=0)
        self.dW = np.dot(self.x_col.T, dout2)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dconv = np.dot(dout2, self.W_col.T)
        dx = col2im(dconv, self.x.shape, FH, FW, self.conv_stride, self.conv_pad)

        # update
        if self.optimizer == 'SGD':
            self.W = self.W - self.lr * self.dW
            if self.bias:
                self.b = self.b - self.lr * self.db
        if self.optimizer == 'Adam':
            if self.m is None:
                self.m = np.zeros_like(self.dW)
                self.v = np.zeros_like(self.dW)
            self.iter += 1
            lr_t  = self.lr * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
            self.m += (1 - beta1) * (self.dW - self.m)
            self.v += (1 - beta2) * (self.dW**2 - self.v)
            self.W = self.W - lr_t * self.m / (np.sqrt(self.v) + 1e-7)

            if self.bias:
                if self.bm is None:
                    self.bm = np.zeros_like(self.db)
                    self.bv = np.zeros_like(self.db)
                self.bm += (1 - beta1) * (self.db - self.bm)
                self.bv += (1 - beta2) * (self.db**2 - self.bv)
                self.b = self.b - lr_t * self.bm / (np.sqrt(self.bv) + 1e-7)

        if self.batchnorm:
            dx = self.BN.backward(dx)

        return dx


    def return_W(self):
        return self.W

    def return_b(self):
        return self.b

    def set_W(self, W):
        self.W = W

    def set_b(self,b):
        self.b = b

class Mlpconvpool:
    def __init__(self, W, mlpsize=2, init='He',lr=0.01, activation='Relu',
                    optimizer = 'SGD', pool_h=2, pool_w=2,
                    conv_stride=1, conv_pad=0, pool_stride=2, pool_pad=0):
        self.W = W
        self.W_col = None
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.pool_stride = pool_stride
        self.pool_pad = pool_pad

        self.activation = activation
        self.mask=None

        self.x = None       # 入力
        self.x_col = None   # im2colしたx
        self.u = None       # Convolutionを取った後
        self.u_col = None   # im2colしたu
        self.conv_y = None  # self.uを活性化関数に通した後
        self.conv_y_col = None # im2colしたself.conv_y
        self.pool_x_col = None # Poolingの入力(=conv_y)を（Pooling用の形状に）im2colしたもの
        self.pool_y = None  # self.conv_yをPoolingした後
        self.arg_max = None

        self.lr = lr
        self.optimizer = optimizer
        self.dW = None
        self.iter = 0
        self.m = None
        self.v = None




class Conv:
    def __init__(self, W, conv_stride=1, conv_pad=0):
        self.W = W
        self.W_col = None   # im2colしたW
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad

        self.x = None       # 入力
        self.x_col = None   # im2colしたx
        self.u = None       # Convolutionを取った後
        self.u_col = None   # im2colしたu
        self.conv_y = None  # self.uを活性化関数に通した後
        self.conv_y_col = None # im2colしたself.conv_y


    def forward(self, x):
        # Convolution
        filternum, channel, filter_h, filter_w = self.W.shape
        batchsize, channel, height, width = x.shape
        conv_out_h = 1 + int((height + 2*self.conv_pad - filter_h) / self.conv_stride)
        conv_out_w = 1 + int((width + 2*self.conv_pad - filter_w) / self.conv_stride)

        self.x = x
        self.x_col = im2col(x, filter_h, filter_w, self.conv_stride, self.conv_pad)
        self.W_col = self.W.reshape(filternum, -1).T

        self.u_col = np.dot(self.x_col, self.W_col)
        self.u = self.u_col.reshape(batchsize, conv_out_h, conv_out_w, -1).transpose(0, 3, 1, 2)

        out = self.u

        return out

class OutputLayer:
    def __init__(self):
        self.x = None
        self.original_x_shape = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        if len(self.original_x_shape) == 4:
            x = x.T
        self.x = x

        return self.x

    def backward(self, t): # この層の教師出力tから前層の教師出力t_0を計算
        if len(self.original_x_shape) == 4:
            t_0 = t.T
            t_0 = t_0.reshape(*self.original_x_shape)

        return t_0
