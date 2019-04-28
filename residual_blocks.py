from np import *
from functions import *
from util import im2col, col2im
import optimizer as op


beta1=0.9
beta2=0.999

class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None, optimizer='Adam', lr=0.001):
        self.gamma = gamma
        self.beta = beta
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

    def return_params(self):
        return {'gamma':self.gamma, 'beta':self.beta}

    def set_params(self, params):
        self.gamma = params['gamma']
        self.beta = params['beta']


class AffineRelu:
    def __init__(self, W, bias=False, b=None, lr=0.001, optimizer='Adam', batchnorm=True, dropout=False, maxnorm = False):
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

    def return_BNparams(self):
        if self.batchnorm:
            return self.BN.return_params
        else:
            return None

    def set_BNparams(self, bnparams):
        self.batchnorm=True
        self.BN.set_params(bnparams)
        self.BN_firstflg = False


class AffineSoftmaxCE:
    def __init__(self, W, bias=False, b=None, lr=0.001, optimizer = 'Adam', batchnorm=False, dropout=False):
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

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.BN = None
            self.BN_firstflg = True

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

        if self.batchnorm:
            if self.BN_firstflg:
                gamma = np.ones((x.shape[1]))
                beta = np.zeros((x.shape[1]))
                self.BN = BatchNormalization(gamma, beta, optimizer=self.optimizer)
                self.BN.forward(x, train_flg)
                self.BN_firstflg = False
            else:
                self.BN.forward(x, train_flg)

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

    def return_BNparams(self):
        if self.batchnorm:
            return self.BN.return_params
        else:
            return None

    def set_BNparams(self, bnparams):
        self.batchnorm=True
        self.BN.set_params(bnparams)
        self.BN_firstflg = False



class ConvPool:
    def __init__(self, W, bias=False, b=None, lr=0.001, activation='Relu', optimizer = 'Adam',
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
        #self.W_col = col_W

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

        col_WT = self.W.reshape(FN, -1)

        dconv = np.dot(dout2, col_WT)
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

    def return_BNparams(self):
        if self.batchnorm:
            return self.BN.return_params
        else:
            return None

    def set_BNparams(self, bnparams):
        self.batchnorm=True
        self.BN.set_params(bnparams)
        self.BN_firstflg = False



class Residual_Block:
    def __init__(self, x_shape, filters=32, filter_size=(3,3), batchnorm=True): #x_shape = (C,H,W)
        self.filters = filters
        self.batchnorm = batchnorm
        W = 1 /filter_size[0] \
                    * np.random.randn(filters, x_shape[0], filter_size[0], filter_size[1])
        b = np.zeros(filters)

        channels = filters
        out_h = conv_output_size(x_shape[1], filter_size[0], 1, 0)
        out_w = conv_output_size(x_shape[2], filter_size[1], 1, 0)
        pad = int((filter_size[0] - 1)/2)

        self.conv1 = ConvPool(W, bias=True, b=b, lr=0.001, conv_pad=pad,
                            pool_or_not=False, batchnorm=batchnorm)

        W = 1 /filter_size[0] \
                    * np.random.randn(filters, channels, filter_size[0], filter_size[1])
        b = np.zeros(filters)
        out_h = conv_output_size(out_h, filter_size[0], 1, pad)
        out_w = conv_output_size(out_w, filter_size[1], 1, pad)

        self.conv2 = ConvPool(W, bias=True, b=b, lr=0.001, conv_pad=pad,
                            pool_or_not=False, batchnorm=batchnorm)

    def forward(self, x, train_flg=False):
        y = self.conv1.forward(x, train_flg)
        y = self.conv2.forward(y, train_flg)

        if x.shape[1] < self.filters:
            shortcut = np.zeros((x.shape[0], self.filters, x.shape[2], x.shape[3]))
            shortcut[:, :x.shape[1]] = x
        else:
            shortcut = x
        y = y + shortcut
        return y

    def backward(self, dout):
        dx = self.conv2.backward(dout)
        dx = self.conv1.backward(dx)
        dx = dout + dx
        return dx

    def return_W(self):
        W = []
        W.append(self.conv1.return_W())
        W.append(self.conv2.return_W())
        return W

    def return_b(self):
        b = []
        b.append(self.conv1.return_b())
        b.append(self.conv2.return_b())
        return b

    def set_W(self, W):
        self.conv1.set_W(W[0])
        self.conv2.set_W(W[1])

    def set_b(self, b):
        self.conv1.set_b(b[0])
        self.conv2.set_b(b[1])

    def return_BNparams(self):
        if self.batchnorm:
            bnp = []
            bnp.append(self.conv1.return_BNparams())
            bnp.append(self.conv2.return_BNparams())
            return bnp
        else:
            return None

    def set_BNparams(self, bnparams):
        self.batchnorm=True
        self.conv1.set_BNparams(bnparams[0])
        self.conv2.set_BNparams(bnparams[1])
