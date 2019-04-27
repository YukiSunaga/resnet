import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
from np import *
from functions import *
from util import *
from collections import OrderedDict
from residual_blocks import *


class ResidualNetwork:
    def __init__(self, input_dim=(1, 28, 28), n_res_blocks=3, filters=32, hidden_num=128,
                 conv_param = {'conv_h':3, 'conv_w':3, 'pool_size':2},
                 output_size=10, class_size=10):
        # 重みの初期化===========
        self.n_res_blocks = n_res_blocks
        self.input_dim = input_dim
        self.conv_list = [conv_param]
        self.hidden_list = [hidden_num]

        pre_channel_num = input_dim[0]

        output_layer_idx = 1    # 最終層は何番目の層か
        conv_out_h, conv_out_w = input_dim[1], input_dim[2]
        W = {}
        b = {}

        for i, c in enumerate(self.conv_list):
             W['conv' + str(i)] = np.sqrt(2.0 /filters) * np.random.randn(filters, pre_channel_num, c['conv_h'], c['conv_w'])
             b['conv' + str(i)] = np.zeros(filters)
             pre_channel_num = filters
             conv_out_h = conv_output_size(conv_out_h, c['conv_h'], 1, 0)
             conv_out_w = conv_output_size(conv_out_w, c['conv_w'], 1, 0)
             conv_out_h = conv_output_size(conv_out_h, c['pool_size'], c['pool_size'], 0)
             conv_out_w = conv_output_size(conv_out_w, c['pool_size'], c['pool_size'], 0)

        self.conv_layers = []
        for i, c in enumerate(self.conv_list):
            self.conv_layers.append(ConvPool(W=W['conv'+str(i)], bias=True, b=b['conv'+str(i)], conv_stride=1, conv_pad=0,pool_h=c['pool_size'], pool_w=c['pool_size'], pool_stride=c['pool_size'],
                                        batchnorm=True, pool='max', pool_or_not=True))

        self.res_blocks = []
        for i in range(self.n_res_blocks):
            self.res_blocks.append( Residual_Block(x_shape=(pre_channel_num, conv_out_h, conv_out_w), filters=filters) )
            pre_channel_num = filters


        c_out_size = int(pre_channel_num*conv_out_h*conv_out_w)

        for i, hn in enumerate(self.hidden_list):
            W['hidden' + str(i)] = np.sqrt(2.0 / c_out_size) * np.random.randn( c_out_size, hn )
            b['hidden' + str(i)] = np.zeros(hn)
            c_out_size = hn

        W['last_layer'] = np.sqrt(2.0 / c_out_size) * np.random.randn( c_out_size, output_size )
        b['last_layer'] = np.zeros(output_size)




        self.hidden_layers = []
        for i, hn in enumerate(self.hidden_list):
            self.hidden_layers.append(AffineRelu(W=W['hidden' + str(i)], bias=True, b=b['hidden'+str(i)], batchnorm=True))

        self.last_layer = AffineSoftmaxCE(W=W['last_layer'], bias=True, b=b['last_layer'], batchnorm=True)

        self.out_size = int(output_size)
        self.class_size = int(class_size)

        del W, b


    def predict(self, x, train_flg=False):
        for i, l in enumerate(self.conv_layers):
            x = l.forward(x, train_flg)
        for i, rb in enumerate(self.res_blocks):
            x = rb.forward(x, train_flg)
        for i, l in enumerate(self.hidden_layers):
            x = l.forward(x, train_flg)
        x = self.last_layer.forward(x,train_flg)
        return x

    def teacher_label(self): # 教師ラベルt0,t1,...,tn T=(t0 t1 ... tn)を出力
        t = np.arange(self.class_size)
        T = to_one_hot_label(t, self.out_size, self.class_size)    # 教師ラベルt0,t1,...,tn T=(t0 t1 ... tn)
        return T

    def classify(self, x, T=None, train_flg=False):  # predict()の出力がどのクラスか（どの教師ラベルに近いか）を調べる
        y = self.predict(x, train_flg)
        clss = np.argmax(y ,axis=1)

        return clss

    def set_lr(self, delta_loss):
        for layer in self.layers:
            layer.set_lr(delta_loss)
        self.last_layer.set_lr(delta_loss)

    def get_lr(self):
        lr = {}
        lr_a = self.layers[0].lr_a
        lr_p = self.layers[0].lr_p
        lr['lr_a'] = lr_a
        lr['lr_p'] = lr_p
        return lr

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        y = cross_entropy_error(y, t)
        return y

    def accuracy(self, x, t, batch_size=100, confusion_mat=False):
        y = self.predict(x, train_flg=False)
        clss = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(clss == t) / float(x.shape[0])
            #tt = np.argmax(t,axis=0)
            #print(y)
            #print(tt)
        if confusion_mat:
            mat, wave = self.confusion_matrix(self.class_size, clss, t, x)
            return acc, mat, wave
        else:
            return acc


    def confusion_matrix(self, classnum, clss, t, x=None):
        mat = np.zeros((classnum, classnum),dtype=int)
        #y = self.network.classify(x)
        for i in range(classnum):
            for j in range(classnum):
                tij = t[clss==i]
                mat[i,j] = tij[tij==j].size

        x_miss21 = x[ (t==0)&(clss==1) ]
        x_miss12 = x[ (t==1)&(clss==0) ]
        conf_wave = {}
        conf_wave['12'] = x_miss12
        conf_wave['21'] = x_miss21
        return mat, conf_wave

    def gradient(self, x, t, train_flg=True):
        loss = self.loss(x, t, train_flg)

        t_0 = self.last_layer.backward(t)

        r = self.hidden_layers.copy()
        r.reverse()
        for i, layer in enumerate(r):
            t_0 = layer.backward(t_0)

        r = self.res_blocks.copy()
        r.reverse()
        for i, layer in enumerate(r):
            t_0 = layer.backward(t_0)

        r = self.conv_layers.copy()
        r.reverse()
        for i, layer in enumerate(r):
            t_0 = layer.backward(t_0)


        return loss


    def save_params(self, file_name="params.pkl"):
        conv_W = []
        conv_b = []
        for i, l in enumerate(self.conv_layers):
            conv_W.append(l.return_W())
            conv_b.append(l.return_b())

        res_W = []
        res_b = []

        for i, l in enumerate(self.res_blocks):
            res_W.append(l.return_W())
            res_b.append(l.return_b())

        hid_W = []
        hid_b = []
        for i, l in enumerate(self.hidden_layers):
            hid_W.append(l.return_W())
            hid_b.append(l.return_b())


        last_W = self.last_layer.return_W()
        last_b = self.last_layer.return_b()


        params = {}
        params['conv_W'] = conv_W
        params['conv_b'] = conv_b
        params['res_W'] = res_W
        params['res_b'] = res_b
        params['hid_W'] = hid_W
        params['hid_b'] = hid_b
        params['last_W'] = last_W
        params['last_b'] = last_b

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        for i, l in enumerate(self.conv_layers):
            l.set_W(params['conv_W'][i])
            l.set_b(params['conv_b'][i])
        for i, l in enumerate(self.res_blocks):
            l.set_W(params['res_W'][i])
            l.set_b(params['res_b'][i])
        for i, l in enumerate(self.hidden_layers):
            l.set_W(params['hid_W'][i])
            l.set_b(params['hid_b'][i])

        self.last_layer.set_W(params['last_W'])
        self.last_layer.set_b(params['last_b'])
