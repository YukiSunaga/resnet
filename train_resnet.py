# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from np import *
import matplotlib.pyplot as plt
from res_network import ResidualNetwork
from trainer import Trainer
import pickle
import datetime
from fashion_mnist import load_fashion_mnist
from util import to_cpu, to_gpu
from config import GPU
from memory_free import memfree


start = datetime.datetime.today()


data_file = "fashion_mnist"

(x_train, t_train), (x_test, t_test) = load_fashion_mnist(flatten=False)
x_train_shape = x_train.shape
if GPU:
    x_train = to_gpu(x_train)
    t_train = to_gpu(t_train)
    x_test = to_gpu(x_test)
    t_test = to_gpu(t_test)

epoch = 20
mini_batch = 32
n_res_blocks=3
filters=32
batchnorm = True
params_load = False
params_file = "result/201904140553/params.pkl"



network = ResidualNetwork(input_dim=x_train_shape[1:], filters=filters,
                n_res_blocks=n_res_blocks, batchnorm=batchnorm)
win_ch, win_row, win_col = network.input_dim

if params_load:
    network.load_params(file_name=params_file)
#network.last_layer.set_b( np.array([-5., 5.]) )

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=epoch, mini_batch_size=mini_batch,
                      evaluate_sample_num_per_epoch=200,
                      set_lr=False, layer_visualize=False, fmap=False, savememory=True)


del x_train, t_train, x_test, t_test

today = datetime.datetime.today()
file_name = "log_" + today.strftime("%Y%m%d%H%M")
path = "result/" + today.strftime("%Y%m%d%H%M") + "/"
os.makedirs(path)

trainer.train(file_name, path)

learning_end = datetime.datetime.today()

with open(path + "h_params" + file_name + ".txt", "a") as f:
    f.write('model : ResidualNetwork \n')
    f.write("data : " + data_file +'\n')
    f.write("x_shape = " + str(x_train_shape) +'\n')
    if params_load:
        f.write("loaded params : " + params_file + '\n')
    f.write("epochs = " + str(epoch) +'\n')
    f.write("mini batch size = " + str(mini_batch) +'\n')

    f.write("n_res_blocks : " + str(n_res_blocks) + '\n')
    f.write("filters per a conv-layer : " + str(filters) + '\n')
    f.write("batchnorm : " + str(batchnorm) + '\n')

    f.write("class size = " + str(network.class_size) + '\n')
    f.write("teacher : \n")
    f.write(str(network.teacher_label()) + '\n')


# パラメータの保存
network.save_params(path + "params.pkl")
print("Saved Network Parameters!")

classify_end = datetime.datetime.today()
time1 = learning_end - start
time2 = classify_end - start
time1_minute = int(time1.days * 24 * 60 + time1.seconds / 60)
time2_minute = int(time2.days * 24 * 60 + time2.seconds / 60)
with open(path + "h_params" + file_name + ".txt", "a") as f:
    f.write("learning timedelta : " + str(time1_minute) + "min \n")
    f.write("start to finish timedelta : " + str(time2_minute) + "min \n")

'''
if GPU:
    memfree()
'''
