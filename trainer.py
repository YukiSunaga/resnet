# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from np import *
import matplotlib.pyplot as plt
from matplotlib import cm
from optimizer import *
import datetime
from functions import *
import numpy
from util import *



class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 evaluate_sample_num_per_epoch=None, verbose=True, set_lr=False, equal=False,
                 layer_visualize=False, fmap=True, savememory=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.set_lr = set_lr
        self.equal = equal
        self.fmap = fmap
        self.savememory = savememory


        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]
        self.iter_per_epoch = int(max(self.train_size / mini_batch_size, 1))
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.test_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        self.previous_loss = 1.0
        self.delta_loss = 0

        self.graph = True
        self.graph_steps = 100 # update graphs every self.graph_steps iterations
        if layer_visualize:
            self.graph_num = 2
        else:
            self.graph_num = 0
        self.first_flg = True
        self.x_count = 1

        if self.graph:
            if self.graph_num == 1:
                self.W_Re = np.zeros(1)
                self.W_Im = np.zeros(1)
                self.t = np.zeros(1)
                self.fig, self.ax = plt.subplots(1, 1)
                self.lines, = self.ax.plot(self.W_Re, self.W_Im,lw=0)
                self.Rmin, self.Rmax, self.Imin, self.Imax = -0.05, 0.05, -0.05, 0.05
                self.ax.set_xlim((-0.05, 0.05))
                self.ax.set_ylim((-0.05, 0.05))
                plt.scatter(self.W_Re, self.W_Im,c=self.t,cmap=cm.jet,marker='.',lw=0)
            elif self.graph_num == 2:
                #self.figures = plt.figure(self.network.conv_num * 2)
                self.first_flg = True
                self.filter_show_fig = {}
                self.filter_show_ax = {}
                self.filter_lines = {}

                for i in range(self.network.conv_num):
                    self.filter_show_fig['W'+str(i+1)], self.filter_show_ax['W'+str(i+1)] = fig_ax(self.network.layers[i].W)

                for i in range(self.network.conv_num):
                    self.filter_lines['W'+str(i+1)] = lines_w(self.network.layers[i].W, self.filter_show_ax['W'+str(i+1)], cmap=plt.cm.gray, vmin=-2.0, vmax=2.0)

            if True:
                self.graph_x = numpy.zeros(1)
                self.graph_y = numpy.zeros(1)
                self.graph_y2 = numpy.zeros(1)
                #plt.figure(self.network.conv_num*2+1)
                self.fig, self.ax = plt.subplots(1, 1)
                self.lines, = self.ax.plot(self.graph_x, self.graph_y, 'b')
                self.testlossline, = self.ax.plot(self.graph_x, self.graph_y2, 'r')
                self.ax.set_xlim((0, self.max_iter))

            '''
            self.w_epoch = {} # epochごとのカーネルを保存
            for i in range(self.network.conv_num):
                self.w_epoch['W'+str(i+1)] = []
            for i in range(self.network.conv_num):
                self.w_epoch['W'+str(i+1)].append(self.network.layers[i].W)
            '''




    def train_step(self, filename, path):
        if self.equal:
            batch_size_pc = int(self.batch_size / self.network.class_size)
            train_size_pc = int(self.train_size / self.network.class_size)
            batch_mask = np.zeros(batch_size_pc * self.network.class_size, dtype=int)
            for c in range(self.network.class_size):
                batch_mask[c*batch_size_pc:(c+1)*batch_size_pc] = np.random.choice(train_size_pc, batch_size_pc) + c*train_size_pc
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
        else:
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

        t_batch = to_one_hot_label(t_batch, self.network.out_size, self.network.class_size)
        self.network.gradient(x_batch, t_batch)

        if GPU:
            loss = to_cpu(self.network.loss(x_batch, t_batch))
        else:
            loss = self.network.loss(x_batch, t_batch)

        if self.set_lr:
            #self.delta_loss = loss - self.previous_loss
            #if self.delta_loss < 0:
            #    self.delta_loss = 0
            #self.network.set_lr(self.delta_loss)
            #self.previous_loss = loss
            if self.current_iter == 1000:
                self.network.set_lr(-0.9)
            print(self.network.get_lr())

        self.train_loss_list.append(loss)


        batch_mask = np.random.choice(self.test_size, self.batch_size)
        x_batch = self.x_test[batch_mask]
        t_batch = self.t_test[batch_mask]

        t_batch = to_one_hot_label(t_batch, self.network.out_size, self.network.class_size)
        if GPU:
            test_loss = to_cpu(self.network.loss(x_batch, t_batch))
        else:
            test_loss = self.network.loss(x_batch, t_batch)
        self.test_loss_list.append(test_loss)

        if self.verbose:
            print(str(self.current_iter) + " train loss:" + str(loss))
            with open(path + "loss" + filename + ".txt", 'a') as f:
                f.write("train loss: " + str(loss) + ' ' + 'test loss: ' + str(test_loss) +'\n')

        if self.graph:
            # 損失関数の値を刻々更新するグラフで表示
            '''
            # この方法はwindowsではフリーズするらしい
            fig = plt.gcf()
            fig.show()
            fig.canvas.draw()
            '''
            # 別の方法でグラフを表示
            if self.first_flg:
                if True:
                    self.graph_y[0] = loss
                    self.graph_y2[0] = test_loss
                    self.graph_x[0] = self.x_count
                    if self.graph_num == 2: plt.figure(self.network.conv_num+1)
                    self.lines.set_data(self.graph_x, self.graph_y)
                    self.testlossline.set_data(self.graph_x, self.graph_y2)
                    self.ax.set_ylim((0., 1.1 * self.graph_y2.max()))
                if self.graph_num == 1:
                    self.W_Re[0] = np.real(self.network.layers[0].W[0,0,0,0])
                    self.W_Im[0] = np.imag(self.network.layers[0].W[0,0,0,0])
                    plt.scatter(self.W_Re, self.W_Im,c=self.t,cmap=cm.jet,marker='.',lw=0)
                    self.lines.set_data(self.W_Re, self.W_Im)
                    if self.W_Re.min() - 0.01 < self.Rmin :
                        self.Rmin = self.W_Re.min() - 0.01
                        self.ax.set_xlim(self.Rmin, self.Rmax)
                    if self.W_Re.max() + 0.01 > self.Rmax :
                        self.Rmax = self.W_Re.max() + 0.01
                        self.ax.set_xlim(self.Rmin, self.Rmax)
                    if self.W_Im.min() - 0.01 < self.Imin :
                        self.Imin = self.W_Im.min() - 0.01
                        self.ax.set_ylim(self.Imin, self.Imax)
                    if self.W_Im.max() + 0.01 > self.Imax :
                        self.Imax = self.W_Im.max() + 0.01
                        self.ax.set_ylim(self.Imin, self.Imax)
                if self.graph_num == 2:
                    for i in range(self.network.conv_num):
                        plt.figure(i+1)
                        if self.current_iter == 1 or self.current_iter % self.iter_per_epoch == 0:
                            plt.savefig(path + "W" + str(i+1) +"_" + str(int(self.current_iter / self.iter_per_epoch)) +".png")
                        plt.pause(1e-10)


                self.first_flg = False
                self.x_count += 1

                #plt.show()
                plt.pause(1e-10) # 引数は表示してから次の処理に移行するまでの待ち時間．0にしたときの動作は環境依存なので0より大きくする

            else:
                if self.current_iter % self.graph_steps == 0 or self.current_iter % self.iter_per_epoch == 0:
                    if True:
                        self.graph_y = numpy.append(self.graph_y, loss)
                        self.graph_x = numpy.append(self.graph_x, self.x_count)
                        self.graph_y2 = numpy.append(self.graph_y2, test_loss)
                        if self.graph_num == 2: plt.figure(self.network.conv_num+1)
                        self.lines.set_data(self.graph_x, self.graph_y)
                        self.testlossline.set_data(self.graph_x, self.graph_y2)
                        self.ax.set_ylim((0., 1.1*self.graph_y2.max()))
                        plt.pause(1e-10)
                    if self.graph_num == 1:
                        self.t = np.append(self.t, self.x_count)
                        self.W_Re = np.append(self.W_Re, np.real(self.network.layers[0].W[0,0,0,0]))
                        self.W_Im = np.append(self.W_Im, np.imag(self.network.layers[0].W[0,0,0,0]))
                        plt.scatter(self.W_Re, self.W_Im, c=self.t, cmap=cm.jet, marker='.', lw=0)
                        self.lines.set_data(self.W_Re, self.W_Im)
                        if self.W_Re.min() - 0.01 < self.Rmin :
                            self.Rmin = self.W_Re.min() - 0.01
                            self.ax.set_xlim(self.Rmin, self.Rmax)
                        if self.W_Re.max() + 0.01 > self.Rmax :
                            self.Rmax = self.W_Re.max() + 0.01
                            self.ax.set_xlim(self.Rmin, self.Rmax)
                        if self.W_Im.min() - 0.01 < self.Imin :
                            self.Imin = self.W_Im.min() - 0.01
                            self.ax.set_ylim(self.Imin, self.Imax)
                        if self.W_Im.max() + 0.01 > self.Imax :
                            self.Imax = self.W_Im.max() + 0.01
                            self.ax.set_ylim(self.Imin, self.Imax)
                            plt.pause(1e-10)


                    elif self.graph_num == 2:
                        for i in range(self.network.conv_num):
                            plt.figure(i+1)
                            set_lines_w(self.network.layers[i].W, self.filter_lines['W'+str(i+1)])
                            if self.current_iter == 1 or self.current_iter % self.iter_per_epoch == 0:
                                plt.savefig(path + "W" + str(i+1) +"_" + str(int(self.current_iter / self.iter_per_epoch)) +".png")
                            plt.pause(1e-10)



                self.x_count += 1
                    #plt.show()

                if self.x_count == self.max_iter + 1:
                    if self.graph_num == 1:
                        plt.scatter(self.W_Re, self.W_Im,c=self.t,cmap=cm.jet,marker='.',lw=0)
                        ay=plt.colorbar()
                        ay.set_label('iter')
                    plt.savefig(path + filename + ".png")
                    if not self.graph_num == 0:
                        self.fig, self.ax = plt.subplots(1, 1)
                        self.lines, = self.ax.plot(self.graph_x, self.graph_y, 'b')
                        self.testlossline, = self.ax.plot(self.graph_x, self.graph_y2, 'r')
                        self.ax.set_xlim((0, self.max_iter))
                        self.ax.set_ylim((0., 1.1*self.graph_y2.max()))
                        plt.savefig(path + "loss_" + filename + ".png")
                #plt.plot(self.graph_x, self.graph_y)
                #fig.canvas.draw()

        test_acc = 0.0
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                tr_batch_mask = np.random.choice(self.train_size, self.evaluate_sample_num_per_epoch)
                te_batch_mask = np.random.choice(self.test_size, self.evaluate_sample_num_per_epoch)

                x_train_sample, t_train_sample = self.x_train[tr_batch_mask], self.t_train[tr_batch_mask]
                x_test_sample, t_test_sample = self.x_test[te_batch_mask], self.t_test[te_batch_mask]

            #print(x_train_sample.shape)
            #print(t_train_sample.shape)
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            '''
            for i in range(self.network.conv_num):
                self.w_epoch['W'+str(i+1)].append(self.network.layers[i].W)
            '''

            if self.verbose:
                print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
                with open(path + "acc" + filename + ".txt", 'a') as f:
                    f.write("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===" +'\n')



        self.current_iter += 1
        return test_acc

    def train(self, filename, path):
        '''
        teacher = self.network.teacher_label()
        print(teacher)
        sea = -np.ones(6) + 0.j
        sea[5] = 1
        sea = sea.reshape((6,1))
        print(sea)
        clss = self.network.classify(sea,teacher)
        print(clss)
        '''


        for i in range(self.max_iter):
            acc = self.train_step(filename, path)
            '''
            if acc == 1.0 :
                x = self.x_test
                t = self.t_test
                test_acc = self.network.accuracy(x, t)
                if test_acc > 0.85:
                    print(test_acc)
                    #break
            '''

        x = self.x_test
        t = self.t_test
        if self.savememory:
            del self.x_test, self.t_test
        if t.size > 5000:
            rand = np.random.randint(0,t.size,(5000,))
            x = x[rand]
            t = t[rand]

        if self.savememory:
            flg = True
            size = int(x.shape[0]/32)
            for i in range(size):
                if flg:
                    test_acc = self.network.accuracy(x[32*i:32*(i+1)], t[32*i:32*(i+1)], confusion_mat=False)
                    flg = False
                else:
                    acc = self.network.accuracy(x[32*i:32*(i+1)], t[32*i:32*(i+1)], confusion_mat=False)
                    test_acc += acc

            test_acc /= size

        else:
            test_acc= self.network.accuracy(x, t, confusion_mat=True)

        x = self.x_train
        t = self.t_train
        if self.savememory:
            del self.x_train, self.t_train
        if t.size > 10000:
            rand = np.random.randint(0,t.size,(10000,))
            x = x[rand]
            t = t[rand]
        if self.savememory:
            flg = True
            size = int(x.shape[0]/32)
            for i in range(size):
                if flg:
                    train_acc= self.network.accuracy(x[32*i:32*(i+1)], t[32*i:32*(i+1)])
                    flg = False
                else:
                    acc = self.network.accuracy(x[32*i:32*(i+1)], t[32*i:32*(i+1)])
                    train_acc += acc
            train_acc /= size
        else:
            train_acc = self.network.accuracy(x, t)

        del x, t

        '''
        for i in range(self.network.conv_num):
            self.w_epoch['W'+str(i+1)].append(self.network.layers[i].W)
        np.save(path+"W_animation.npy", self.w_epoch)
        '''

        '''
        #最終層の出力を調べる
        x0 = self.x_test[0:1]
        t0 = to_one_hot_label(self.t_test[0:1], self.network.out_size)
        x1 = self.x_test[600:601]
        t1 = to_one_hot_label(self.t_test[600:601], self.network.out_size)
        x2 = self.x_test[1200:1201]
        t2 = to_one_hot_label(self.t_test[1200:1201], self.network.out_size)
        x3 = self.x_test[1800:1801]
        t3 = to_one_hot_label(self.t_test[1800:1801], self.network.out_size)
        x4 = self.x_test[2400:2401]
        t4 = to_one_hot_label(self.t_test[2400:2401], self.network.out_size)
        x5 = self.x_test[3000:3001]
        t5 = to_one_hot_label(self.t_test[3000:3001], self.network.out_size)
        x6 = self.x_test[3600:3601]
        t6 = to_one_hot_label(self.t_test[3600:3601], self.network.out_size)


        prd0 = self.network.predict(x0,t0)
        prd1 = self.network.predict(x1,t1)
        prd2 = self.network.predict(x2,t2)
        prd3 = self.network.predict(x3,t3)
        prd4 = self.network.predict(x4,t4)
        prd5 = self.network.predict(x5,t5)
        prd6 = self.network.predict(x6,t6)

        print(prd0)
        print(prd1)
        print(prd2)
        print(prd3)
        print(prd4)
        print(prd5)
        print(prd6)
        '''
        if self.fmap and not(self.savememory):
            x_size = self.x_train.shape[0]
            fmap(self.network, self.x_train[0].reshape(( (1,) + self.x_train[0].shape)), path, 'n')
            fmap(self.network, self.x_train[int(x_size * 2 / 3)].reshape(( (1,) + self.x_train[0].shape)), path, 'ab')
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc) + " train acc:" + str(train_acc))
            with open(path + "acc" + filename + ".txt", 'a') as f:
                    f.write("=== fin_acc:" + "train acc:" + str(train_acc) + ", test acc:" + str(test_acc))
