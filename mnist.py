#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : mnist.py
# @Author: WangYe
# @Date  : 2019/1/11
# @Software: PyCharm
from __future__ import print_function
import os
from PIL import Image

#from paddle.v2.plot import Ploter
import numpy
import paddle
import paddle.fluid as fluid

def net(x,y):
    hidden = fluid.layers.fc(input=x, size=200, act='relu')
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    cost = fluid.layers.square_error_cost(input=prediction, label=y)
    avg_cost = fluid.layers.mean(cost)
    return prediction, avg_cost

def train():
    x = fluid.layer.data(name="input", shape=[None,784],dtype='f')
    y = fluid.layers.data(name='label', shape=[None,10], dtype='int64')
    y_predict, avg_cost = net(x, y)
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)
    BATCH_SIZE = 64
    train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=BATCH_SIZE)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program):
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe.run(fluid.default_startup_program())

        PASS_NUM = 1000
        for pass_id in range(PASS_NUM):
            total_loss_pass = 0
            for data in train_reader():
                avg_loss_value, = exe.run(
                    main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                total_loss_pass += avg_loss_value
            print("Pass %d, total avg cost = %f" % (pass_id, total_loss_pass))
    train_loop(fluid.default_main_program())

# Run train and infer.
if __name__ == "__main__":
    train()

