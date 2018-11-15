# import chainer.functions as F
import os

import chainer
import numpy as np
# import sys
from chainer.backends import cuda
# from chainer import Variable
from PIL import Image


def check_and_make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def out_generated_image(gen, dst, fixed_noise, seed=0, rows=10, cols=10):
    @chainer.training.make_extension()
    def make_image(trainer):

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(fixed_noise)
        x = cuda.to_cpu(x.data)
        # np.random.seed()

        def save_figure(x, file_name="image"):
            file_name += "_iteration:{:0>6}.png".format(
                trainer.updater.iteration)
            preview_path = os.path.join(dst, file_name)
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(1, 3, 0, 4, 2)
            x = x.reshape((rows * H, cols * W, C))
            if x.shape[2] == 1:
                Image.fromarray(x[:, :, 0]).save(preview_path)
            else:
                Image.fromarray(x).save(preview_path)
            # gen output_activation_func is tanh (-1 ~ 1)
        x_ = np.asarray((x * 0.5 + 0.5) * 255.0, dtype=np.uint8)
        save_figure(x_, file_name="image")

    return make_image


class VATRule(object):
    name = 'VATRule'

    def __init__(self, start_decay_epoch, whole_epoch, lr=0.001, beta1=0.9, beta2=0.5):
        self.alpha_plan = [lr] * whole_epoch
        self.beta1_plan = [beta1] * start_decay_epoch + \
            [beta2] * (whole_epoch - start_decay_epoch)
        for epoch in range(start_decay_epoch, whole_epoch):
            self.alpha_plan[epoch] = float(
                whole_epoch - epoch) / (whole_epoch - start_decay_epoch) * lr
        print(self.alpha_plan)
        print(self.beta1_plan)

    def __call__(self, opt):
        opt.hyperparam.alpha = self.alpha_plan[opt.epoch]
        opt.hyperparam.beta1 = self.beta1_plan[opt.epoch]


class GANRule(object):

    name = "GANRule"

    def __init__(self, whole_epoch, lr=0.0003, change_alpha_rate=400):
        self.whole_epoch = whole_epoch
        self.change_alpha_rate = change_alpha_rate

    def __call__(self, opt):
        opt.hyperparam.alpha = opt.hyperparam.alpha * \
            min(1., 3.-opt.epoch / self.change_alpha_rate)
