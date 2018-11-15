import chainer
import os
import numpy as np
# import sys
from chainer.backends import cuda
# from chainer import Variable
from PIL import Image


def out_generated_image(gen, rows, cols, dst, fixed_noise, seed=0, from_gaussian=False):
    @chainer.training.make_extension()
    def make_image(trainer):

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(fixed_noise)
        x = cuda.to_cpu(x.data)
        # np.random.seed()

        def save_figure(x, file_name="image"):
            file_name += "_iteration:{:0>6}.png".format(trainer.updater.iteration)
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
