# import chainer.functions as F
import os


def check_and_make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# def accuracy(model, data):
