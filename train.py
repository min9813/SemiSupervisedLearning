#!usr/bin/python
# -*- coding: UTF-8 -*-

import chainer
import argparse
import os
import time
import common.net
import common.dataset
import common.utils
import numpy as np
from common import evaluator
from chainer import training
from chainer.training import extensions


DATASET_LIST = ["mnist", "cifar10"]
ARCHITECTURE_LIST = ["small", "large", "linear"]
EPSILON_PARAMS = {"mnist": 2.0, "cifar10": 10.0}
LABEL_NUM = {"cifar10": 4000, "mnist": 1000}


def make_parametor_plan(start_decay_epoch=460, whole_epoch=500, lr=0.001, mom1=0.9, mom2=0.5):
    lr_plan = [lr] * whole_epoch
    beta1_plan = [mom1] * start_decay_epoch + \
        [mom2] * (whole_epoch - start_decay_epoch)
    for epoch in range(start_decay_epoch, whole_epoch):
        lr_plan[epoch] = lr * (whole_epoch - epoch) / \
            (whole_epoch - start_decay_epoch)

    return {"lr": lr_plan, "beta1": beta1_plan}


def main():
    start = time.time()
    gpu_id = 0
    out_folder = os.path.join("./result", args.method, args.dataset, args.architecture,
                              "label_size:{}".format(args.label_size))
    if os.path.isdir(out_folder):
        new_folder = "folder_{}".format(len(os.listdir(out_folder)))
    else:
        new_folder = "folder_0"
    out_folder = os.path.join(out_folder, new_folder)
    if args.no_debug:
        max_time = (args.max_epoch, "epoch")
    else:
        max_time = (3, "epoch")
    batchsize = {"labeled": args.label_batchsize,
                 "unlabeled": args.unlabel_batchsize,
                 "test": args.test_batchsize}
    if args.label_size != LABEL_NUM[args.dataset]:
        dataset_all = common.dataset.load_dataset(
            args.dataset, label_size=args.label_size, zca=args.zca)
    else:
        dataset_all = common.dataset.load_dataset(
            args.dataset, zca=args.zca)
    for dataset_key, dataset_value in dataset_all.items():
        if dataset_key == "train":
            for train_type, train_data in dataset_value.items():
                dataset_value[train_type] = chainer.iterators.SerialIterator(
                    train_data, batch_size=batchsize[train_type])
            dataset_all[dataset_key] = dataset_value
        else:
            np.random.shuffle(dataset_value)
            valid_data = dataset_value[:args.valid_length]
            print("use valid data size:", len(valid_data))
            dataset_all[dataset_key] = chainer.iterators.SerialIterator(
                valid_data, batch_size=batchsize[dataset_key], repeat=False)
    updater_args = {"iterator": dataset_all["train"],
                    "method": args.method}
    acc_report = ["train/acc", "val/acc"]
    loss_report = ["train/loss", "val/loss"]
    if args.method == "vat":
        from vat.updater import Updater
        if args.architecture == "large":
            model = common.net.LargeCNN(10)
        elif args.architecture == "small":
            model = common.net.SmallCNN(10)
        else:
            raise NotImplementedError
        model.to_gpu(gpu_id)
        optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9)
        optimizer.setup(model)
        optimizer.add_hook(common.utils.VATRule(args.start_decay_epoch, args.max_epoch))
        # params_plan = make_parametor_plan(whole_epoch=args.max_epoch + 1)

        updater_args["epsilon"] = EPSILON_PARAMS[args.dataset]
        # updater_args["params_plan"] = params_plan

        print_report = ["loss/classify", "loss/lsd", "val/loss"] + acc_report
    elif args.method == "gan":
        from gan.updater import Updater
        if args.dataset == "mnist":
            gen = common.net.MnistGenerator(28, args.latent_dim)
            dis = common.net.MnistDiscriminator(10)
            # dis = common.net.LinearLayer(10)
        else:
            gen = common.net.CifarGenerator(args.latent_dim)
            dis = common.net.CifarDiscriminator(10)
        gen.to_gpu()
        dis.to_gpu()
        model = {"gen": gen, "dis": dis}
        opt_gen = chainer.optimizers.Adam(alpha=0.0003, beta1=0.5)
        opt_gen.setup(gen)
        opt_gen.add_hook(common.utils.GANRule(args.max_epoch), timing="post")
        opt_dis = chainer.optimizers.Adam(alpha=0.0003, beta1=0.5)
        opt_dis.setup(dis)
        opt_dis.add_hook(common.utils.GANRule(args.max_epoch), timing="post")

        fixed_noise = gen.make_hidden(10**2)
        optimizer = {"gen": opt_gen, "dis": opt_dis}

        gan_report = ["dis/loss", "gen/loss"]
        print_report = loss_report + acc_report + gan_report

    elif args.method == "base":
        from vat.updater import BaseUpdater as Updater
        if args.architecture == "large":
            model = common.net.LargeCNN(10)
        elif args.architecture == "small":
            model = common.net.SmallCNN(10)
        elif args.architecture == "linear":
            model = common.net.LinearClassifier(10)
        else:
            raise NotImplementedError
        model.to_gpu(gpu_id)
        optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9)
        optimizer.setup(model)
        # optimizer.add_hook(timing='post')
        updater_args["iterator"] = dataset_all["train"]["labeled"]

        print_report = loss_report + acc_report
    else:
        raise NotImplementedError
    updater_args["models"] = model
    updater_args["optimizer"] = optimizer
    updater = Updater(**updater_args)

    # epoch_interval = (1, 'epoch')
    save_snapshot_interval = (10000, "iteration")
    if args.method == "base":
        display_interval = (500, 'iteration')
        progress_interval = 100
        # max_time = (50000, "iteration")
    else:
        display_interval = (100, 'iteration')
        progress_interval = 10
    plot_interval = (200, "iteration")
    trainer = training.Trainer(updater, stop_trigger=max_time, out=out_folder)
    common.utils.check_and_make_dir(out_folder)

    # trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    if args.snapshot:
        trainer.extend(extensions.snapshot_object(
            model, args.method + '_iteration_{.updater.iteration}.npz'), trigger=save_snapshot_interval)
    trainer.extend(extensions.dump_graph(
        root_name="train/loss", out_name="cg_all.dot"))
    if args.method == "vat":
        trainer.extend(extensions.dump_graph(
            root_name="loss/lsd", out_name="cg_lsd.dot"))
        trainer.extend(extensions.dump_graph(
            root_name="loss/classify", out_name="cg_smloss.dot"))
        trainer.extend(evaluator.Evaluator(
            dataset_all["test"], model, device=gpu_id), name='val', trigger=display_interval)
    elif args.method == "gan":
        out_image_folder = os.path.join(out_folder, "preview")
        common.utils.check_and_make_dir(out_image_folder)
        trainer.extend(extensions.dump_graph(
            root_name="dis/loss", out_name="cg_dis.dot"))
        trainer.extend(extensions.dump_graph(
            root_name="gen/loss", out_name="cg_gen.dot"))
        trainer.extend(common.utils.out_generated_image(gen, out_image_folder, fixed_noise), trigger=(200, "iteration"))
        trainer.extend(extensions.PlotReport(
            gan_report, x_key='iteration', file_name='gan.png', trigger=plot_interval))
        trainer.extend(evaluator.Evaluator(
            dataset_all["test"], model["dis"], device=gpu_id), name='val', trigger=display_interval)
    elif args.method == "base":
        trainer.extend(evaluator.Evaluator(
            dataset_all["test"], model, device=gpu_id), name='val', trigger=display_interval)

    trainer.extend(extensions.PlotReport(
        acc_report, x_key='iteration', file_name='accuracy.png', trigger=plot_interval))
    trainer.extend(extensions.PlotReport(
        loss_report, x_key='iteration', file_name='loss.png', trigger=plot_interval))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'elapsed_time'] + print_report), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=progress_interval))

    print("finish setup training environment in {} s".format(time.time() - start))
    print("start training ...")
    # Run the training
    trainer.run()


parser = argparse.ArgumentParser(
    description="This file is used to train semi-supervised model")
parser.add_argument("-m", "--method",
                    help="method for semi supervised learning.",
                    default="vat")
parser.add_argument("--dataset",
                    help="dataset to train",
                    default="mnist",
                    choices=DATASET_LIST)
parser.add_argument("--test_batchsize",
                    help="batchsize used in test/valid data",
                    type=int,
                    default=100)
parser.add_argument("--label_batchsize",
                    help="batchsize used in train labeled data",
                    type=int,
                    default=32)
parser.add_argument("--unlabel_batchsize",
                    help="batchsize used in train unlabeled data",
                    type=int,
                    default=128)
parser.add_argument("--label_size",
                    help="labeled data size",
                    type=int,
                    default=1000)
parser.add_argument("--start_decay_epoch",
                    help="the epoch to start linear-change of optimizer's parametor",
                    type=int,
                    default=460)
parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--mom1",
                    help=" value of beta1 used in former training",
                    type=float,
                    default=0.9)
parser.add_argument("--mom2",
                    help=" value of beta1 used in latter training",
                    type=float,
                    default=0.5)
parser.add_argument("-ndb", "--no_debug",
                    help="flag if not debug, default is False", action="store_true")
parser.add_argument("--max_iter",
                    help="max iteration", type=int, default=1000)
parser.add_argument("--max_epoch",
                    help="max epoch", type=int, default=120)
parser.add_argument(
    "--snapshot", help="falg to save snapshot", action="store_true")
parser.add_argument(
    "--architecture", help="use what network structure", default="small", choices=ARCHITECTURE_LIST)
parser.add_argument(
    "--zca", help="flag to use zca processed data", action="store_true")
parser.add_argument(
    "--valid_length", help="flag to use zca processed data", type=int, default=1000)
parser.add_argument(
    "--latent_dim", help="dimension of latent variables in generator", type=int, default=128)

args = parser.parse_args()


if __name__ == "__main__":
    main()
