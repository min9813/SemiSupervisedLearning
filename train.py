#!usr/bin/python
# -*- coding: UTF-8 -*-

import chainer
import argparse
import os
import time
import common.net
import common.dataset
from common import evaluator
from chainer import training
from chainer.training import extensions


DATASET_LIST = ["mnist", "cifar10"]
ARCHITECTURE_LIST = ["small", "large"]


def main():
    start = time.time()
    gpu_id = 0
    out_folder = os.path.join("./result", args.method,
                              "label_size:{}".format(args.label_size))
    if os.path.isdir(out_folder):
        new_folder = "folder_{}".format(len(os.listdir(out_folder)))
    else:
        new_folder = "folder_0"
    out_folder = os.path.join(out_folder, new_folder)
    if args.no_debug:
        max_time = (args.max_iter, "iteration")
    else:
        max_time = (1000, "iteration")
    batchsize = {"labeled": args.label_batchsize,
                 "unlabeled": args.unlabel_batchsize,
                 "test": args.test_batchsize}
    if args.label_size != 1000:
        dataset_all = common.dataset.load_dataset(
            args.dataset, batchsize, label_size=args.label_size)
    else:
        dataset_all = common.dataset.load_dataset(
            args.dataset, batchsize)
    updater_args = {"iterator": dataset_all["train"],
                    "method": args.method}
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
        updater_args["models"] = model
        updater_args["optimizer"] = optimizer

        plot_report = ["val/acc", "train/acc"]
        print_report = ["loss/label", "loss/lsd", "val/loss"] + plot_report
    elif args.method == "base":
        from vat.updater import BaseUpdater as Updater
        if args.architecture == "large":
            model = common.net.LargeCNN(10)
        elif args.architecture == "small":
            model = common.net.SmallCNN(10)
        else:
            raise NotImplementedError
        model.to_gpu(gpu_id)
        optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9)
        optimizer.setup(model)
        updater_args["models"] = model
        updater_args["optimizer"] = optimizer
        updater_args["iterator"] = dataset_all["train"]["labeled"]

        plot_report = ["val/acc", "train/acc"]
        print_report = ["train/loss", "val/loss"] + plot_report
    else:
        raise NotImplementedError

    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, stop_trigger=max_time, out=out_folder)

    # epoch_interval = (1, 'epoch')
    save_snapshot_interval = (10000, "iteration")
    if args.method == "base" and args.architecture == "small":
        display_interval = (500, 'iteration')
    else:
        display_interval = (100, 'iteration')
    plot_interval = (500, "iteration")

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

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
            root_name="loss/label", out_name="cg_smloss.dot"))
    trainer.extend(evaluator.Evaluator(
        dataset_all["test"], model, device=gpu_id), name='val', trigger=display_interval)
    trainer.extend(extensions.PlotReport(
        plot_report, x_key='iteration', file_name='accuracy.png', trigger=plot_interval))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'elapsed_time'] + print_report), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    print("finish setup training environment in {} s".format(time.time() - start))
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
                    default=256)
parser.add_argument("--label_batchsize",
                    help="batchsize used in train labeled data",
                    type=int,
                    default=64)
parser.add_argument("--unlabel_batchsize",
                    help="batchsize used in train unlabeled data",
                    type=int,
                    default=256)
parser.add_argument("--label_size",
                    help="labeled data size",
                    type=int,
                    default=1000)
parser.add_argument("-ndb", "--no_debug",
                    help="flag if not debug, default is False", action="store_true")
parser.add_argument("--max_iter",
                    help="max iteration", type=int, default=1000)
parser.add_argument(
    "--snapshot", help="falg to save snapshot", action="store_true")
parser.add_argument(
    "--architecture", help="use what network structure", default="small", choices=ARCHITECTURE_LIST)

args = parser.parse_args()


if __name__ == "__main__":
    main()
