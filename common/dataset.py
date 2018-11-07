# import os

import numpy as np
import chainer


def normalize_image(tuple_dataset):
    image, label = zip(*tuple_dataset)
    image = np.array(image) * 2 - 1
    dataset = list(zip(image, label))

    return dataset


def load_dataset(dataset_name, batchsize, label_size=None, train_size=0.9, need_valid=False, label_num_dict={"cifar10": 4000, "mnist": 1000}):
    assert isinstance(batchsize, dict)
    if label_size is None:
        label_size = label_num_dict[dataset_name]

    if dataset_name == "mnist":
        d_train, d_test = chainer.datasets.get_mnist(ndim=3, scale=1.0)
    else:
        d_train, d_test = chainer.datasets.get_cifar10(ndim=3, scale=1.0)
    d_test = normalize_image(d_test)
    d_test = chainer.iterators.SerialIterator(
        d_test, batch_size=batchsize["test"], repeat=False)
    _dataset = {"test": d_test}
    if need_valid:
        train_size = int(len(d_train) * train_size)
        d_train, d_valid = chainer.datasets.split_dataset_random(
            d_train, train_size, seed=0)
        d_valid = normalize_image(d_valid)
        d_valid = chainer.iterators.SerialIterator(
            d_valid, batch_size=batchsize["test"], repeat=False, shuffle=False)
        _dataset["valid"] = d_valid

    train_image, train_label = zip(*d_train)
    train_image = np.array(train_image) * 2 - 1
    train_label = np.array(train_label)
    class_label_num = len(set(train_label))
    label_size_per_class = label_size // class_label_num
    print("label size:", label_size)
    print("label size per class:", label_size_per_class)
    d_train_labeled = []
    d_train_unlabeled = []
    label_list = []
    for class_label in range(class_label_num):
        label_idx, = np.where(train_label == class_label)
        use_label_idx = np.random.choice(
            label_idx, size=label_size_per_class, replace=False)
        assert len(use_label_idx) == len(set(use_label_idx)), print(
            len(use_label_idx), len(set(use_label_idx)))
        unlabel_idx = set(label_idx) - set(use_label_idx)
        unlabel_idx = np.array(list(unlabel_idx)).astype("i")
        d_train_unlabeled.append(train_image[unlabel_idx])
        d_train_labeled.append(train_image[use_label_idx])
        label_list.append(train_label[use_label_idx])
    d_train_labeled = np.concatenate(d_train_labeled).astype("f")
    d_train_unlabeled = np.concatenate(d_train_unlabeled).astype("f")
    print("load {} with label. shape: ".format(
        dataset_name), d_train_labeled.shape)
    print("load {} without label. shape: ".format(
        dataset_name), d_train_unlabeled.shape)

    label_list = np.concatenate(label_list)
    d_train_labeled = list(zip(d_train_labeled, label_list))
    np.random.shuffle(d_train_unlabeled)
    np.random.shuffle(d_train_labeled)

    d_train_labeled = chainer.iterators.SerialIterator(
        d_train_labeled, batch_size=batchsize["labeled"])
    d_train_unlabeled = chainer.iterators.SerialIterator(
        d_train_unlabeled, batch_size=batchsize["unlabeled"])
    _dataset["train"] = {"labeled": d_train_labeled,
                         "unlabeled": d_train_unlabeled}
    return _dataset
