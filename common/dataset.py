# import os

import numpy as np
import chainer
import os
import pickle


LABEL_NUM = {"cifar10": 4000, "mnist": 1000}
SAVE_DIR = "./common/data"


def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = np.linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten


def normalize_image(tuple_dataset, data_dir, zca):
    image, label = zip(*tuple_dataset)
    image = np.array(image)
    if zca:
        image_shape = image.shape
        component, mean, image = ZCA(image.reshape(image.shape[0], -1))
        image = image.reshape(image_shape)
        print("ZCA processing Done! save data as pickle file:{}".format(data_dir))
    else:
        image = image * 2 - 1
        print("Normalize data to {} ~ {}, save data as pickle file:{}".format(
            np.max(image), np.max(image), data_dir))
    dataset = list(zip(image, label))
    with open(data_dir, "wb") as pkl:
        pickle.dump(dataset, pkl)

    return dataset


def load_dataset(dataset_name, zca=True, label_size=None, train_size=0.9, need_valid=False, label_num_dict=LABEL_NUM):
    if label_size is None:
        label_size = label_num_dict[dataset_name]

    data_dir = os.path.join(SAVE_DIR, dataset_name)
    if zca:
        print("use ZCA processed data to learn.")
        test_data_dir = os.path.join(data_dir, "test_zca.dump")
        valid_data_dir = os.path.join(data_dir, "valid_zca.dump")
        train_data_dir = os.path.join(data_dir, "train_zca.dump")
    else:
        print("use normalizing processed data to learn.")

        test_data_dir = os.path.join(data_dir, "test_norm.dump")
        valid_data_dir = os.path.join(data_dir, "valid_norm.dump")
        train_data_dir = os.path.join(data_dir, "train_norm.dump")
    _dataset = dict()
    try:
        with open(test_data_dir, "rb") as pkl:
            d_test = pickle.load(pkl)
        with open(train_data_dir, "rb") as pkl:
            d_train = pickle.load(pkl)
        print("load pickle file")
    except FileNotFoundError:
        if dataset_name == "mnist":
            d_train, d_test = chainer.datasets.get_mnist(ndim=3, scale=1.0)
        else:
            d_train, d_test = chainer.datasets.get_cifar10(ndim=3, scale=1.0)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        d_test = normalize_image(d_test, test_data_dir, zca)
        d_train = normalize_image(d_train, train_data_dir, zca)

    if need_valid:
        try:
            with open(valid_data_dir, "rb") as pkl:
                d_valid = pickle.load(pkl)
            _dataset["valid"] = d_valid
        except FileNotFoundError:
            train_size = int(len(d_train) * train_size)
            d_train, d_valid = chainer.datasets.split_dataset_random(
                d_train, train_size, seed=0)
            d_valid = normalize_image(d_valid, valid_data_dir, zca)
            _dataset["valid"] = d_valid

    _dataset["test"] = d_test

    # make label and unlabel data
    train_image, train_label = map(np.array, zip(*d_train))
    class_label_num = len(set(train_label))
    label_size_per_class = label_size // class_label_num
    print("label size:", label_size)
    print("label size per class:", label_size_per_class)
    d_train_labeled = []
    label_list = []
    for class_label in range(class_label_num):
        label_idx, = np.where(train_label == class_label)
        label_idx = np.random.choice(
            label_idx, size=label_size_per_class, replace=False)
        d_train_labeled.append(train_image[label_idx])
        label_list.append(train_label[label_idx])
    d_train_labeled = np.concatenate(d_train_labeled).astype("f")
    print("load {} with label. shape: ".format(
        dataset_name), d_train_labeled.shape)
    print("load {} without label. shape: ".format(
        dataset_name), train_image.shape)

    label_list = np.concatenate(label_list)
    d_train_labeled = list(zip(d_train_labeled, label_list))
    np.random.shuffle(d_train_labeled)

    _dataset["train"] = {"labeled": d_train_labeled,
                         "unlabeled": train_image}
    return _dataset
