### This repository implemented semi-supervised learning

#### Overview
- implemented by chainer
- [VAT](https://arxiv.org/pdf/1704.03976) and [GAN](https://arxiv.org/abs/1606.03498) are implemented

#### Architecture
- VAT
  - SmallCNN
    - Two convolutional layer
  - LargeCNN
    - Nine convolutional layer
    - This is used in [original implementation](https://github.com/takerum/vat_chainer)
    - However this take much more time to training than SmallCNN
  - If you want to just check result, I recommend you to use SmallCNN, which spend less time to training

- GAN
  - Linear
    - Discriminator:4 layers of full connected layer with gaussian noise in each hidden layer.
    - Generator:3 layers of full connected layer.

  - CNN
    - Both generator and discriminator use 3 layers of convlutional layer.

#### Result
- VAT
  - 20000 iteration on MNIST with 100 labeled data and 59900 unlabeled data.
    - valid on 10000 test data, and the accuracy fluctuated around 97%.
    ![MNIST 20000 iteration with VAT and SmallCNN](https://github.com/min9813/SemiSupervisedLearning/blob/master/result_sample/mnist_vat_accuracy.png)

    - Base model(without any technique.) on Mnist with 20000 iteration and SmallCNN.
    - The accuracy fluctuated around 87%.
    ![MNIST 20000 iteration without any method(base) and SmallCNN](https://github.com/min9813/SemiSupervisedLearning/blob/master/result_sample/mnist_base_accuracy.png)

  - 500 epoch on Cifar10 with 4000 labeled data and 10000 valid data.
  - The accuracy with VAT is a bit lower than orignal paper'S one, but I'm not sure what's wrong with my code.
    ![CIFAR10 500 epoch with VAT and large cnn](https://github.com/min9813/SemiSupervisedLearning/blob/master/result_sample/cifar10_vat_accuracy.png)
    ![CIFAR10 500 epoch without any method(base) and large cnn](https://github.com/min9813/SemiSupervisedLearning/blob/master/result_sample/cifar10_base_accuracy.png)

#### Notice
- In calculating LSD loss, learning performed better when I use LSD both in labeled and unlabeled data. Otherwise, learning maybe failed.
- Original Paper implemented linear change of the optimizer's hyper parametor.
- In training GAN, using linear architecture makes learning process success.
