### This repository implemented semi-supervised learning

#### Overview
- implemented by chainer
- Only [VAT](https://arxiv.org/pdf/1704.03976.pdf) is implemented

#### Architecture
- SmallCNN
  - Two convolutional layer
- LargeCNN
  - Nine convolutional layer
  - This is used in [original implementation](https://github.com/takerum/vat_chainer)
  - However this take much more time to training than SmallCNN
- If you want to just check result, I recommend you to use SmallCNN, which spend less time to training

#### Result
- 20000 iteration on MNIST with 100 labeled data and 59900 unlabeled data.
- valid on 10000 test data.
<!-- ![MNIST 20000 iteration with VAT and SmallCNN](https://github.com/min9813/GAN/blob/master//accuracy.png) -->

#### Notice
- In calculating LSD loss, learning performed better when I use LSD both in labeled and unlabeled data. Otherwise, learning maybe failed.

#### TODO
- Implement Gan's semi-supervised learning
