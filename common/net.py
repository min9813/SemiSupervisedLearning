import chainer
import chainer.functions as F
import chainer.links as L


class LargeCNN(chainer.Chain):

    def __init__(self, n_output, dropout_rate=0.5, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        self.dropout_rate = dropout_rate
        super(LargeCNN, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, initialW=w)
            self.c2 = L.Convolution2D(128, 128, ksize=3, stride=1, pad=1, initialW=w)
            self.c4 = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, initialW=w)
            self.c3 = L.Convolution2D(128, 128, ksize=3, stride=1, pad=1, initialW=w)
            self.c5 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, initialW=w)
            self.c6 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, initialW=w)
            self.c7 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=0, initialW=w)
            self.c8 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, initialW=w)
            self.c9 = L.Convolution2D(256, 128, ksize=1, stride=1, pad=0, initialW=w)
            self.l_cl = L.Linear(128, n_output)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(128)
            self.bn4 = L.BatchNormalization(256)
            self.bn5 = L.BatchNormalization(256)
            self.bn6 = L.BatchNormalization(256)
            self.bn7 = L.BatchNormalization(512)
            self.bn8 = L.BatchNormalization(256)
            self.bn9 = L.BatchNormalization(128)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.c1(x)), slope=0.1)
        h = F.leaky_relu(self.bn2(self.c2(h)), slope=0.1)
        h = F.leaky_relu(self.bn3(self.c3(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, ratio=self.dropout_rate)
        h = F.leaky_relu(self.bn4(self.c4(h)), slope=0.1)
        h = F.leaky_relu(self.bn5(self.c5(h)), slope=0.1)
        h = F.leaky_relu(self.bn6(self.c6(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, ratio=self.dropout_rate)
        h = F.leaky_relu(self.bn7(self.c7(h)), slope=0.1)
        h = F.leaky_relu(self.bn8(self.c8(h)), slope=0.1)
        h = F.leaky_relu(self.bn9(self.c9(h)), slope=0.1)
        h = F.average_pooling_2d(h, h.data.shape[2])
        # print(h.shape)
        h = self.l_cl(h)
        return h


class SmallCNN(chainer.Chain):

    def __init__(self, n_output, dropout_rate=0.5, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        super(SmallCNN, self).__init__()
        self.dropout_rate = dropout_rate
        with self.init_scope():
            self.c1 = L.Convolution2D(None, 128, ksize=4, stride=2, pad=1, initialW=w)
            self.c2 = L.Convolution2D(128, 256, ksize=4, stride=2, pad=1, initialW=w)

            self.l_cl = L.Linear(256*2*2, n_output)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.c1(x)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=1)
        h = F.dropout(h, self.dropout_rate)
        h = F.leaky_relu(self.bn2(self.c2(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, self.dropout_rate)
        h = self.l_cl(h)

        return h
