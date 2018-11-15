import chainer
import chainer.functions as F
import chainer.links as L


class MnistGenerator(chainer.Chain):

    def __init__(self, n_hidden, bottom_width=3, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        super(MnistGenerator, self).__init__()
        with self.init_scope():
            self.z2l = L.Linear(None, 4 * 4 * 512, initialW=w)
            self.bn1 = L.BatchNormalization(512)
            self.deconv1 = L.Deconvolution2D(
                512, 256, ksize=4, stride=2, pad=1, outsize=(8, 8), initialW=w)
            self.bn2 = L.BatchNormalization(256)
            self.deconv2 = L.Deconvolution2D(
                256, 128, ksize=4, stride=2, pad=2, outsize=(14, 14), initialW=w)
            self.bn3 = L.BatchNormalization(128)
            self.deconv3 = L.Deconvolution2D(
                128, 1, ksize=4, stride=2, pad=1, outsize=(28, 28), initialW=w)

    def make_hidden(self, batchsize):
        return self.xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype("f")

    def __call__(self, x):
        h = F.reshape(F.relu(self.z2l(x)), (x.shape[0], 512, 4, 4))
        h = F.relu(self.deconv1(self.bn1(h)))
        h = F.relu(self.deconv2(self.bn2(h)))
        h = self.deconv3(self.bn3(h))
        return F.tanh(h)


class CifarGenerator(chainer.Chain):

    def __init__(self, n_hidden, wscale=0.02, ch=512, bottom_width=4):
        super(CifarGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.output_activation = F.tanh
        self.hidden_activation = F.relu
        self.ch = ch
        self.bottom_width = bottom_width
        with self.init_scope():
            w = chainer.initializers.Normal(scale=wscale)
            self.l0 = L.Linear(None, bottom_width * bottom_width * ch,
                               initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)
            # if self.use_bn:
            self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        return self.xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(self.xp.float32)
        # return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
        # .astype(np.float32)

    def __call__(self, z):
        h = F.reshape(self.hidden_activation(self.bn0(self.l0(z))),
                      (len(z), self.ch, self.bottom_width, self.bottom_width))
        h = self.hidden_activation(self.bn1(self.dc1(h)))
        h = self.hidden_activation(self.bn2(self.dc2(h)))
        h = self.hidden_activation(self.bn3(self.dc3(h)))
        x = self.output_activation(self.dc4(h))
        return x


class CifarDiscriminator(chainer.Chain):

    def __init__(self, output, wscale=0.02, ch=512):
        super(CifarDiscriminator, self).__init__()
        w = chainer.initializers.Normal(scale=wscale)
        with self.init_scope():
            # input=(32,32), output=(32, 32)
            self.conv1 = L.Convolution2D(
                3, ch//8, ksize=3, stride=1, pad=1, initialW=w)
            self.conv2 = L.Convolution2D(
                ch//8, ch//4, ksize=4, stride=2, pad=1, initialW=w)
            self.conv3 = L.Convolution2D(
                ch//4, ch//2, ksize=4, stride=2, pad=1, initialW=w)
            self.conv4 = L.Convolution2D(
                ch//2, ch, ksize=4, stride=2, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(ch//8)
            self.bn2 = L.BatchNormalization(ch//4)
            self.bn3 = L.BatchNormalization(ch//2)
            self.bn4 = L.BatchNormalization(ch)

            self.lout = L.Linear(ch*4*4, output)

            self.feature = None

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.2)
        h = self.bn5(self.conv4(h))
        self.feature = h
        h = F.reshape(h, (x.shape[0], -1))

        return self.lout(h)


class MnistDiscriminator(chainer.Chain):

    def __init__(self, output_dim=1):
        # self.in_channel = in_channel
        w = chainer.initializers.Normal(scale=0.02)
        super(MnistDiscriminator, self).__init__()
        with self.init_scope():
            # 28
            self.conv1 = L.Convolution2D(
                None, 128, ksize=4, stride=2, pad=1, initialW=w)
            # 14
            self.conv2 = L.Convolution2D(
                128, 256, ksize=4, stride=2, pad=2, initialW=w)
            # 8
            self.conv3 = L.Convolution2D(
                256, 512, ksize=4, stride=2, pad=1, initialW=w)
            # 4
            self.lout = L.Linear(512, output_dim, initialW=w)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)

            self.g1 = GaussianLayer(mean=0, std=0.3)
            self.g2 = GaussianLayer(mean=0, std=0.5)

            self.feature = None

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(self.g1(x))), slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(self.g2(h))), slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(self.g2(h))), slope=0.2)
        self.feature = F.average_pooling_2d(h, h.data.shape[2])
        h = F.reshape(self.feature, (x.shape[0], -1))

        return self.lout(h)


class LargeCNN(chainer.Chain):

    def __init__(self, n_output, dropout_rate=0.5, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        self.dropout_rate = dropout_rate
        super(LargeCNN, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(
                None, 128, ksize=3, stride=1, pad=1, initialW=w)
            self.c2 = L.Convolution2D(
                128, 128, ksize=3, stride=1, pad=1, initialW=w)
            self.c4 = L.Convolution2D(
                128, 256, ksize=3, stride=1, pad=1, initialW=w)
            self.c3 = L.Convolution2D(
                128, 128, ksize=3, stride=1, pad=1, initialW=w)
            self.c5 = L.Convolution2D(
                256, 256, ksize=3, stride=1, pad=1, initialW=w)
            self.c6 = L.Convolution2D(
                256, 256, ksize=3, stride=1, pad=1, initialW=w)
            self.c7 = L.Convolution2D(
                256, 512, ksize=3, stride=1, pad=0, initialW=w)
            self.c8 = L.Convolution2D(
                512, 256, ksize=1, stride=1, pad=0, initialW=w)
            self.c9 = L.Convolution2D(
                256, 128, ksize=1, stride=1, pad=0, initialW=w)
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
            self.c1 = L.Convolution2D(
                None, 128, ksize=4, stride=2, pad=1, initialW=w)
            self.c2 = L.Convolution2D(
                128, 256, ksize=4, stride=2, pad=1, initialW=w)

            self.l_cl = L.Linear(256 * 2 * 2, n_output)
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


class GaussianLayer(chainer.Chain):

    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std
        super(GaussianLayer, self).__init__()

    def __call__(self, x):
        mean = self.xp.ones_like(x) * self.mean
        std = self.xp.ones_like(x) * self.std

        return x + F.gaussian(mean, std)


class LinearDiscriminator(chainer.Chain):

    def __init__(self, n_output, dropout_rate=0.5, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        super(LinearDiscriminator, self).__init__()
        self.dropout_rate = dropout_rate
        with self.init_scope():
            self.l1 = L.Linear(None, 1000, initialW=w)
            self.l2 = L.Linear(1000, 500, initialW=w)
            self.l3 = L.Linear(500, 250, initialW=w)
            self.l4 = L.Linear(250, n_output, initialW=w)

            self.g1 = GaussianLayer(std=0.3)
            self.g2 = GaussianLayer(std=0.5)

    def __call__(self, x):

        h = F.reshape(x, (x.shape[0], -1))
        h = self.l1(self.g1(h))
        h = self.l2(F.softplus(self.g2(h)))
        h = self.l3(F.softplus(self.g2(h)))
        self.feature = F.softplus(self.g2(h))
        h = self.l4(self.feature)

        return h


class LinearGenerator(chainer.Chain):

    def __init__(self, image_width, n_hidden, dropout_rate=0.5, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        self.n_hidden = n_hidden
        self.image_width = image_width
        super(LinearGenerator, self).__init__()
        self.dropout_rate = dropout_rate
        with self.init_scope():
            self.l1 = L.Linear(None, 500, initialW=w)
            self.l2 = L.Linear(500, 500, initialW=w)
            self.l3 = L.Linear(500, image_width**2, initialW=w)

            self.bn1 = L.BatchNormalization(500)
            self.bn2 = L.BatchNormalization(500)

    def __call__(self, x):

        h = F.relu(self.bn1(self.l1(x)))
        h = F.relu(self.bn2(self.l2(h)))
        h = F.tanh(self.l3(h))

        return F.reshape(h, (x.shape[0], -1, self.image_width, self.image_width))

    def make_hidden(self, batchsize):

        return self.xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(self.xp.float32)


class LinearClassifier(chainer.Chain):

    def __init__(self, n_output, dropout_rate=0.5, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        super(LinearClassifier, self).__init__()
        self.dropout_rate = dropout_rate
        with self.init_scope():
            self.l1 = L.Linear(None, 1000, initialW=w)
            self.l2 = L.Linear(1000, 500, initialW=w)
            self.l3 = L.Linear(500, 250, initialW=w)
            self.l4 = L.Linear(250, n_output, initialW=w)

    def __call__(self, x):

        h = F.reshape(x, (x.shape[0], -1))
        h = self.l1(h)
        h = self.l2(F.softplus(h))
        h = self.l3(F.softplus(h))
        self.feature = F.softplus(h)
        h = self.l4(self.feature)

        return h
