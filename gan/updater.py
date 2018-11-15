import chainer
import chainer.functions as F
import warnings
import os
import sys
# import sys
sys.path.append(os.getcwd())
warnings.filterwarnings('ignore')


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        net = kwargs.pop("models")
        self.gen = net["gen"]
        self.dis = net["dis"]
        self.method = kwargs.pop("method")
        self.xp = self.gen.xp
        super(Updater, self).__init__(*args, **kwargs)
        self.iteration = 0

    def loss_label(self, y_predict, label):
        loss = F.softmax_cross_entropy(
            y_predict, label)

        accuracy = F.accuracy(y_predict.data, label)
        chainer.reporter.report({'train/acc': accuracy})
        chainer.reporter.report({'train/loss': loss})

        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        label_batch = self.get_iterator('labeled').next()
        x_labeled, true_label = zip(*label_batch)
        x_labeled = self.xp.asarray(x_labeled).astype("f")
        true_label = self.xp.asarray(true_label).astype("i")

        unlabel_batch = self.get_iterator("unlabeled").next()
        batchsize = len(unlabel_batch)
        x_unlabeled = self.xp.asarray(unlabel_batch).astype("f")

        y_with_label = self.dis(x_labeled)
        y_with_unlabel = self.dis(x_unlabeled)

        # real_feature = self.dis.feature.data

        z = self.gen.make_hidden(batchsize)
        x_fake = self.gen(z)
        y_fake = self.dis(x_fake)

        # fake_feature = self.dis.feature
        log_sum_y_fake = F.logsumexp(y_fake, axis=1)
        # loss_gen = F.mean(F.softplus(log_sum_y_fake))
        loss_gen_softplus = F.softplus(log_sum_y_fake)
        loss_gen = -F.mean(log_sum_y_fake - loss_gen_softplus)
        # loss_feature = F.mean_squared_error(fake_feature, real_feature)
        self.gen.cleargrads()
        loss_gen.backward()
        # loss_feature.backward()
        gen_optimizer.update()
        chainer.reporter.report({'gen/loss': loss_gen})
        # chainer.reporter.report({'gen/loss_f': loss_feature})

        log_sum_y_real = F.logsumexp(y_with_unlabel, axis=1)
        loss_classify = self.loss_label(y_with_label, true_label)
        loss_unlabel = log_sum_y_real - F.softplus(log_sum_y_real)
        z = self.gen.make_hidden(batchsize)
        x_fake = self.gen(z)
        y_fake = self.dis(x_fake.data)
        loss_from_gen = F.logsumexp(y_fake, axis=1)
        loss_from_gen = F.softplus(loss_from_gen)
        loss_dis = F.mean(- loss_unlabel + loss_from_gen)
        self.dis.cleargrads()
        loss_dis.backward()
        loss_classify.backward()
        dis_optimizer.update()

        chainer.reporter.report({'dis/loss': loss_dis})

    @property
    def epoch_detail(self):
        return self._iterators['unlabeled'].epoch_detail

    @property
    def epoch(self):
        return self._iterators['unlabeled'].epoch

    @property
    def previous_epoch_detail(self):
        return self._iterators['unlabeled'].previous_epoch_detail

    @property
    def is_new_epoch(self):
        return self._iterators['unlabeled'].is_new_epoch
