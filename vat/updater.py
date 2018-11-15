import chainer
import chainer.functions as F
import warnings
import os
import sys
from common import distance
# import sys
sys.path.append(os.getcwd())
warnings.filterwarnings('ignore')


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.net = kwargs.pop("models")
        self.method = kwargs.pop("method")
        self.epsilon = kwargs.pop("epsilon")
        # params_plan = kwargs.pop("params_plan")
        # self.lr_plan = params_plan["lr"]
        # self.beta1_plan = params_plan["beta1"]
        self.xp = self.net.xp
        self.opt_iteration = 0
        super(Updater, self).__init__(*args, **kwargs)

    def loss_label(self, y_predict, label):
        loss = F.softmax_cross_entropy(
            y_predict, self.xp.array(label).astype("i"))

        accuracy = F.accuracy(y_predict.data, self.xp.array(label))
        chainer.reporter.report({'train/acc': accuracy})

        return loss

    def loss_unlabeled(self, model, x_data, y_data):
            # Virtual adversarial training loss
        return distance.vat_loss(model, x_data, y_data.data, epsilon=self.epsilon)

    def change_optimze_param(self, optimizer):
        # for our difinition of epoch is different from original paper's one
        if self.opt_iteration > 0:
            epoch = self.opt_iteration // 400
            optimizer.hyperparam.alpha = self.lr_plan[epoch]
            optimizer.hyperparam.beta1 = self.beta1_plan[epoch]
        self.opt_iteration += 1
        return optimizer

    def update_core(self):
        optimizer = self.get_optimizer("main")
        # optimizer = self.change_optimze_param(optimizer)

        label_batch = self.get_iterator('labeled').next()
        x_labeled, true_label = zip(*label_batch)
        x_labeled = self.xp.asarray(x_labeled).astype("f")

        unlabel_batch = self.get_iterator("unlabeled").next()
        x_unlabeled = self.xp.asarray(unlabel_batch).astype("f")

        y_with_label = self.net(x_labeled)

        loss_label = self.loss_label(y_with_label, true_label)

        y_unlabeled = self.net(x_unlabeled)
        loss_lsd = self.loss_unlabeled(self.net, x_unlabeled, y_unlabeled)

        self.net.cleargrads()
        loss_label.backward()
        loss_lsd.backward()
        optimizer.update()

        chainer.reporter.report({'train/loss': loss_label + loss_lsd})
        chainer.reporter.report({'loss/classify': loss_label})
        chainer.reporter.report({'loss/lsd': loss_lsd})

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


class BaseUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.net = kwargs.pop("models")
        self.method = kwargs.pop("method")
        self.xp = self.net.xp
        super(BaseUpdater, self).__init__(*args, **kwargs)

    def loss_label(self, y_predict, label):
        loss = F.softmax_cross_entropy(
            y_predict, self.xp.array(label).astype("i"))

        accuracy = F.accuracy(y_predict.data, self.xp.array(label))
        chainer.reporter.report({'train/acc': accuracy})

        return loss

    def update_core(self):
        optimizer = self.get_optimizer("main")

        label_batch = self.get_iterator('main').next()
        x_labeled, true_label = zip(*label_batch)
        x_labeled = self.xp.asarray(x_labeled).astype("f")

        y_predict = self.net(x_labeled)

        true_label = self.xp.array(true_label).astype("int8")
        loss = F.softmax_cross_entropy(
            y_predict, true_label)

        accuracy = F.accuracy(y_predict.data, true_label)
        chainer.reporter.report({'train/acc': accuracy})

        self.net.cleargrads()
        loss.backward()
        optimizer.update()

        chainer.reporter.report({'train/loss': loss})
