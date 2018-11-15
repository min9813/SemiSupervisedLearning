import copy
from chainer.training import extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import reporter
import chainer.functions as F
from chainer import link


class Evaluator(extensions.Evaluator):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None):
        assert isinstance(iterator, iterator_module.Iterator)
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        if isinstance(target, link.Link):
            self.xp = target.xp
            target = {"main": target}
        else:
            NotImplementedError
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func

    def evaluate(self):
        iterator = self._iterators['main']
        it = copy.copy(iterator)
        summary = reporter.DictSummary()
        observation = {}
        iter_times = 0
        loss = 0
        accuracy = 0
        for batch in it:
            x_data, true_label = zip(*batch)
            x_data = self.xp.asarray(x_data).astype("f")
            y_data = self.xp.asarray(true_label).astype("int8")

            y_predict = self._targets["main"](x_data).data

            loss += F.softmax_cross_entropy(y_predict, y_data)
            accuracy += F.accuracy(y_predict, y_data)

            iter_times += 1

        observation["val/loss"] = loss / iter_times
        observation["val/acc"] = accuracy / iter_times

        reporter.report(observation)

        summary.add(observation)

        return summary.compute_mean()
